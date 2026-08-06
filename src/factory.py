import logging

import anthropic
import ollama
import openai
from google.genai import errors as genai_errors
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_deepseek import ChatDeepSeek

from src.checkpointer import create_sqlite_checkpointer
from src.config import AppConfig, LLMConfig, LLMProviderConfig
from src.rag.graph import PolicyRagGraph
from src.rag.nodes import (
    make_answer_generator_node,
    make_policy_checker_node,
    make_policy_selector_node,
    make_retrieval_planner_node,
    make_retriever_node,
)
from src.rag.retrievers import (
    BM25PolicyRetriever,
    DensePolicyRetriever,
    EnsemblePolicyRetriever,
)

logger = logging.getLogger(__name__)

CHAT_MODEL_CLASSES = {
    "google": ChatGoogleGenerativeAI,
    "openai": ChatOpenAI,
    "upstage": ChatUpstage,
    "anthropic": ChatAnthropic,
    "ollama": ChatOllama,
    "deepseek": ChatDeepSeek,
}

# provider-side failures worth retrying (timeout/5xx/rate-limit), never
# client errors like a bad prompt (400) or bad credentials (401)
_OPENAI_COMPATIBLE_RETRYABLE = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)
RETRYABLE_EXCEPTIONS = {
    "openai": _OPENAI_COMPATIBLE_RETRYABLE,
    "upstage": _OPENAI_COMPATIBLE_RETRYABLE,
    "deepseek": _OPENAI_COMPATIBLE_RETRYABLE,
    "anthropic": (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
    ),
    "google": (genai_errors.ServerError,),
    # ponytail: ollama's SDK has no 4xx/5xx split like the others, so this
    # also retries on bad requests; narrow it with a status_code check if
    # that turns out to be wasteful in practice
    "ollama": (ollama.RequestError, ollama.ResponseError),
}

EMBEDDING_MODEL_CLASSES = {
    "google": GoogleGenerativeAIEmbeddings,
    "openai": OpenAIEmbeddings,
    "upstage": UpstageEmbeddings,
    "ollama": OllamaEmbeddings
}


def create_chat_model(provider: str, model_name: str, **kwargs):
    try:
        model_class = CHAT_MODEL_CLASSES[provider]
    except KeyError as error:
        supported = ", ".join(sorted(CHAT_MODEL_CLASSES))
        raise ValueError(
            f"지원하지 않는 chat provider입니다: {provider}. "
            f"지원 provider: {supported}"
        ) from error
    if provider == "deepseek":
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        extra_body.setdefault("thinking", {"type": "disabled"})
        kwargs["extra_body"] = extra_body
    return model_class(model=model_name, **kwargs)


def create_chat_model_with_retry(provider_config: LLMProviderConfig, **kwargs):
    llm = create_chat_model(
        provider=provider_config.provider,
        model_name=provider_config.model,
        **kwargs,
    )
    return llm.with_retry(
        retry_if_exception_type=RETRYABLE_EXCEPTIONS[provider_config.provider],
        stop_after_attempt=provider_config.max_retries,
        exponential_jitter_params={
            "initial": provider_config.retry_wait_initial,
            "max": provider_config.retry_wait_max,
        },
    )


def create_chat_model_with_fallback(config: LLMConfig, **kwargs):
    llm = create_chat_model_with_retry(config.main, **kwargs)
    if config.fallbacks:
        llm = llm.with_fallbacks(
            [create_chat_model_with_retry(fallback, **kwargs) for fallback in config.fallbacks]
        )
    return llm


def create_embedding_model(provider: str, model_name: str, **kwargs):
    try:
        model_class = EMBEDDING_MODEL_CLASSES[provider]
    except KeyError as error:
        supported = ", ".join(sorted(EMBEDDING_MODEL_CLASSES))
        raise ValueError(
            f"지원하지 않는 embedding provider입니다: {provider}. "
            f"지원 provider: {supported}"
        ) from error
    return model_class(model=model_name, **kwargs)


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def create_passage_embedding_model(
    provider: str,
    model_name: str,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
):
    kwargs = {}
    if provider == "ollama":
        kwargs["base_url"] = ollama_base_url
    return create_embedding_model(
        provider=provider,
        model_name=model_name,
        **kwargs,
    )


EMBEDDING_PROVIDER_KEY = "embedding_provider"
EMBEDDING_PASSAGE_MODEL_KEY = "embedding_passage_model"


def verify_embedding_consistency(
    vector_store,
    *,
    provider: str,
    passage_model: str,
) -> None:

    collection = getattr(vector_store, "_collection", None)
    if collection is None:
        logger.warning(
            "Chroma 내부 컬렉션에 접근할 수 없어 임베딩 정합성 검증을 스킵합니다."
        )
        return

    metadata = getattr(collection, "metadata", None) or {}
    recorded_provider = metadata.get(EMBEDDING_PROVIDER_KEY)
    recorded_model = metadata.get(EMBEDDING_PASSAGE_MODEL_KEY)
    if not recorded_provider or not recorded_model:
        logger.warning(
            "적재 정보가 없는 레거시 Chroma 인덱스입니다. 현재 설정"
            "(provider=%s, passage_model=%s)과 실제 적재 조합이 같은지 "
            "검증할 수 없습니다.",
            provider,
            passage_model,
        )
        return

    if recorded_provider != provider or recorded_model != passage_model:
        raise RuntimeError(
            "임베딩 조합이 Chroma 인덱스와 일치하지 않습니다: "
            f"인덱스={recorded_provider}/{recorded_model}, "
            f"설정={provider}/{passage_model}. "
            "config.yaml의 retriever.provider·retriever.passage_model과 "
            "data.chroma_dir이 같은 적재를 가리키는지 확인하세요."
        )


def build_rag_graph(
    config: AppConfig,
) -> PolicyRagGraph:
    embeddings = create_embedding_model(
        provider=config.retriever.provider,
        model_name=config.retriever.query_model,
    )
    vector_store = Chroma(
        collection_name=config.data.chroma_collection_name,
        persist_directory=config.path(config.data.chroma_dir),
        embedding_function=embeddings,
    )
    verify_embedding_consistency(
        vector_store,
        provider=config.retriever.provider,
        passage_model=config.retriever.passage_model,
    )
    dense_retriever = DensePolicyRetriever(
        vector_store=vector_store,
        search_k=(
            config.retriever.dense_candidate_k
            if config.retriever.mode == "hybrid"
            else config.retriever.search_k
        ),
    )
    bm25_retriever = None
    if config.retriever.mode == "hybrid":
        bm25_retriever = BM25PolicyRetriever(
            collection=vector_store,
            search_k=config.retriever.bm25_candidate_k,
        )
        policy_retriever = EnsemblePolicyRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[
                config.retriever.hybrid_dense_weight,
                1 - config.retriever.hybrid_dense_weight,
            ],
            search_k=config.retriever.search_k,
            rrf_k=config.retriever.hybrid_rrf_k,
        )
    else:
        policy_retriever = dense_retriever

    llm = create_chat_model_with_fallback(config.llm)
    checkpointer = create_sqlite_checkpointer(
        config.path(config.data.conversation_db)
    )
    return PolicyRagGraph(
        retrieval_planner=make_retrieval_planner_node(
            llm,
            config.rag.planner.history_window,
        ),
        retriever=make_retriever_node(policy_retriever),
        policy_checker=make_policy_checker_node(llm),
        policy_selector=make_policy_selector_node(),
        answer_generator=make_answer_generator_node(
            llm,
            config.rag.answer_generator.history_window,
        ),
        checkpointer=checkpointer,
        max_retrieval_retries=config.rag.policy_checker.max_retries,
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
    )
