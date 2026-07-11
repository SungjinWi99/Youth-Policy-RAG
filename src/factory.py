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
from src.config import AppConfig
from src.rag.graph import PolicyRagGraph
from src.rag.nodes.retriever import PolicyRetriever
from src.rag.nodes.agent import PolicyAgent
from src.rag.nodes.turn_planner import TurnPlanner


CHAT_MODEL_CLASSES = {
    "google": ChatGoogleGenerativeAI,
    "openai": ChatOpenAI,
    "upstage": ChatUpstage,
    "anthropic": ChatAnthropic,
    "ollama": ChatOllama,
    "deepseek": ChatDeepSeek,
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


def build_rag_graph(config: AppConfig) -> PolicyRagGraph:
    embeddings = create_embedding_model(
        provider=config.retriever.provider,
        model_name=config.retriever.query_model,
    )
    vector_store = Chroma(
        collection_name=config.data.chroma_collection_name,
        persist_directory=config.path(config.data.chroma_dir),
        embedding_function=embeddings,
    )
    retriever = PolicyRetriever(
        vector_store=vector_store,
        search_k=config.retriever.search_k,
    )

    llm = create_chat_model(
        provider=config.llm.provider,
        model_name=config.llm.model,
    )
    agent = PolicyAgent(llm)
    planner = TurnPlanner(llm)

    checkpointer = create_sqlite_checkpointer(
        config.path(config.data.conversation_db)
    )

    return PolicyRagGraph(
        planner=planner,
        retriever=retriever,
        agent=agent,
        checkpointer=checkpointer,
        planner_history_window=config.rag.planner.history_window,
        agent_history_window=config.rag.agent.history_window,
    )
