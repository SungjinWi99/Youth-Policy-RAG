from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langgraph.prebuilt import ToolNode

from src.checkpoint import create_sqlite_checkpointer
from src.config import AppConfig
from src.rag.agent import PolicyAgent
from src.rag.graph import RAGGraph
from src.rag.retriever import PolicyRetriever
from src.rag.summarizer import ConversationSummarizer
from src.rag.tools import create_search_policies_tool


CHAT_MODEL_CLASSES = {
    "google": ChatGoogleGenerativeAI,
    "openai": ChatOpenAI,
    "upstage": ChatUpstage,
}

EMBEDDING_MODEL_CLASSES = {
    "google": GoogleGenerativeAIEmbeddings,
    "openai": OpenAIEmbeddings,
    "upstage": UpstageEmbeddings,
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


def build_rag_graph(config: AppConfig) -> RAGGraph:
    embeddings = create_embedding_model(
        provider=config.retriever.provider,
        model_name=config.retriever.query_model,
    )
    vector_store = Chroma(
        collection_name=config.data.chroma_collection_name,
        persist_directory=config.path(config.data.chroma_dir),
        embedding_function=embeddings,
    )
    llm = create_chat_model(
        provider=config.llm.provider,
        model_name=config.llm.model,
    )
    checkpointer = create_sqlite_checkpointer(
        config.path(config.data.conversation_db)
    )
    retriever = PolicyRetriever(
        vector_store=vector_store,
        search_k=config.retriever.search_k,
    )
    summarizer = ConversationSummarizer(
        llm,
        max_input_tokens=config.llm.max_input_tokens,
        summary_trigger_ratio=config.llm.summary_trigger_ratio,
        keep_recent_turns=config.llm.summary_keep_recent_turns,
        chars_per_token=config.llm.token_chars_per_token,
    )
    search_policies_tool = create_search_policies_tool(retriever)
    agent = PolicyAgent(llm, [search_policies_tool])
    tool_node = ToolNode(
        [search_policies_tool],
        handle_tool_errors=(
            "정책 검색 중 오류가 발생했습니다. "
            "현재 문서 범위에서 답변하세요."
        ),
    )
    return RAGGraph(
        summarizer=summarizer,
        agent=agent,
        tool_node=tool_node,
        checkpointer=checkpointer,
    )
