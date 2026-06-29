from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import ChatUpstage, UpstageEmbeddings

from src.chat.rag import RAGPipeline
from src.config import AppConfig


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


def build_rag_pipeline(config: AppConfig) -> RAGPipeline:
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
    return RAGPipeline(
        llm=llm,
        vector_store=vector_store,
        search_k=config.retriever.search_k,
    )
