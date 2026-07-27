from fastapi import Request

from src.database import get_session
from src.langfeather_runtime import LangFeatherRuntime
from src.observability import ObservabilityRuntime
from src.rag.graph import PolicyRagGraph


def get_rag_graph(request: Request) -> PolicyRagGraph:
    return request.app.state.rag_graph


def get_observability(request: Request) -> ObservabilityRuntime:
    return request.app.state.observability


def get_langfeather_runtime(request: Request) -> LangFeatherRuntime:
    return request.app.state.langfeather_runtime


def get_db():
    yield from get_session()
