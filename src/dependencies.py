from fastapi import Request

from src.database import get_session
from src.langfeather_runtime import LangFeatherRuntime
from src.rag.graph import PolicyRagGraph
from src.session.rate_limit import TokenBucketLimiter


def get_rag_graph(request: Request) -> PolicyRagGraph:
    return request.app.state.rag_graph


def get_langfeather_runtime(request: Request) -> LangFeatherRuntime:
    return request.app.state.langfeather_runtime


def get_chat_rate_limiter(request: Request) -> TokenBucketLimiter:
    return request.app.state.chat_rate_limiter


def get_session_create_rate_limiter(request: Request) -> TokenBucketLimiter:
    return request.app.state.session_create_rate_limiter


def get_db():
    yield from get_session()
