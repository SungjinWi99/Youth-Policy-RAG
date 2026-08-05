import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from src.policy.router import policy_router
from src.session.router import session_router
from src.config import load_config
from src.database import create_db_and_tables
from src.factory import build_rag_graph
from src.langfeather_runtime import create_langfeather_runtime
from src.rag.retrievers import run_bm25_refresh
from src.session.cleanup import run_expired_session_cleanup

load_dotenv()
config = load_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    langfeather_runtime = create_langfeather_runtime()
    rag_graph = None
    cleanup_task = None
    bm25_refresh_task = None
    try:
        create_db_and_tables()
        rag_graph = build_rag_graph(config)
        rag_graph.graph = langfeather_runtime.wrap_graph(rag_graph.graph)
        app.state.rag_graph = rag_graph
        app.state.langfeather_runtime = langfeather_runtime
        cleanup_task = asyncio.create_task(
            run_expired_session_cleanup(rag_graph)
        )
        if rag_graph.bm25_retriever is not None:
            bm25_refresh_task = asyncio.create_task(
                run_bm25_refresh(
                    rag_graph.bm25_retriever,
                    Path(config.path(config.data.raw)),
                )
            )
        yield
    finally:
        for task in (cleanup_task, bm25_refresh_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if rag_graph is not None:
            rag_graph.close()
        langfeather_runtime.shutdown()

app = FastAPI(title="청년정책 RAG API", lifespan=lifespan)
app.include_router(policy_router)
app.include_router(session_router)
