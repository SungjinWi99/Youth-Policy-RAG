import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.policy.router import policy_router
from src.session.router import session_router
from src.config import load_config
from src.database import create_db_and_tables
from src.factory import build_rag_graph
from src.langfeather_runtime import create_langfeather_runtime
from src.rag.retrievers import run_bm25_refresh
from src.session.cleanup import run_expired_session_cleanup
from src.session.rate_limit import TokenBucketLimiter

load_dotenv()
config = load_config()

# uvicorn은 자기 로거를 따로 설정한다. 여기서는 root만 설정하며,
# src.* 로거가 propagate로 이 핸들러를 탄다.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

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
        # 유료 LLM API 남용 방지. 단일 워커 전제로 in-memory에 둔다
        # (워커를 늘리면 워커마다 카운터가 따로 생겨 무력화된다).
        app.state.chat_rate_limiter = TokenBucketLimiter(
            capacity=5, refill_period_seconds=30
        )
        app.state.session_create_rate_limiter = TokenBucketLimiter(
            capacity=5, refill_period_seconds=720
        )
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


@app.get("/health")
def health() -> JSONResponse:
    # 로컬 디스크 읽기만 한다. LLM·임베딩 provider를 찌르면 provider 지연이
    # 컨테이너 재시작으로 번진다(ISSUE-002에서 A안을 기각한 것과 같은 이유).
    # 잡으려는 것은 배포 직후 사고 — data 볼륨 미마운트, 빈 인덱스.
    rag_graph = getattr(app.state, "rag_graph", None)
    try:
        count = rag_graph.vector_store._collection.count()
    except Exception:
        logger.exception("health check 실패: Chroma 컬렉션에 접근할 수 없습니다.")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": "chroma_unavailable"},
        )
    if count == 0:
        logger.error("health check 실패: Chroma 컬렉션이 비어 있습니다.")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": "chroma_empty"},
        )
    return JSONResponse(content={"status": "ok", "documents": count})
