import asyncio

from sqlmodel import Session

from src.database import engine
from src.rag.graph import PolicyRagGraph
from src.session.service import cleanup_expired_sessions


SESSION_CLEANUP_INTERVAL_SECONDS = 60 * 60


def cleanup_expired_sessions_once(rag: PolicyRagGraph) -> int:
    with Session(engine) as db:
        return cleanup_expired_sessions(db, rag)


async def run_expired_session_cleanup(rag: PolicyRagGraph) -> None:
    while True:
        await asyncio.to_thread(cleanup_expired_sessions_once, rag)
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)
