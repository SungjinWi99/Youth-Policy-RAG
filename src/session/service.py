import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, Response
from sqlmodel import Session, select

from src.chat.models import ConversationThread
from src.dependencies import get_db
from src.rag.graph import PolicyRagGraph
from src.session.models import AnonymousSession, utc_now
from src.user.models import UserProfile
from src.user.schemas import UserBase


SESSION_COOKIE_NAME = "youth_policy_session"
SESSION_RETENTION_DAYS = 30
SESSION_MAX_AGE_SECONDS = SESSION_RETENTION_DAYS * 24 * 60 * 60


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def cookie_secure() -> bool:
    value = os.getenv("SESSION_COOKIE_SECURE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=cookie_secure(),
        httponly=True,
        samesite="lax",
    )


def find_session(request: Request, db: Session) -> AnonymousSession | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    return db.get(AnonymousSession, _token_hash(token))


def is_expired(session: AnonymousSession, now: datetime | None = None) -> bool:
    return as_utc(session.expires_at) <= (now or utc_now())


def touch_session(session: AnonymousSession, db: Session) -> AnonymousSession:
    now = utc_now()
    session.time_updated = now
    session.expires_at = now + timedelta(days=SESSION_RETENTION_DAYS)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def create_session(
    profile_data: UserBase,
    db: Session,
) -> tuple[str, AnonymousSession, UserProfile]:
    token = secrets.token_urlsafe(32)
    now = utc_now()
    user_id = f"anon_{uuid4().hex}"
    profile = UserProfile(
        user_id=user_id,
        **profile_data.model_dump(exclude_none=True),
    )
    session = AnonymousSession(
        token_hash=_token_hash(token),
        user_id=user_id,
        time_created=now,
        time_updated=now,
        expires_at=now + timedelta(days=SESSION_RETENTION_DAYS),
    )
    db.add(profile)
    db.add(session)
    db.commit()
    db.refresh(profile)
    db.refresh(session)
    return token, session, profile


def delete_session_data(
    session: AnonymousSession,
    db: Session,
    rag: PolicyRagGraph | None,
) -> None:
    user_id = session.user_id
    conversation = db.get(ConversationThread, user_id)
    thread_id = conversation.thread_id if conversation else user_id
    profile = db.get(UserProfile, user_id)

    if conversation:
        db.delete(conversation)
    if profile:
        db.delete(profile)
    db.delete(session)
    db.commit()

    if rag:
        rag.delete_conversation(thread_id)
        if thread_id != user_id:
            rag.delete_conversation(user_id)


def cleanup_expired_sessions(
    db: Session,
    rag: PolicyRagGraph | None,
) -> int:
    now = utc_now()
    sessions = db.exec(select(AnonymousSession)).all()
    expired = [session for session in sessions if is_expired(session, now)]
    for session in expired:
        delete_session_data(session, db, rag)
    return len(expired)


def get_current_session(
    request: Request,
    db: Session = Depends(get_db),
) -> AnonymousSession:
    session = find_session(request, db)
    if session is None:
        raise HTTPException(status_code=401, detail="상담 세션이 필요합니다.")

    if is_expired(session):
        rag = getattr(request.app.state, "rag_graph", None)
        delete_session_data(session, db, rag)
        raise HTTPException(
            status_code=401,
            detail="상담 세션이 만료되었습니다.",
        )

    return touch_session(session, db)
