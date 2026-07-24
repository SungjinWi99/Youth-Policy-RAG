from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from src.chat.models import ConversationThread
from src.dependencies import get_db, get_observability, get_rag_graph
from src.observability import ObservabilityRuntime
from src.rag.graph import PolicyRagGraph
from src.session.models import AnonymousSession
from src.session.schemas import (
    AnonymousSessionCreate,
    ConversationSnapshot,
    PublicProfile,
    SessionStatus,
    UserFeedbackRequest,
    WebChatRequest,
)
from src.session.service import (
    SESSION_COOKIE_NAME,
    as_utc,
    cleanup_expired_sessions,
    clear_session_cookie,
    create_session,
    delete_session_data,
    find_session,
    get_current_session,
    is_expired,
    set_session_cookie,
    touch_session,
)
from src.user.models import UserProfile
from src.user.schemas import UserUpdate


session_router = APIRouter(tags=["web"])


def _public_profile(profile: UserProfile) -> PublicProfile:
    return PublicProfile.model_validate(
        profile.model_dump(
            include={"age", "gender", "job", "income", "region"}
        )
    )


def _session_status(
    session: AnonymousSession,
    profile: UserProfile,
) -> SessionStatus:
    return SessionStatus(
        expires_at=as_utc(session.expires_at),
        profile=_public_profile(profile),
    )


@session_router.post(
    "/sessions/anonymous",
    response_model=SessionStatus,
    status_code=201,
)
def start_anonymous_session(
    payload: AnonymousSessionCreate,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    rag: PolicyRagGraph = Depends(get_rag_graph),
) -> SessionStatus:
    cleanup_expired_sessions(db, rag)
    existing = find_session(request, db)
    profile_update = UserUpdate.model_validate(
        payload.model_dump(exclude={"accepted_storage"})
    )

    if existing and not is_expired(existing):
        profile = UserProfile.update(existing.user_id, profile_update, db)
        touch_session(existing, db)
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if token:
            set_session_cookie(response, token)
        return _session_status(existing, profile)

    token, session, profile = create_session(profile_update, db)
    set_session_cookie(response, token)
    return _session_status(session, profile)


@session_router.get(
    "/sessions/current",
    response_model=SessionStatus,
)
def get_session_status(
    request: Request,
    response: Response,
    session: AnonymousSession = Depends(get_current_session),
    db: Session = Depends(get_db),
) -> SessionStatus:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        set_session_cookie(response, token)
    profile = UserProfile.get(session.user_id, db)
    return _session_status(session, profile)


@session_router.patch(
    "/me/profile",
    response_model=PublicProfile,
)
def update_profile(
    payload: UserUpdate,
    session: AnonymousSession = Depends(get_current_session),
    db: Session = Depends(get_db),
) -> PublicProfile:
    profile = UserProfile.update(session.user_id, payload, db)
    return _public_profile(profile)


@session_router.get(
    "/me/conversation",
    response_model=ConversationSnapshot,
)
async def get_conversation(
    session: AnonymousSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    rag: PolicyRagGraph = Depends(get_rag_graph),
) -> ConversationSnapshot:
    conversation = db.get(ConversationThread, session.user_id)
    if not conversation:
        return ConversationSnapshot()
    return ConversationSnapshot.model_validate(
        await rag.get_conversation(conversation.thread_id)
    )


@session_router.post("/me/chat")
async def stream_session_answer(
    payload: WebChatRequest,
    session: AnonymousSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    rag: PolicyRagGraph = Depends(get_rag_graph),
    observability: ObservabilityRuntime = Depends(get_observability),
) -> StreamingResponse:
    try:
        profile = UserProfile.get(session.user_id, db)
        rag_user_profile = profile.model_dump(
            include={"age", "gender", "job", "income", "region"}
        )
        thread_id = ConversationThread.get_thread_id(session.user_id, db)
        trace_id = observability.create_trace_id()
        generator = rag.stream_answer(
            user_profile=rag_user_profile,
            user_input=payload.user_input,
            exclude_expired=payload.exclude_expired,
            thread_id=thread_id,
            trace_user_id=session.user_id,
            trace_id=trace_id,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except HTTPException:
        raise
    except Exception as error:
        print(error)
        raise HTTPException(
            status_code=500,
            detail="LLM 답변 생성 오류",
        ) from error


@session_router.post("/me/feedback")
def submit_user_feedback(
    payload: UserFeedbackRequest,
    session: AnonymousSession = Depends(get_current_session),
    observability: ObservabilityRuntime = Depends(get_observability),
) -> dict[str, str]:
    try:
        observability.record_user_feedback(
            trace_id=payload.trace_id,
            helpful=payload.helpful,
            reason=payload.reason,
            comment=payload.comment,
            anonymous_user_id=session.user_id,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        print(error)
        raise HTTPException(
            status_code=502,
            detail="피드백 저장소에 연결할 수 없습니다.",
        ) from error
    return {"message": "피드백이 저장되었습니다."}


@session_router.delete("/me/conversation")
def delete_conversation(
    session: AnonymousSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    rag: PolicyRagGraph = Depends(get_rag_graph),
) -> dict[str, str]:
    old_thread_id, _ = ConversationThread.reset_thread_id(
        session.user_id,
        db,
    )
    rag.delete_conversation(old_thread_id)
    if old_thread_id != session.user_id:
        rag.delete_conversation(session.user_id)
    return {"message": "대화 기록 삭제 완료"}


@session_router.delete("/me/data")
def delete_all_my_data(
    response: Response,
    session: AnonymousSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    rag: PolicyRagGraph = Depends(get_rag_graph),
) -> dict[str, str]:
    delete_session_data(session, db, rag)
    clear_session_cookie(response)
    return {"message": "프로필과 상담 기록 삭제 완료"}
