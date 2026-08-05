import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from src.dependencies import (
    get_db,
    get_langfeather_runtime,
    get_rag_graph,
)
from src.langfeather_runtime import LangFeatherRuntime
from src.rag.graph import PolicyRagGraph
from src.session.models import (
    PROFILE_FIELD_NAMES,
    AnonymousSession,
)
from src.session.schemas import (
    AnonymousSessionCreate,
    ConversationSnapshot,
    PublicProfile,
    SessionProfileUpdate,
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
    reset_conversation,
    update_session_profile,
)


logger = logging.getLogger(__name__)

session_router = APIRouter(tags=["web"])


def _public_profile(session: AnonymousSession) -> PublicProfile:
    return PublicProfile.model_validate(
        session.model_dump(
            include=PROFILE_FIELD_NAMES
        )
    )


def _session_status(
    session: AnonymousSession,
) -> SessionStatus:
    return SessionStatus(
        expires_at=as_utc(session.expires_at),
        profile=_public_profile(session),
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
    profile_update = SessionProfileUpdate.model_validate(
        payload.model_dump(exclude={"accepted_storage"})
    )

    if existing and not is_expired(existing):
        session = update_session_profile(existing, profile_update, db)
        session = touch_session(session, db)
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if token:
            set_session_cookie(response, token)
        return _session_status(session)

    token, session = create_session(profile_update, db)
    set_session_cookie(response, token)
    return _session_status(session)


@session_router.get(
    "/sessions/current",
    response_model=SessionStatus,
)
def get_session_status(
    request: Request,
    response: Response,
    session: AnonymousSession = Depends(get_current_session),
) -> SessionStatus:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        set_session_cookie(response, token)
    return _session_status(session)


@session_router.patch(
    "/me/profile",
    response_model=PublicProfile,
)
def update_profile(
    payload: SessionProfileUpdate,
    session: AnonymousSession = Depends(get_current_session),
    db: Session = Depends(get_db),
) -> PublicProfile:
    session = update_session_profile(session, payload, db)
    return _public_profile(session)


@session_router.get(
    "/me/conversation",
    response_model=ConversationSnapshot,
)
async def get_conversation(
    session: AnonymousSession = Depends(get_current_session),
    rag: PolicyRagGraph = Depends(get_rag_graph),
) -> ConversationSnapshot:
    return ConversationSnapshot.model_validate(
        await rag.get_conversation(session.thread_id)
    )


@session_router.post("/me/chat")
async def stream_session_answer(
    payload: WebChatRequest,
    session: AnonymousSession = Depends(get_current_session),
    rag: PolicyRagGraph = Depends(get_rag_graph),
    langfeather_runtime: LangFeatherRuntime = Depends(get_langfeather_runtime),
) -> StreamingResponse:
    # stream_answer는 async generator라 호출만으로는 본문이 실행되지 않는다.
    # 여기서 잡히는 것은 스트림 시작 전 실패뿐이고(아직 응답 헤더가 안 나갔으므로
    # 500으로 알릴 수 있다), 스트리밍 중 실패는 stream_answer 안에서
    # error 이벤트로 처리한다.
    trace_id = None
    try:
        rag_user_profile = session.model_dump(
            include=PROFILE_FIELD_NAMES
        )
        trace_id = langfeather_runtime.create_trace_id()
        generator = rag.stream_answer(
            user_profile=rag_user_profile,
            user_input=payload.user_input,
            exclude_expired=payload.exclude_expired,
            thread_id=session.thread_id,
            trace_id=trace_id,
            trace_metadata=langfeather_runtime.trace_metadata(trace_id),
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
        logger.exception(
            "chat 스트림 시작 실패 trace_id=%s thread_id=%s",
            trace_id,
            session.thread_id,
        )
        raise HTTPException(
            status_code=500,
            detail="LLM 답변 생성 오류",
        ) from error


@session_router.post("/me/feedback")
def submit_user_feedback(
    payload: UserFeedbackRequest,
    session: AnonymousSession = Depends(get_current_session),
    langfeather_runtime: LangFeatherRuntime = Depends(get_langfeather_runtime),
) -> dict[str, str]:
    recorded = False
    errors: list[RuntimeError] = []
    if langfeather_runtime.enabled:
        try:
            langfeather_runtime.record_user_feedback(
                trace_id=payload.trace_id,
                helpful=payload.helpful,
                reason=payload.reason,
                comment=payload.comment,
                anonymous_user_id=session.thread_id,
            )
            recorded = True
        except RuntimeError as error:
            errors.append(error)
        except Exception:
            logger.exception(
                "피드백 저장 실패 trace_id=%s",
                payload.trace_id,
            )
            errors.append(RuntimeError("피드백 저장소에 연결할 수 없습니다."))
    if not recorded:
        message = str(errors[-1]) if errors else "피드백 수집이 비활성화되어 있습니다."
        raise HTTPException(status_code=503, detail=message)
    return {"message": "피드백이 저장되었습니다."}


@session_router.delete("/me/conversation")
def delete_conversation(
    session: AnonymousSession = Depends(get_current_session),
    db: Session = Depends(get_db),
    rag: PolicyRagGraph = Depends(get_rag_graph),
) -> dict[str, str]:
    old_thread_id = reset_conversation(session, db)
    rag.delete_conversation(old_thread_id)
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
