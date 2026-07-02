from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from src.dependencies import get_rag_graph, get_db
from src.chat.schemas import (
    ChatRequest,
    ConversationResetResponse,
)
from src.rag.graph import RAGGraph
from src.user.models import UserProfile


chat_router = APIRouter(tags=['youth_policies'])

@chat_router.post("/chat")
async def stream_answer(request: ChatRequest,
                        rag: RAGGraph = Depends(get_rag_graph),
                        db: Session = Depends(get_db)):
  try:
    user_profile = UserProfile.get(request.user_id, db)
    generator = rag.stream_answer(user_profile=user_profile,
                                  user_input=request.user_input,
                                  exclude_expired=request.exclude_expired)
    return StreamingResponse(generator, media_type='text/event-stream')
  except HTTPException:
    raise
  except Exception as e:
    print(e)
    raise HTTPException(
        status_code=500,
        detail="LLM 답변 생성 오류"
    )


@chat_router.delete(
    "/chat/history/{user_id}",
    response_model=ConversationResetResponse,
)
def reset_conversation(
    user_id: str,
    rag: RAGGraph = Depends(get_rag_graph),
    db: Session = Depends(get_db),
) -> ConversationResetResponse:
  UserProfile.get(user_id, db)
  rag.delete_conversation(user_id)
  return ConversationResetResponse(
      user_id=user_id,
      message="대화 기록이 초기화되었습니다.",
  )
