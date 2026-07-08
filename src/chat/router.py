from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from src.dependencies import get_rag_graph, get_db
from src.chat.models import ConversationThread
from src.chat.schemas import ChatRequest
from src.rag.graph import PolicyRagGraph
from src.user.models import UserProfile


chat_router = APIRouter(tags=['youth_policies'])

@chat_router.post("/chat")
async def stream_answer(request: ChatRequest,
                        rag: PolicyRagGraph = Depends(get_rag_graph),
                        db: Session = Depends(get_db)):
  try:
    user_profile = UserProfile.get(request.user_id, db)
    rag_user_profile = user_profile.model_dump(
      include={"age", "gender", "job", "income", "region"}
    )
    thread_id = ConversationThread.get_thread_id(request.user_id, db)
    generator = rag.stream_answer(user_profile=rag_user_profile,
                                  user_input=request.user_input,
                                  exclude_expired=request.exclude_expired,
                                  user_id=thread_id)
    return StreamingResponse(generator, media_type='text/event-stream')
  except HTTPException:
    raise
  except Exception as e:
    print(e)
    raise HTTPException(
        status_code=500,
        detail="LLM 답변 생성 오류"
    )


@chat_router.delete("/chat/{user_id}")
def delete_chat_history(
    user_id: str,
    rag: PolicyRagGraph = Depends(get_rag_graph),
    db: Session = Depends(get_db),
):
  old_thread_id, _new_thread_id = ConversationThread.reset_thread_id(user_id, db)
  rag.delete_conversation(old_thread_id)
  if old_thread_id != user_id:
    rag.delete_conversation(user_id)
  return {"message": "대화 기록 삭제 완료"}
