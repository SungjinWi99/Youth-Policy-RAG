from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from src.dependencies import get_rag_service, get_db
from src.chat.rag import RAGPipeline
from src.chat.schemas import ChatRequest
from src.user.models import UserProfile


chat_router = APIRouter(tags=['youth_policies'])

@chat_router.post("/chat")
async def stream_answer(request: ChatRequest,
                        rag: RAGPipeline = Depends(get_rag_service),
                        db: Session = Depends(get_db)):
  try:
    user_profile = UserProfile.get(request.user_id, db)
    generator = rag.stream_answer(user_profile=user_profile,
                                  user_input=request.user_input,
                                  exclude_expired=request.exclude_expired)
    return StreamingResponse(generator, media_type='text/event-stream')
  except Exception as e:
    print(e)
    raise HTTPException(
        status_code=500,
        detail="LLM 답변 생성 오류"
    )
