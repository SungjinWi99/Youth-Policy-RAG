from fastapi import Depends, APIRouter, HTTPException
from sqlmodel import Session

from src.chat.models import ConversationThread
from src.rag.graph import PolicyRagGraph
from src.user.models import UserProfile
from src.user.schemas import UserCreate, UserUpdate
from src.dependencies import get_db, get_rag_graph

user_router = APIRouter(tags=['youth_policies'])

@user_router.post("/user/registration")
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
  return UserProfile.create(user_data, db)

@user_router.get("/user/{user_id}")
def get_user_profile(user_id: str, db: Session = Depends(get_db)):
  return UserProfile.get(user_id, db)

@user_router.post("/user/{user_id}")
def update_user_profile(user_id: str, user_data: UserUpdate, db: Session = Depends(get_db)):
  return UserProfile.update(user_id, user_data, db)

@user_router.delete("/user/{user_id}")
def delete_user_profile(
    user_id: str,
    db: Session = Depends(get_db),
    rag: PolicyRagGraph = Depends(get_rag_graph),
):
  thread_id = ConversationThread.delete_for_user(user_id, db)
  result = UserProfile.delete(user_id, db)
  rag.delete_conversation(thread_id)
  if thread_id != user_id:
    rag.delete_conversation(user_id)
  return result
