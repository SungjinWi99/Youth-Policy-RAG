from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Field, Session, SQLModel


class ConversationThread(SQLModel, table=True):
  user_id: str = Field(primary_key=True)
  thread_id: str = Field(index=True)
  time_created: datetime = Field(
      default_factory=lambda: datetime.now(timezone.utc)
  )
  time_updated: datetime = Field(
      default_factory=lambda: datetime.now(timezone.utc)
  )

  @classmethod
  def get_thread_id(cls, user_id: str, db: Session) -> str:
    conversation = db.get(cls, user_id)
    if conversation:
      return conversation.thread_id

    # Preserve existing checkpoints that used user_id directly as thread_id.
    conversation = cls(user_id=user_id, thread_id=user_id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation.thread_id

  @classmethod
  def reset_thread_id(cls, user_id: str, db: Session) -> tuple[str, str]:
    conversation = db.get(cls, user_id)
    old_thread_id = conversation.thread_id if conversation else user_id
    new_thread_id = f"{user_id}:{uuid4().hex}"

    if conversation:
      conversation.thread_id = new_thread_id
      conversation.time_updated = datetime.now(timezone.utc)
    else:
      conversation = cls(user_id=user_id, thread_id=new_thread_id)

    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return old_thread_id, conversation.thread_id

  @classmethod
  def delete_for_user(cls, user_id: str, db: Session) -> str:
    conversation = db.get(cls, user_id)
    if not conversation:
      return user_id

    thread_id = conversation.thread_id
    db.delete(conversation)
    db.commit()
    return thread_id
