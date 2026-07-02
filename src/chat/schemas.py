from pydantic import BaseModel


class ChatRequest(BaseModel):
  user_id: str
  user_input: str
  exclude_expired: bool = True


class ConversationResetResponse(BaseModel):
  user_id: str
  message: str
