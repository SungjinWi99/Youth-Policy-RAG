from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.user.schemas import UserBase


class AnonymousSessionCreate(UserBase):
    accepted_storage: Literal[True]


class PublicProfile(UserBase):
    pass


class SessionStatus(BaseModel):
    expires_at: datetime
    profile: PublicProfile


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationSnapshot(BaseModel):
    messages: list[ConversationMessage] = Field(default_factory=list)
    active_policy_ids: list[str] = Field(default_factory=list)


class WebChatRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=4000)
    exclude_expired: bool = True

    @field_validator("user_input")
    @classmethod
    def normalize_user_input(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("질문을 입력하세요.")
        return normalized


FeedbackReason = Literal[
    "policy-mismatch",
    "outdated-information",
    "missing-details",
    "unclear-answer",
    "other",
]


class UserFeedbackRequest(BaseModel):
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    helpful: bool
    reason: FeedbackReason | None = None
    comment: str | None = Field(default=None, max_length=500)

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def require_reason_for_negative_feedback(self):
        if not self.helpful and self.reason is None:
            raise ValueError("아쉬운 점을 하나 선택해 주세요.")
        if self.helpful and self.reason is not None:
            raise ValueError("도움됐어요 평가에는 아쉬운 점을 지정할 수 없습니다.")
        return self
