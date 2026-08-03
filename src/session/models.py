from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


PROFILE_FIELD_NAMES = {"age", "gender", "job", "income", "region"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionProfile(SQLModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    job: Optional[str] = None
    income: Optional[int] = None
    region: Optional[str] = None


class AnonymousSession(SessionProfile, table=True):
    token_hash: str = Field(primary_key=True)
    thread_id: str = Field(index=True)
    time_created: datetime = Field(default_factory=utc_now)
    time_updated: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(index=True)
