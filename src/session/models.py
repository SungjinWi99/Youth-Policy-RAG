from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnonymousSession(SQLModel, table=True):
    token_hash: str = Field(primary_key=True)
    user_id: str = Field(index=True, unique=True)
    time_created: datetime = Field(default_factory=utc_now)
    time_updated: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(index=True)
