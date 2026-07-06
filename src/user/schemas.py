from typing import Optional
from sqlmodel import SQLModel

from policy.utils import RegionName


class UserBase(SQLModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    job: Optional[str] = None
    income: Optional[int] = None
    region: Optional[str] = None


class UserCreate(UserBase):
    user_id: str
    region: Optional[RegionName] = None


class UserUpdate(UserBase):
    region: Optional[RegionName] = None
