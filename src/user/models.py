from fastapi import HTTPException
from sqlmodel import Field, Session
from datetime import datetime, timezone
from src.user.schemas import UserBase, UserCreate, UserUpdate

class UserProfile(UserBase, table=True):
  user_id: str = Field(primary_key=True)
  time_created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

  @classmethod
  def create(cls, user_data: UserCreate, db: Session) -> "UserProfile":
      if db.get(cls, user_data.user_id):
         raise HTTPException(status_code=409, detail="이미 사용 중인 아이디입니다.")
      new_user = cls(**user_data.model_dump())
      db.add(new_user)
      db.commit()
      db.refresh(new_user)
      return new_user

  @classmethod
  def get(cls, user_id: str, db: Session) -> "UserProfile":
    user = db.get(cls, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return user

  @classmethod
  def update(cls, user_id: str, user_data: UserUpdate, db: Session) -> "UserProfile":
    user = cls.get(user_id, db)
    user.sqlmodel_update(user_data.model_dump(exclude_unset=True))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

  @classmethod
  def delete(cls, user_id: str, db: Session) -> dict:
    user = cls.get(user_id, db)
    db.delete(user)
    db.commit()
    return {"message": "삭제 완료"}