from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

from src.chat.models import ConversationThread  # noqa: F401
from src.config import load_config
from src.session.models import AnonymousSession  # noqa: F401
from src.user.models import UserProfile  # noqa: F401


config = load_config()
sqlite_path = Path(config.path(config.data.sqlite_db))
sqlite_path.parent.mkdir(parents=True, exist_ok=True)

sqlite_url = f"sqlite:///{sqlite_path}"

engine = create_engine(
    sqlite_url,
    echo=config.database.echo,
    connect_args={"check_same_thread": False},
)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
