from pathlib import Path

from sqlalchemy import Connection, inspect
from sqlmodel import SQLModel, Session, create_engine

from src.config import load_config
from src.session.models import AnonymousSession  # noqa: F401


config = load_config()
sqlite_path = Path(config.path(config.data.sqlite_db))
sqlite_path.parent.mkdir(parents=True, exist_ok=True)

sqlite_url = f"sqlite:///{sqlite_path}"

engine = create_engine(
    sqlite_url,
    echo=config.database.echo,
    connect_args={"check_same_thread": False},
)

def migrate_legacy_session_schema(connection: Connection) -> None:
    """Move legacy profile/thread rows into the single anonymous-session table."""
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    legacy_table = "anonymoussession_legacy"

    if legacy_table not in tables:
        if "anonymoussession" not in tables:
            return
        columns = {
            column["name"]
            for column in inspector.get_columns("anonymoussession")
        }
        if "user_id" not in columns:
            return

        # SQLite index names are database-wide and survive a table rename.
        connection.exec_driver_sql(
            "DROP INDEX IF EXISTS ix_anonymoussession_expires_at"
        )
        connection.exec_driver_sql(
            "ALTER TABLE anonymoussession RENAME TO anonymoussession_legacy"
        )
        tables.remove("anonymoussession")
        tables.add(legacy_table)

    if "anonymoussession" not in tables:
        SQLModel.metadata.create_all(connection)

    profile_join = (
        "LEFT JOIN userprofile AS profile ON profile.user_id = legacy.user_id"
        if "userprofile" in tables
        else ""
    )
    thread_join = (
        "LEFT JOIN conversationthread AS conversation "
        "ON conversation.user_id = legacy.user_id"
        if "conversationthread" in tables
        else ""
    )
    profile_columns = {
        name: f"profile.{name}" if profile_join else "NULL"
        for name in ("age", "gender", "job", "income", "region")
    }
    thread_id = (
        "COALESCE(conversation.thread_id, legacy.user_id)"
        if thread_join
        else "legacy.user_id"
    )
    connection.exec_driver_sql(
        f"""
        INSERT OR IGNORE INTO anonymoussession (
            token_hash, thread_id, age, gender, job, income, region,
            time_created, time_updated, expires_at
        )
        SELECT
            legacy.token_hash, {thread_id}, {profile_columns['age']},
            {profile_columns['gender']}, {profile_columns['job']},
            {profile_columns['income']}, {profile_columns['region']},
            legacy.time_created, legacy.time_updated, legacy.expires_at
        FROM anonymoussession_legacy AS legacy
        {profile_join} {thread_join}
        """
    )
    connection.exec_driver_sql(f"DROP TABLE {legacy_table}")
    for table in ("conversationthread", "userprofile"):
        if table in tables:
            connection.exec_driver_sql(f"DROP TABLE {table}")
    SQLModel.metadata.create_all(connection)


def create_db_and_tables():
    with engine.begin() as connection:
        migrate_legacy_session_schema(connection)
        SQLModel.metadata.create_all(connection)

def get_session():
    with Session(engine) as session:
        yield session
