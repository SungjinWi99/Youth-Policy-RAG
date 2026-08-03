from sqlalchemy import inspect
from sqlmodel import SQLModel, create_engine

from src.database import migrate_legacy_session_schema


def test_legacy_profile_and_thread_are_moved_into_anonymous_session():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE anonymoussession (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                time_created TEXT NOT NULL,
                time_updated TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE userprofile (
                user_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                job TEXT,
                income INTEGER,
                region TEXT,
                time_created TEXT NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            "CREATE INDEX ix_anonymoussession_expires_at "
            "ON anonymoussession (expires_at)"
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE conversationthread (
                user_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                time_created TEXT NOT NULL,
                time_updated TEXT NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO anonymoussession VALUES
            ('token-hash', 'anon_legacy', '2026-07-01', '2026-07-02', '2026-08-01')
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO userprofile VALUES
            ('anon_legacy', 27, '여성', '구직자', 3000, '서울', '2026-07-01')
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO conversationthread VALUES
            ('anon_legacy', 'session:legacy-thread', '2026-07-01', '2026-07-02')
            """
        )

        migrate_legacy_session_schema(connection)

        row = connection.exec_driver_sql(
            """
            SELECT token_hash, thread_id, age, gender, job, income, region
            FROM anonymoussession
            """
        ).one()
        assert row == (
            "token-hash",
            "session:legacy-thread",
            27,
            "여성",
            "구직자",
            3000,
            "서울",
        )
        tables = set(inspect(connection).get_table_names())
        assert "userprofile" not in tables
        assert "conversationthread" not in tables

    SQLModel.metadata.drop_all(engine)
