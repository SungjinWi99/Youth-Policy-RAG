import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from src.dependencies import get_db, get_langfeather_runtime, get_rag_graph
from src.session.models import AnonymousSession
from src.session.router import session_router


class FakeRagGraph:
    def __init__(self):
        self.stream_calls = []
        self.deleted_thread_ids = []

    async def stream_answer(self, **kwargs):
        self.stream_calls.append(kwargs)
        yield (
            'data: {"type": "metadata", "data": '
            '{"contexts": [], "retrieved_policy_ids": ["POLICY-1"], '
            f'"trace_id": "{kwargs["trace_id"]}"'
            "}}\n\n"
        )
        yield 'data: {"type": "chunk", "data": "지원 정책입니다."}\n\n'
        yield 'data: {"type": "done"}\n\n'

    async def get_conversation(self, thread_id):
        return {
            "messages": [
                {"role": "user", "content": "주거 정책을 알려줘."},
                {"role": "assistant", "content": "지원 정책입니다."},
            ],
            "active_policy_ids": ["POLICY-1"],
        }

    def delete_conversation(self, thread_id):
        self.deleted_thread_ids.append(thread_id)


class FakeLangFeatherRuntime:
    def __init__(self):
        self.feedback_calls = []

    @property
    def enabled(self):
        return True

    def create_trace_id(self):
        return "b" * 32

    def trace_metadata(self, trace_id):
        return {"langfeather_trace_id": trace_id}

    def record_user_feedback(self, **kwargs):
        self.feedback_calls.append(kwargs)


def build_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    rag = FakeRagGraph()
    langfeather_runtime = FakeLangFeatherRuntime()
    rag.langfeather_runtime = langfeather_runtime
    app = FastAPI()
    app.state.rag_graph = rag
    app.state.langfeather_runtime = langfeather_runtime
    app.include_router(session_router)

    def override_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_rag_graph] = lambda: rag
    app.dependency_overrides[get_langfeather_runtime] = lambda: langfeather_runtime
    return TestClient(app), engine, rag


def start_session(client: TestClient):
    return client.post(
        "/sessions/anonymous",
        json={
            "age": 27,
            "region": "서울",
            "job": "취업준비생",
            "accepted_storage": True,
        },
    )


def test_anonymous_session_sets_private_cookie_and_restores_profile():
    client, engine, _ = build_client()

    response = start_session(client)

    assert response.status_code == 201
    assert response.json()["profile"] == {
        "age": 27,
        "gender": None,
        "job": "취업준비생",
        "income": None,
        "region": "서울",
    }
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "max-age=2592000" in cookie

    restored = client.get("/sessions/current")
    assert restored.status_code == 200
    assert restored.json()["profile"]["region"] == "서울"

    with Session(engine) as db:
        assert len(db.exec(select(AnonymousSession)).all()) == 1
        session = db.exec(select(AnonymousSession)).one()
        assert session.region == "서울"


def test_session_chat_uses_session_thread_and_returns_sse():
    client, _, rag = build_client()
    start_session(client)

    response = client.post(
        "/me/chat",
        json={
            "user_input": "월세 지원을 알려줘.",
            "exclude_expired": True,
        },
    )

    assert response.status_code == 200
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == [
        "metadata",
        "chunk",
        "done",
    ]
    call = rag.stream_calls[0]
    assert call["user_input"] == "월세 지원을 알려줘."
    assert call["user_profile"]["region"] == "서울"
    assert call["thread_id"].startswith("session:")
    assert call["trace_id"] == "b" * 32
    assert call["trace_metadata"] == {"langfeather_trace_id": "b" * 32}
    assert events[0]["data"]["trace_id"] == "b" * 32


def test_session_feedback_is_recorded_against_trace():
    client, _, rag = build_client()
    start_session(client)

    response = client.post(
        "/me/feedback",
        json={
            "trace_id": "a" * 32,
            "helpful": False,
            "reason": "missing-details",
            "comment": "신청 방법이 더 필요해요.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"message": "피드백이 저장되었습니다."}
    assert rag.langfeather_runtime.feedback_calls[0]["trace_id"] == "a" * 32
    assert rag.langfeather_runtime.feedback_calls[0]["reason"] == "missing-details"
    assert rag.langfeather_runtime.feedback_calls[0][
        "anonymous_session_id"
    ].startswith("session:")


def test_negative_feedback_requires_reason():
    client, _, _ = build_client()
    start_session(client)

    response = client.post(
        "/me/feedback",
        json={"trace_id": "a" * 32, "helpful": False},
    )

    assert response.status_code == 422


def test_conversation_restore_and_delete_all_data():
    client, engine, rag = build_client()
    start_session(client)
    client.post("/me/chat", json={"user_input": "주거 정책"})

    snapshot = client.get("/me/conversation")
    assert snapshot.status_code == 200
    assert snapshot.json()["active_policy_ids"] == ["POLICY-1"]
    assert snapshot.json()["messages"][0]["role"] == "user"

    deleted = client.delete("/me/data")
    assert deleted.status_code == 200
    assert deleted.json() == {"message": "프로필과 상담 기록 삭제 완료"}
    assert client.get("/sessions/current").status_code == 401
    assert rag.deleted_thread_ids

    with Session(engine) as db:
        assert db.exec(select(AnonymousSession)).all() == []


def test_clear_conversation_keeps_profile_and_rotates_thread():
    client, engine, rag = build_client()
    start_session(client)
    client.post("/me/chat", json={"user_input": "주거 정책"})

    with Session(engine) as db:
        old_thread_id = db.exec(select(AnonymousSession)).one().thread_id

    cleared = client.delete("/me/conversation")

    assert cleared.status_code == 200
    assert cleared.json() == {"message": "대화 기록 삭제 완료"}
    assert rag.deleted_thread_ids == [old_thread_id]
    assert client.get("/sessions/current").status_code == 200

    with Session(engine) as db:
        session = db.exec(select(AnonymousSession)).one()
        assert session.thread_id != old_thread_id
        assert session.region == "서울"


def test_anonymous_session_requires_storage_consent():
    client, _, _ = build_client()

    response = client.post(
        "/sessions/anonymous",
        json={"age": 27, "accepted_storage": False},
    )

    assert response.status_code == 422
