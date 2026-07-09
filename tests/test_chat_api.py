import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.chat.router import chat_router
from src.chat.models import ConversationThread
from src.dependencies import get_db, get_rag_graph
from src.user.models import UserProfile


class FakeSession:
    def __init__(self):
        self.objects = {}

    def get(self, model, user_id):
        if model is UserProfile:
            return UserProfile(user_id=user_id, age=27, region="서울")
        if model is ConversationThread:
            return self.objects.get((model, user_id))
        raise AssertionError(f"unexpected model: {model}")

    def add(self, obj):
        self.objects[(type(obj), obj.user_id)] = obj

    def commit(self):
        pass

    def refresh(self, obj):
        pass

    def delete(self, obj):
        self.objects.pop((type(obj), obj.user_id), None)


class FakeRagGraph:
    def __init__(self):
        self.stream_calls = []
        self.deleted_thread_ids = []

    async def stream_answer(self, **kwargs):
        self.stream_calls.append(kwargs)
        yield 'data: {"type": "metadata", "data": {"contexts": [], "retrieved_policy_ids": []}}\n\n'
        yield 'data: {"type": "chunk", "data": "테스트 답변"}\n\n'
        yield 'data: {"type": "done"}\n\n'

    def delete_conversation(self, thread_id):
        self.deleted_thread_ids.append(thread_id)


def build_client():
    app = FastAPI()
    app.include_router(chat_router)
    rag = FakeRagGraph()
    session = FakeSession()
    app.dependency_overrides[get_rag_graph] = lambda: rag
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app), rag, session


def test_chat_endpoint_forwards_request_to_policy_rag_graph():
    client, rag, _ = build_client()

    response = client.post(
        "/chat",
        json={
            "user_id": "api-user",
            "user_input": "월세 지원 정책을 알려줘.",
            "exclude_expired": False,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
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
    assert rag.stream_calls[0]["user_input"] == "월세 지원 정책을 알려줘."
    assert rag.stream_calls[0]["user_profile"] == {
        "age": 27,
        "gender": None,
        "job": None,
        "income": None,
        "region": "서울",
    }
    assert rag.stream_calls[0]["exclude_expired"] is False
    assert rag.stream_calls[0]["thread_id"] == "api-user"
    assert rag.stream_calls[0]["trace_user_id"] == "api-user"


def test_delete_chat_endpoint_clears_graph_checkpoint():
    client, rag, session = build_client()
    session.add(
        ConversationThread(
            user_id="api-user",
            thread_id="api-user:existing-thread",
        )
    )

    response = client.delete("/chat/api-user")

    assert response.status_code == 200
    assert response.json() == {"message": "대화 기록 삭제 완료"}
    assert rag.deleted_thread_ids == ["api-user:existing-thread", "api-user"]
    new_thread = session.get(ConversationThread, "api-user")
    assert new_thread.thread_id.startswith("api-user:")
    assert new_thread.thread_id != "api-user:existing-thread"
