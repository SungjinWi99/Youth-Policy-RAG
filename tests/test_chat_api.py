import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.chat.router import chat_router
from src.dependencies import get_db, get_rag_graph
from src.user.models import UserProfile


class FakeSession:
    def get(self, model, user_id):
        assert model is UserProfile
        return UserProfile(user_id=user_id, age=27, region="서울")


class FakeRagGraph:
    def __init__(self):
        self.stream_calls = []
        self.deleted_user_ids = []

    async def stream_answer(self, **kwargs):
        self.stream_calls.append(kwargs)
        yield 'data: {"type": "metadata", "data": {"contexts": [], "retrieved_policy_ids": []}}\n\n'
        yield 'data: {"type": "chunk", "data": "테스트 답변"}\n\n'
        yield 'data: {"type": "done"}\n\n'

    def delete_conversation(self, user_id):
        self.deleted_user_ids.append(user_id)


def build_client():
    app = FastAPI()
    app.include_router(chat_router)
    rag = FakeRagGraph()
    app.dependency_overrides[get_rag_graph] = lambda: rag
    app.dependency_overrides[get_db] = lambda: FakeSession()
    return TestClient(app), rag


def test_chat_endpoint_forwards_request_to_policy_rag_graph():
    client, rag = build_client()

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
    assert rag.stream_calls[0]["user_id"] == "api-user"


def test_delete_chat_endpoint_clears_graph_checkpoint():
    client, rag = build_client()

    response = client.delete("/chat/api-user")

    assert response.status_code == 200
    assert response.json() == {"message": "대화 기록 삭제 완료"}
    assert rag.deleted_user_ids == ["api-user"]
