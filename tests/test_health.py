from types import SimpleNamespace

from fastapi.testclient import TestClient

from main import app


class FakeCollection:
    def __init__(self, count):
        self._count = count

    def count(self):
        if self._count is None:
            raise RuntimeError("컬렉션을 열 수 없습니다.")
        return self._count


def build_client(count):
    app.state.rag_graph = SimpleNamespace(
        vector_store=SimpleNamespace(_collection=FakeCollection(count))
    )
    # lifespan을 돌리지 않으려고 컨텍스트 매니저를 쓰지 않는다.
    return TestClient(app)


def test_health_reports_document_count():
    response = build_client(2695).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents": 2695}


def test_health_fails_when_collection_is_empty():
    response = build_client(0).get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "chroma_empty"


def test_health_fails_when_collection_is_unavailable():
    response = build_client(None).get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "chroma_unavailable"


def test_health_fails_before_the_graph_is_built():
    app.state.rag_graph = None

    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "chroma_unavailable"
