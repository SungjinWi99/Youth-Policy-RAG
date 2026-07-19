from langchain_core.documents import Document

from src.rag.reranker import LlamaCppReranker


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload)


def test_llama_cpp_reranker_orders_by_relevance_score():
    session = FakeSession({
        "results": [
            {"index": 0, "relevance_score": -2.0},
            {"index": 1, "relevance_score": 5.0},
        ],
    })
    reranker = LlamaCppReranker(
        base_url="http://127.0.0.1:11435/",
        model="bge-reranker-v2-m3",
        session=session,
    )
    documents = [
        Document(page_content="first", metadata={"plcyNo": "A"}),
        Document(page_content="second", metadata={"plcyNo": "B"}),
    ]

    results = reranker.rerank(query="query", documents=documents)

    assert [result.document.metadata["plcyNo"] for result in results] == [
        "B",
        "A",
    ]
    assert [result.original_rank for result in results] == [2, 1]
    assert session.calls == [(
        "http://127.0.0.1:11435/v1/rerank",
        {
            "json": {
                "model": "bge-reranker-v2-m3",
                "query": "query",
                "documents": ["first", "second"],
            },
            "timeout": 60.0,
        },
    )]


def test_llama_cpp_reranker_returns_empty_without_http_call():
    session = FakeSession({"results": []})
    reranker = LlamaCppReranker(
        base_url="http://127.0.0.1:11435",
        model="bge-reranker-v2-m3",
        session=session,
    )

    assert reranker.rerank(query="query", documents=[]) == []
    assert session.calls == []
