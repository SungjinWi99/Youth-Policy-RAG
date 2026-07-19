import asyncio

from langchain_core.documents import Document

from src.rag.nodes.retriever import (
    BM25DocumentIndex,
    BM25PolicyRetriever,
    EnsemblePolicyRetriever,
    RetrievalRequest,
    tokenize_korean_legacy,
    tokenize_korean_lexical,
)


def policy_document(policy_id: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"plcyNo": policy_id},
    )


class FakeRetriever:
    search_k = 3

    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, request):
        self.calls.append(request)
        return list(self.documents)

    async def aretrieve(self, request):
        return self.retrieve(request)


class FakeCollection:
    def __init__(self):
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("include") == ["documents", "metadatas"]:
            return {
                "ids": ["POLICY"],
                "documents": ["청년 월세 지원"],
                "metadatas": [{"plcyNo": "POLICY"}],
            }
        return {"ids": ["POLICY"]}


def test_legacy_korean_lexical_tokenizer_adds_character_bigrams():
    assert tokenize_korean_legacy("청년월세 UPSTAGE") == [
        "청년월세",
        "청년",
        "년월",
        "월세",
        "upstage",
    ]


def test_korean_lexical_tokenizer_uses_kiwi_morphemes_and_compound_noun():
    assert tokenize_korean_lexical("청년월세 UPSTAGE") == [
        "청년",
        "월세",
        "청년월세",
        "upstage",
    ]


def test_korean_lexical_tokenizer_removes_particles_and_endings():
    assert tokenize_korean_lexical("학자금대출 이자를 지원받으려면") == [
        "학자금",
        "대출",
        "학자금대출",
        "이자",
        "지원",
        "받",
    ]


def test_bm25_index_prefers_exact_lexical_match():
    exact = policy_document("EXACT", "청년 월세 지원 주거비")
    other = policy_document("OTHER", "청년 취업 면접 지원")
    index = BM25DocumentIndex([other, exact])

    results = index.search("월세 주거비", limit=2)

    assert [document.metadata["plcyNo"] for document, _ in results] == [
        "EXACT",
    ]


def test_bm25_index_respects_empty_allowed_policy_set():
    index = BM25DocumentIndex([
        policy_document("POLICY", "청년 월세 지원"),
    ])

    assert index.search(
        "월세 지원",
        limit=10,
        allowed_policy_ids=set(),
    ) == []


def test_bm25_retriever_omits_where_when_filter_is_empty():
    collection = FakeCollection()
    retriever = BM25PolicyRetriever(collection=collection, search_k=3)

    documents = retriever.retrieve(RetrievalRequest(
        query="월세 지원",
        user_profile={},
        exclude_expired=False,
    ))

    assert [document.metadata["plcyNo"] for document in documents] == [
        "POLICY",
    ]
    assert collection.calls[-1] == {"include": []}


def test_ensemble_retriever_combines_dense_and_bm25_ranks():
    documents = {
        policy_id: policy_document(policy_id, policy_id)
        for policy_id in ["A", "B", "C", "D"]
    }
    dense = FakeRetriever([
        documents["A"],
        documents["B"],
        documents["C"],
    ])
    bm25 = FakeRetriever([
        documents["B"],
        documents["D"],
        documents["A"],
    ])
    retriever = EnsemblePolicyRetriever(
        retrievers=[dense, bm25],
        weights=[0.5, 0.5],
        search_k=3,
        rrf_k=60,
    )
    request = RetrievalRequest(
        query="query",
        user_profile={},
        exclude_expired=False,
    )

    results, source_results = retriever.retrieve_with_sources(request)

    assert [document.metadata["plcyNo"] for document in source_results[0]] == [
        "A", "B", "C",
    ]
    assert [document.metadata["plcyNo"] for document in source_results[1]] == [
        "B", "D", "A",
    ]
    assert [document.metadata["plcyNo"] for document in results] == [
        "B", "A", "D",
    ]
    assert dense.calls == [request]
    assert bm25.calls == [request]


def test_ensemble_rrf_does_not_reward_lower_ranked_documents():
    documents = {
        policy_id: policy_document(policy_id, policy_id)
        for policy_id in ["FIRST", "SECOND", "THIRD"]
    }
    source = FakeRetriever([
        documents["FIRST"],
        documents["SECOND"],
        documents["THIRD"],
    ])
    retriever = EnsemblePolicyRetriever(
        retrievers=[source],
        weights=[1.0],
        search_k=3,
        rrf_k=1,
    )

    results = retriever.retrieve(RetrievalRequest(
        query="query",
        user_profile={},
        exclude_expired=False,
    ))

    assert [document.metadata["plcyNo"] for document in results] == [
        "FIRST", "SECOND", "THIRD",
    ]


def test_ensemble_async_api_uses_the_same_fusion():
    documents = {
        policy_id: policy_document(policy_id, policy_id)
        for policy_id in ["A", "B", "C"]
    }
    retriever = EnsemblePolicyRetriever(
        retrievers=[
            FakeRetriever([documents["A"], documents["B"]]),
            FakeRetriever([documents["B"], documents["C"]]),
        ],
        weights=[0.65, 0.35],
        search_k=3,
        rrf_k=1,
    )

    results = asyncio.run(retriever.aretrieve(RetrievalRequest(
        query="query",
        user_profile={},
        exclude_expired=False,
    )))

    assert [document.metadata["plcyNo"] for document in results] == [
        "B", "A", "C",
    ]


def test_ensemble_rejects_mismatched_retrievers_and_weights():
    source = FakeRetriever([])

    try:
        EnsemblePolicyRetriever(
            retrievers=[source],
            weights=[0.5, 0.5],
            search_k=3,
            rrf_k=1,
        )
    except ValueError as error:
        assert "길이" in str(error)
    else:
        raise AssertionError("ValueError가 필요합니다.")
