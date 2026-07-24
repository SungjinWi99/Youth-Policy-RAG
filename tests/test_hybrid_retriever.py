import asyncio
from datetime import date

from langchain_core.documents import Document

from src.rag.retrievers import (
    BM25DocumentIndex,
    BM25PolicyRetriever,
    DensePolicyRetriever,
    EnsemblePolicyRetriever,
    RetrievalRequest,
    add_policy_exclusion,
    build_filter_from_profile,
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


class FakeVectorRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        return list(self.documents)

    async def ainvoke(self, query):
        return list(self.documents)


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.search_kwargs = []

    def as_retriever(self, *, search_kwargs):
        self.search_kwargs.append(search_kwargs)
        return FakeVectorRetriever(self.documents)


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


def test_income_is_not_used_as_a_hard_retrieval_filter():
    assert build_filter_from_profile(
        {"income": 3200},
        exclude_expired=False,
        today=date(2026, 7, 24),
    ) is None


def test_income_does_not_change_other_hard_filters():
    without_income = build_filter_from_profile(
        {"age": 27, "region": "서울"},
        exclude_expired=False,
        today=date(2026, 7, 24),
    )
    with_income = build_filter_from_profile(
        {"age": 27, "region": "서울", "income": 3200},
        exclude_expired=False,
        today=date(2026, 7, 24),
    )

    assert with_income == without_income


def test_policy_exclusion_filter_is_added_without_profile_filters():
    assert add_policy_exclusion(
        None,
        frozenset({"POLICY-B", "POLICY-A"}),
    ) == {
        "plcyNo": {
            "$nin": ["POLICY-A", "POLICY-B"],
        }
    }


def test_policy_exclusion_filter_preserves_existing_profile_filter():
    profile_filter = {"agePolicy": {"$eq": "all"}}

    assert add_policy_exclusion(
        profile_filter,
        frozenset({"POLICY"}),
    ) == {
        "$and": [
            profile_filter,
            {"plcyNo": {"$nin": ["POLICY"]}},
        ]
    }


def test_dense_retriever_passes_policy_exclusion_to_vector_store():
    vector_store = FakeVectorStore([
        policy_document("OTHER", "다른 정책"),
    ])
    retriever = DensePolicyRetriever(
        vector_store=vector_store,
        search_k=3,
        today_provider=lambda: date(2026, 7, 24),
    )

    retriever.retrieve(RetrievalRequest(
        query="월세 지원",
        user_profile={},
        exclude_expired=False,
        excluded_policy_ids=frozenset({"REJECTED"}),
    ))

    assert vector_store.search_kwargs == [{
        "k": 3,
        "filter": {
            "plcyNo": {
                "$nin": ["REJECTED"],
            }
        },
    }]


def test_bm25_retriever_excludes_rejected_policy_id():
    collection = FakeCollection()
    retriever = BM25PolicyRetriever(collection=collection, search_k=3)

    documents = retriever.retrieve(RetrievalRequest(
        query="월세 지원",
        user_profile={},
        exclude_expired=False,
        excluded_policy_ids=frozenset({"POLICY"}),
    ))

    assert documents == []


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


def test_ensemble_defensively_excludes_policy_from_all_sources():
    documents = {
        policy_id: policy_document(policy_id, policy_id)
        for policy_id in ["A", "B", "C"]
    }
    retriever = EnsemblePolicyRetriever(
        retrievers=[
            FakeRetriever([documents["B"], documents["A"]]),
            FakeRetriever([documents["B"], documents["C"]]),
        ],
        weights=[0.5, 0.5],
        search_k=3,
        rrf_k=1,
    )
    request = RetrievalRequest(
        query="query",
        user_profile={},
        exclude_expired=False,
        excluded_policy_ids=frozenset({"B"}),
    )

    results, source_results = retriever.retrieve_with_sources(request)

    assert [
        [document.metadata["plcyNo"] for document in source]
        for source in source_results
    ] == [["A"], ["C"]]
    assert [document.metadata["plcyNo"] for document in results] == ["A", "C"]


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
