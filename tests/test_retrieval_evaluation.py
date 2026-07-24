from langchain_core.documents import Document

from src.evaluation.models import EvaluationCase
from src.evaluation.retrieval import evaluate_retrieval
from src.rag.retrievers import DensePolicyRetriever


class FakeRunnableRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def invoke(self, query):
        self.vector_store.queries.append(query)
        return list(self.vector_store.documents)


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.search_kwargs = []
        self.queries = []

    def as_retriever(self, *, search_kwargs):
        self.search_kwargs.append(search_kwargs)
        return FakeRunnableRetriever(self)


class FakeCollection:
    def __init__(self, policy_ids):
        self.policy_ids = policy_ids

    def count(self):
        return len(self.policy_ids)

    def get(self, **kwargs):
        return {"ids": list(self.policy_ids)}


def test_evaluation_uses_policy_retriever_with_unmodified_user_input():
    documents = [
        Document(page_content="2위", metadata={"plcyNo": "POLICY-002"}),
        Document(page_content="1위", metadata={"plcyNo": "POLICY-001"}),
        Document(page_content="3위", metadata={"plcyNo": "POLICY-003"}),
    ]
    vector_store = FakeVectorStore(documents)
    retriever = DensePolicyRetriever(
        vector_store=vector_store,
        search_k=3,
    )
    collection = FakeCollection([
        "POLICY-001",
        "POLICY-002",
        "POLICY-003",
    ])
    cases = [EvaluationCase(
        case_id="case-1",
        expected_policy_ids=["POLICY-001"],
        user_input="월세 지원 정책 있어?",
        user_profile={"region": "서울"},
    )]

    summary, details = evaluate_retrieval(
        collection=collection,
        retriever=retriever,
        cases=cases,
        rank_depth=3,
        k_values=(1, 3),
        today_yyyymmdd=20260714,
    )

    assert vector_store.queries == ["월세 지원 정책 있어?"]
    assert vector_store.search_kwargs == [{
        "k": 3,
        "filter": {"region_11": {"$eq": True}},
    }]
    assert summary["evaluation"]["retriever_class"] == "DensePolicyRetriever"
    assert summary["evaluation"]["query_source"] == "user_input"
    assert summary["metrics"]["recall_at_1"] == 0.0
    assert summary["metrics"]["recall_at_3"] == 1.0
    assert summary["metrics"]["mrr"] == 0.5
    assert details[0]["first_relevant_rank"] == 2
    assert details[0]["top_retrieved_policy_ids"] == [
        "POLICY-002",
        "POLICY-001",
        "POLICY-003",
    ]
