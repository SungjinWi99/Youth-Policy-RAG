from types import SimpleNamespace

from langchain_core.documents import Document

from src.evaluation.langfuse import (
    build_mean_run_evaluator,
    build_recall_evaluator,
    reciprocal_rank_evaluator,
)
from src.evaluation.models import PlannerQueryRecord
from src.evaluation.retrieval import (
    build_retrieval_task,
    reciprocal_rank_fusion,
)
from src.rag.nodes.retriever import (
    EnsemblePolicyRetriever,
    RetrievalRequest,
)


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, request):
        self.calls.append(request)
        return [
            Document(page_content="", metadata={"plcyNo": "POLICY-002"}),
            Document(page_content="", metadata={"plcyNo": "POLICY-001"}),
        ]


class QueryAwareFakeRetriever:
    search_k = 3

    def __init__(self):
        self.calls = []
        self.results = {
            "query-1": ["A", "B", "C"],
            "query-2": ["B", "D", "A"],
            "query-3": ["B", "E", "F"],
        }

    def retrieve(self, request):
        self.calls.append(request)
        return [
            Document(page_content="", metadata={"plcyNo": policy_id})
            for policy_id in self.results[request.query]
        ]


class FakeReranker:
    model = "fake-reranker"

    def rerank(self, *, query, documents):
        from src.rag.reranker import RerankedDocument

        return [
            RerankedDocument(
                document=document,
                original_rank=index,
                relevance_score=float(index),
            )
            for index, document in reversed(list(enumerate(documents, start=1)))
        ]


class StaticSourceRetriever:
    search_k = 3

    def __init__(self, policy_ids):
        self.policy_ids = policy_ids
        self.calls = []

    def retrieve(self, request):
        self.calls.append(request)
        return [
            Document(page_content="", metadata={"plcyNo": policy_id})
            for policy_id in self.policy_ids
        ]

    async def aretrieve(self, request):
        return self.retrieve(request)


def test_retrieval_task_returns_ranked_policy_ids():
    retriever = FakeRetriever()
    task = build_retrieval_task(retriever)

    output = task(item=SimpleNamespace(input={
        "user_input": "월세 지원 있어?",
        "user_profile": {"region": "서울"},
        "exclude_expired": False,
    }))

    assert output["raw_query"] == output["executed_query"]
    assert output["used_raw_fallback"] is False
    assert output["retrieved_policy_ids"] == ["POLICY-002", "POLICY-001"]
    assert retriever.calls == [RetrievalRequest(
        query="월세 지원 있어?",
        user_profile={"region": "서울"},
        exclude_expired=False,
    )]


def test_retrieval_task_reranks_dense_candidates():
    retriever = FakeRetriever()
    task = build_retrieval_task(retriever, reranker=FakeReranker())

    output = task(item=SimpleNamespace(input={
        "user_input": "월세 지원 있어?",
        "user_profile": {},
        "exclude_expired": False,
    }))

    assert output["dense_retrieved_policy_ids"] == [
        "POLICY-002",
        "POLICY-001",
    ]
    assert output["retrieved_policy_ids"] == [
        "POLICY-001",
        "POLICY-002",
    ]
    assert output["reranker_results"] == [
        {
            "policy_id": "POLICY-001",
            "dense_rank": 2,
            "rerank_score": 2.0,
        },
        {
            "policy_id": "POLICY-002",
            "dense_rank": 1,
            "rerank_score": 1.0,
        },
    ]


def test_retrieval_task_uses_ensemble_request_api_and_records_sources():
    dense = StaticSourceRetriever(["A", "B", "C"])
    bm25 = StaticSourceRetriever(["B", "D", "A"])
    retriever = EnsemblePolicyRetriever(
        retrievers=[dense, bm25],
        weights=[0.65, 0.35],
        search_k=3,
        rrf_k=1,
    )

    output = build_retrieval_task(retriever)(item=SimpleNamespace(input={
        "user_input": "월세 지원",
        "user_profile": {"region": "서울"},
        "exclude_expired": False,
    }))

    assert output["source_retrieved_policy_ids"] == [
        ["A", "B", "C"],
        ["B", "D", "A"],
    ]
    assert output["dense_retrieved_policy_ids"] == ["A", "B", "C"]
    assert output["bm25_retrieved_policy_ids"] == ["B", "D", "A"]
    assert output["retrieved_policy_ids"] == ["A", "B", "C"]
    assert dense.calls == bm25.calls == [RetrievalRequest(
        query="월세 지원",
        user_profile={"region": "서울"},
        exclude_expired=False,
    )]


def test_retrieval_task_uses_first_planner_query():
    retriever = FakeRetriever()
    planner_records = {
        "case-1": PlannerQueryRecord(
            case_id="case-1",
            raw_query="월세 지원 있어?",
            user_profile={"region": "서울"},
            planner_route="retriever",
            answer_strategy="policy_recommendation",
            retrieval_queries=["서울 월세 지원", "서울 주거비 지원"],
            route_reason="새 정책 검색이 필요함",
            planner_provider="deepseek",
            planner_model="deepseek-v4-flash",
            planner_prompt_sha256="hash",
            generated_at="2026-07-14T00:00:00+00:00",
        ),
    }
    task = build_retrieval_task(retriever, planner_records)

    output = task(item=SimpleNamespace(
        input={
            "user_input": "월세 지원 있어?",
            "user_profile": {"region": "서울"},
            "exclude_expired": False,
        },
        metadata={"case_id": "case-1"},
    ))

    assert output["executed_query"] == "서울 월세 지원"
    assert output["planner_queries"] == [
        "서울 월세 지원",
        "서울 주거비 지원",
    ]
    assert output["retrieved_policy_ids"] == ["POLICY-002", "POLICY-001"]
    assert retriever.calls[0].query == "서울 월세 지원"


def test_retrieval_task_respects_planner_agent_route():
    retriever = FakeRetriever()
    planner_records = {
        "case-1": PlannerQueryRecord(
            case_id="case-1",
            raw_query="안녕",
            user_profile={},
            planner_route="agent",
            answer_strategy="brief_reply",
            retrieval_queries=[],
            route_reason="검색이 필요하지 않음",
            planner_provider="deepseek",
            planner_model="deepseek-v4-flash",
            planner_prompt_sha256="hash",
            generated_at="2026-07-14T00:00:00+00:00",
        ),
    }
    task = build_retrieval_task(retriever, planner_records)

    output = task(item=SimpleNamespace(
        input={"user_input": "안녕", "user_profile": {}},
        metadata={"case_id": "case-1"},
    ))

    assert output["executed_query"] is None
    assert output["retrieved_policy_ids"] == []
    assert retriever.calls == []


def test_same_planner_query_as_raw_is_not_a_fallback():
    retriever = FakeRetriever()
    planner_records = {
        "case-1": PlannerQueryRecord(
            case_id="case-1",
            raw_query="same query",
            user_profile={},
            planner_route="retriever",
            answer_strategy="policy_recommendation",
            retrieval_queries=["same query", "alternative query"],
            route_reason="search",
            planner_provider="deepseek",
            planner_model="deepseek-v4-flash",
            planner_prompt_sha256="hash",
            generated_at="2026-07-14T00:00:00+00:00",
        ),
    }

    output = build_retrieval_task(retriever, planner_records)(
        item=SimpleNamespace(
            input={"user_input": "same query", "user_profile": {}},
            metadata={"case_id": "case-1"},
        )
    )

    assert output["executed_query"] == "same query"
    assert output["used_raw_fallback"] is False


def test_deterministic_retrieval_evaluators():
    output = {"retrieved_policy_ids": ["POLICY-002", "POLICY-001"]}
    expected_output = {"expected_policy_ids": ["POLICY-001"]}

    recall_at_1 = build_recall_evaluator(1)(
        output=output,
        expected_output=expected_output,
    )
    recall_at_3 = build_recall_evaluator(3)(
        output=output,
        expected_output=expected_output,
    )
    reciprocal_rank = reciprocal_rank_evaluator(
        output=output,
        expected_output=expected_output,
    )

    assert recall_at_1.value == 0.0
    assert recall_at_3.value == 1.0
    assert reciprocal_rank.value == 0.5


def test_run_evaluator_averages_item_scores():
    item_results = [
        SimpleNamespace(evaluations=[
            SimpleNamespace(name="reciprocal_rank", value=1.0),
        ]),
        SimpleNamespace(evaluations=[
            SimpleNamespace(name="reciprocal_rank", value=0.5),
        ]),
    ]
    evaluator = build_mean_run_evaluator(
        item_metric_name="reciprocal_rank",
        run_metric_name="mrr",
    )

    result = evaluator(item_results=item_results)

    assert result.name == "mrr"
    assert result.value == 0.75


def test_reciprocal_rank_fusion_is_deterministic():
    fused = reciprocal_rank_fusion(
        [
            ["A", "B", "C"],
            ["B", "D", "A"],
            ["B", "E", "F"],
        ],
        rrf_k=60,
        limit=3,
    )

    assert [policy_id for policy_id, _ in fused] == ["B", "A", "D"]
    assert fused[0][1] == 1 / 62 + 1 / 61 + 1 / 61


def test_retrieval_task_fuses_all_planner_queries_with_rrf():
    retriever = QueryAwareFakeRetriever()
    planner_records = {
        "case-1": PlannerQueryRecord(
            case_id="case-1",
            raw_query="raw query",
            user_profile={},
            planner_route="retriever",
            answer_strategy="policy_recommendation",
            retrieval_queries=["query-1", "query-2", "query-3"],
            route_reason="search",
            planner_provider="deepseek",
            planner_model="deepseek-v4-flash",
            planner_prompt_sha256="hash",
            generated_at="2026-07-14T00:00:00+00:00",
        ),
    }
    task = build_retrieval_task(
        retriever,
        planner_records,
        planner_query_mode="rrf",
        rrf_k=60,
    )

    output = task(item=SimpleNamespace(
        input={"user_input": "raw query", "user_profile": {}},
        metadata={"case_id": "case-1"},
    ))

    assert output["executed_queries"] == ["query-1", "query-2", "query-3"]
    assert output["retrieved_policy_ids"] == ["B", "A", "D"]
    assert output["per_query_retrieved_policy_ids"] == [
        ["A", "B", "C"],
        ["B", "D", "A"],
        ["B", "E", "F"],
    ]
    assert [call.query for call in retriever.calls] == [
        "query-1",
        "query-2",
        "query-3",
    ]
