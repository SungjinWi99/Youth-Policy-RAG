import asyncio
import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from src.rag.graph import PolicyRagGraph
from src.rag.nodes import (
    make_policy_selector_node,
    make_retriever_node,
)
from src.rag.retrievers import RetrievalRequest
from src.rag.utils import window_history


def policy_document(policy_id: str, content: str | None = None) -> Document:
    return Document(
        page_content=content or f"{policy_id} 정책",
        metadata={
            "plcyNo": policy_id,
            "agePolicy": "all",
            "incomePolicy": "all",
        },
    )


class FakePlanner:
    def __init__(self, force_search_inputs=None):
        self.calls = []
        self.force_search_inputs = set(force_search_inputs or [])

    def _run(self, state):
        self.calls.append(dict(state))
        retrieval_count = state.get("retrieval_count", 0)
        active_policies = state.get("active_policies", [])
        needs_retrieval = (
            retrieval_count > 0
            or not active_policies
            or state["user_input"] in self.force_search_inputs
        )
        if (
            retrieval_count > 0
            and not state.get("retrieved_policies", [])
        ):
            needs_retrieval = False
        update = {
            "user_requirement": f"요구: {state['user_input']}",
            "needs_retrieval": needs_retrieval,
            "retrieval_reason": (
                "새 검색이 필요합니다."
                if needs_retrieval
                else "활성 정책을 재사용합니다."
            ),
            "retrieval_query": (
                state["user_input"] if needs_retrieval else ""
            ),
            "documents": [] if needs_retrieval else list(active_policies),
        }
        return update

    async def _arun(self, state):
        return self._run(state)

    def runnable(self):
        return RunnableLambda(self._run, afunc=self._arun)


class FakePolicyRetriever:
    def __init__(self, documents, documents_by_query=None, search_k=3):
        self.documents = list(documents)
        self.documents_by_query = documents_by_query
        self.search_k = search_k
        self.calls = []

    def _retrieve(self, request: RetrievalRequest):
        candidates = (
            self.documents_by_query.get(request.query, [])
            if self.documents_by_query is not None
            else self.documents
        )
        return [
            document
            for document in candidates
            if document.metadata["plcyNo"]
            not in request.excluded_policy_ids
        ][:self.search_k]

    def retrieve(self, request: RetrievalRequest):
        self.calls.append(request)
        return self._retrieve(request)

    async def aretrieve(self, request: RetrievalRequest):
        self.calls.append(request)
        return self._retrieve(request)


class FakeChecker:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = []
        self.active = 0
        self.max_active = 0

    def _result(self, state):
        document = state["policy"]
        policy_id = document.metadata["plcyNo"]
        self.calls.append(policy_id)
        return {
            "checked_policies": [{
                "verdict": self.verdicts[policy_id],
                "document": document,
                "reasoning": f"{policy_id} 평가",
                "retrieval_rank": state.get("retrieval_rank", 1),
                "retrieval_round": state.get("retrieval_round", 1),
            }]
        }

    async def _aresult(self, state):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        try:
            return self._result(state)
        finally:
            self.active -= 1

    def runnable(self):
        return RunnableLambda(self._result, afunc=self._aresult)


class FakeAnswerGenerator:
    def __init__(self):
        self.calls = []

    def _run(self, state):
        self.calls.append({
            "documents": list(state.get("documents", [])),
            "chat_history": window_history(state.get("messages", []), 10),
        })
        answer = f"답변: {state['user_input']}"
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    async def _arun(self, state):
        return self._run(state)

    def runnable(self):
        return RunnableLambda(self._run, afunc=self._arun)


def build_graph(
    *,
    documents=None,
    documents_by_query=None,
    verdicts=None,
    max_retries=3,
    trace_config_factory=None,
    retriever_search_k=3,
    force_search_inputs=None,
):
    documents = documents or [policy_document("POLICY-001")]
    planner = FakePlanner(force_search_inputs)
    policy_retriever = FakePolicyRetriever(
        documents,
        documents_by_query=documents_by_query,
        search_k=retriever_search_k,
    )
    checker = FakeChecker(
        verdicts or {
            document.metadata["plcyNo"]: "direct_fit"
            for document in documents
        }
    )
    answer_generator = FakeAnswerGenerator()
    graph = PolicyRagGraph(
        retrieval_planner=planner.runnable(),
        retriever=make_retriever_node(policy_retriever),
        policy_checker=checker.runnable(),
        policy_selector=make_policy_selector_node(),
        answer_generator=answer_generator.runnable(),
        checkpointer=InMemorySaver(),
        max_retrieval_retries=max_retries,
        trace_config_factory=trace_config_factory,
    )
    return graph, planner, policy_retriever, checker, answer_generator


def test_compiled_graph_has_parallel_checker_topology():
    graph, *_ = build_graph()

    compiled = graph.graph.get_graph()
    edges = {
        (edge.source, edge.target, edge.conditional)
        for edge in compiled.edges
    }

    assert set(compiled.nodes) == {
        "__start__",
        "retrieval_planner",
        "retriever",
        "policy_checker",
        "policy_selector",
        "answer_generator",
        "__end__",
    }
    assert ("__start__", "retrieval_planner", False) in edges
    assert ("retrieval_planner", "retriever", True) in edges
    assert ("retriever", "policy_checker", True) in edges
    assert ("policy_checker", "policy_selector", False) in edges
    assert ("policy_selector", "retrieval_planner", True) in edges
    assert ("answer_generator", "__end__", False) in edges


def test_send_passes_raw_question_and_profile_to_each_checker():
    document = policy_document("POLICY-001")

    sends = PolicyRagGraph._dispatch_retrieved_policies({
        "user_input": "소득 조건을 확인해줘.",
        "user_profile": {
            "age": 27,
            "income": 3200,
            "region": "서울",
        },
        "user_requirement": "서울 청년 주거 정책의 소득 조건 확인",
        "retrieved_policies": [document],
    })

    assert len(sends) == 1
    assert sends[0].arg["current_question"] == "소득 조건을 확인해줘."
    assert sends[0].arg["user_profile"] == {
        "age": 27,
        "income": 3200,
        "region": "서울",
    }
    assert sends[0].arg["policy"] == document


def test_send_checks_each_document_and_verdict_filters_answer_context():
    documents = [
        policy_document("HIGH"),
        policy_document("LOW"),
        policy_document("BORDER"),
    ]
    graph, _, retriever, checker, answer_generator = build_graph(
        documents=documents,
        verdicts={
            "HIGH": "direct_fit",
            "LOW": "indirect",
            "BORDER": "fit_needs_clarification",
        },
    )

    result = graph.generate_answer(
        user_input="월세 지원 정책을 알려줘.",
        user_profile={"region": "서울"},
        thread_id="verdict-user",
    )

    assert set(checker.calls) == {"HIGH", "LOW", "BORDER"}
    assert result.retrieved_policy_ids == ["HIGH", "BORDER"]
    assert [
        document.metadata["plcyNo"]
        for document in answer_generator.calls[0]["documents"]
    ] == ["HIGH", "BORDER"]
    assert retriever.calls == [RetrievalRequest(
        query="월세 지원 정책을 알려줘.",
        user_profile={"region": "서울"},
        exclude_expired=True,
    )]


def test_async_send_runs_policy_checkers_in_parallel():
    documents = [
        policy_document("A"),
        policy_document("B"),
        policy_document("C"),
    ]
    graph, _, _, checker, _ = build_graph(
        documents=documents,
        verdicts={
            "A": "direct_fit",
            "B": "direct_fit",
            "C": "fit_needs_clarification",
        },
    )

    result = asyncio.run(graph.agenerate_answer(
        user_input="청년 지원 정책",
        user_profile={},
        thread_id="parallel-user",
    ))

    assert result.retrieved_policy_ids == ["A", "B", "C"]
    assert checker.max_active == 3


def test_failed_check_retries_same_query_while_excluding_rejected_policy():
    rejected = policy_document("REJECTED")
    accepted = policy_document("ACCEPTED")
    graph, _, retriever, checker, _ = build_graph(
        documents=[rejected, accepted],
        documents_by_query={
            "주거비 지원": [rejected, accepted],
        },
        verdicts={"REJECTED": "indirect", "ACCEPTED": "direct_fit"},
        max_retries=3,
        retriever_search_k=1,
    )

    result = graph.generate_answer(
        user_input="주거비 지원",
        user_profile={},
        thread_id="retry-user",
    )

    assert result.retrieved_policy_ids == ["ACCEPTED"]
    assert [call.query for call in retriever.calls] == [
        "주거비 지원",
        "주거비 지원",
    ]
    assert retriever.calls[0].excluded_policy_ids == frozenset()
    assert retriever.calls[1].excluded_policy_ids == frozenset({"REJECTED"})
    assert checker.calls == ["REJECTED", "ACCEPTED"]


def test_retry_limit_returns_empty_documents_after_initial_plus_retries():
    rejected = policy_document("REJECTED")
    graph, _, retriever, checker, answer_generator = build_graph(
        documents=[rejected],
        documents_by_query={
            "정책 찾아줘": [rejected],
        },
        verdicts={"REJECTED": "indirect"},
        max_retries=3,
    )

    result = graph.generate_answer(
        user_input="정책 찾아줘",
        user_profile={},
        thread_id="retry-limit-user",
    )

    assert result.retrieved_policy_ids == []
    assert len(retriever.calls) == 2
    assert [call.query for call in retriever.calls] == ["정책 찾아줘"] * 2
    assert checker.calls == ["REJECTED"]
    assert retriever.calls[1].excluded_policy_ids == frozenset({"REJECTED"})
    assert answer_generator.calls[0]["documents"] == []


def test_failed_new_search_preserves_active_policy_for_later_follow_up():
    active = policy_document("ACTIVE")
    rejected = policy_document("REJECTED")
    graph, _, retriever, checker, answer_generator = build_graph(
        documents=[active, rejected],
        documents_by_query={
            "첫 정책": [active],
            "다른 정책": [rejected],
        },
        verdicts={"ACTIVE": "direct_fit", "REJECTED": "mismatch"},
        max_retries=0,
        force_search_inputs={"다른 정책"},
    )

    first = graph.generate_answer(
        user_input="첫 정책",
        user_profile={},
        thread_id="active-policy-user",
    )
    failed_search = graph.generate_answer(
        user_input="다른 정책",
        user_profile={},
        thread_id="active-policy-user",
    )

    async def collect_follow_up():
        return [
            json.loads(event.removeprefix("data: ").strip())
            async for event in graph.stream_answer(
                user_input="신청 방법은?",
                user_profile={},
                thread_id="active-policy-user",
            )
        ]

    follow_up_events = asyncio.run(collect_follow_up())

    assert first.retrieved_policy_ids == ["ACTIVE"]
    assert failed_search.retrieved_policy_ids == []
    assert len(retriever.calls) == 2
    assert retriever.calls[1].excluded_policy_ids == frozenset({"ACTIVE"})
    assert checker.calls == ["ACTIVE", "REJECTED"]
    assert [
        document.metadata["plcyNo"]
        for document in answer_generator.calls[2]["documents"]
    ] == ["ACTIVE"]
    assert follow_up_events[0]["type"] == "metadata"
    assert follow_up_events[0]["data"]["retrieved_policy_ids"] == ["ACTIVE"]


def test_rejected_policy_exclusion_resets_on_next_user_turn():
    rejected = policy_document("REJECTED")
    accepted = policy_document("ACCEPTED")
    graph, _, retriever, checker, _ = build_graph(
        documents=[rejected, accepted],
        documents_by_query={
            "주거비 지원": [rejected, accepted],
            "조건 바꿔서 다시 검색": [rejected],
        },
        verdicts={"REJECTED": "indirect", "ACCEPTED": "direct_fit"},
        max_retries=1,
        retriever_search_k=1,
        force_search_inputs={"조건 바꿔서 다시 검색"},
    )

    graph.generate_answer(
        user_input="주거비 지원",
        user_profile={},
        thread_id="turn-local-exclusion-user",
    )
    graph.generate_answer(
        user_input="조건 바꿔서 다시 검색",
        user_profile={},
        thread_id="turn-local-exclusion-user",
    )

    assert retriever.calls[1].excluded_policy_ids == frozenset({"REJECTED"})
    assert retriever.calls[2].excluded_policy_ids == frozenset({"ACCEPTED"})
    assert retriever.calls[3].excluded_policy_ids == frozenset({
        "ACCEPTED",
        "REJECTED",
    })
    assert checker.calls == ["REJECTED", "ACCEPTED", "REJECTED"]


def test_follow_up_reuses_accepted_documents_and_chat_history():
    graph, _, retriever, checker, answer_generator = build_graph()

    graph.generate_answer(
        user_input="월세 정책을 알려줘.",
        user_profile={},
        thread_id="history-user",
    )
    result = graph.generate_answer(
        user_input="신청 방법은?",
        user_profile={},
        thread_id="history-user",
    )

    assert result.answer == "답변: 신청 방법은?"
    assert len(retriever.calls) == 1
    assert len(checker.calls) == 1
    assert [
        (type(message), message.content)
        for message in answer_generator.calls[1]["chat_history"]
    ] == [
        (HumanMessage, "월세 정책을 알려줘."),
        (AIMessage, "답변: 월세 정책을 알려줘."),
    ]


def test_graph_config_skips_langfuse_when_disabled(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_PROVIDER", raising=False)
    monkeypatch.delenv("LANGFUSE_TRACING", raising=False)
    graph, *_ = build_graph()

    assert graph._build_graph_config(
        "thread-1",
        trace_user_id="user-1",
    ) == {"configurable": {"thread_id": "thread-1"}}


def test_graph_config_adds_langfuse_callback_metadata(monkeypatch):
    class FakeCallbackHandler:
        pass

    def trace_config_factory(**kwargs):
        return {
            "callbacks": [FakeCallbackHandler()],
            "tags": kwargs["tags"],
            "metadata": {
                "langfuse_user_id": kwargs["user_id"],
                **kwargs["metadata"],
            },
        }

    graph, *_ = build_graph(trace_config_factory=trace_config_factory)

    config = graph._build_graph_config(
        "thread-1",
        trace_user_id="user-1",
        trace_tags=["rag-test"],
        trace_metadata={"case_id": "case-1"},
    )

    assert isinstance(config["callbacks"][0], FakeCallbackHandler)
    assert config["tags"] == ["rag-test"]
    assert config["metadata"]["langfuse_user_id"] == "user-1"
    assert config["metadata"]["case_id"] == "case-1"


def test_stream_answer_exposes_only_checker_accepted_policies():
    documents = [policy_document("HIGH"), policy_document("LOW")]
    graph, *_ = build_graph(
        documents=documents,
        verdicts={"HIGH": "direct_fit", "LOW": "indirect"},
    )

    async def collect_events():
        return [
            json.loads(event.removeprefix("data: ").strip())
            async for event in graph.stream_answer(
                user_input="지원 정책",
                user_profile={},
                thread_id="stream-user",
            )
        ]

    events = asyncio.run(collect_events())

    assert [event["type"] for event in events] == [
        "metadata",
        "chunk",
        "done",
    ]
    assert events[0]["data"]["retrieved_policy_ids"] == ["HIGH"]
    assert events[1]["data"] == "답변: 지원 정책"


def test_delete_conversation_removes_persisted_documents():
    graph, _, retriever, *_ = build_graph()

    graph.generate_answer(
        user_input="첫 질문",
        user_profile={},
        thread_id="delete-user",
    )
    graph.delete_conversation("delete-user")
    graph.generate_answer(
        user_input="삭제 후 질문",
        user_profile={},
        thread_id="delete-user",
    )

    assert len(retriever.calls) == 2
