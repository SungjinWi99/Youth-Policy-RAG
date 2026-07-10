import asyncio
import json
import os
import sys
import types

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from src.rag.graph import PolicyRagGraph
from src.rag.nodes.agent import PolicyAgent
from src.rag.prompts import ANSWER_STRATEGY_INSTRUCTIONS
from src.rag.nodes.turn_planner import TurnPlan


class FakePlanner:
    def __init__(self):
        self.calls = []

    def decide(self, *, current_question, user_profile, documents, chat_history):
        self.calls.append({
            "current_question": current_question,
            "user_profile": user_profile,
            "documents": list(documents),
            "chat_history": list(chat_history),
        })
        return TurnPlan(
            route="agent" if documents else "retriever",
            answer_strategy=(
                "focused_followup"
                if documents
                else "policy_recommendation"
            ),
            retrieval_queries=[] if documents else [current_question],
            route_reason=(
                "활성 문서를 재사용합니다."
                if documents
                else "활성 문서가 없어 검색합니다."
            ),
        )

    async def adecide(self, **kwargs):
        return self.decide(**kwargs)


class FakeRetriever:
    def __init__(self, documents, documents_by_query=None):
        self.documents = documents
        self.documents_by_query = documents_by_query
        self.calls = []

    def retrieve(self, *, query, user_profile, exclude_expired):
        self.calls.append({
            "query": query,
            "user_profile": user_profile,
            "exclude_expired": exclude_expired,
        })
        if self.documents_by_query is not None:
            return self.documents_by_query.get(query, [])
        return self.documents

    async def aretrieve(self, **kwargs):
        self.calls.append(kwargs)
        if self.documents_by_query is not None:
            return self.documents_by_query.get(kwargs["query"], [])
        return self.documents


class FakeAgent:
    def __init__(self):
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return f"답변: {kwargs['user_input']}"

    async def ainvoke(self, **kwargs):
        self.calls.append(kwargs)
        return f"답변: {kwargs['user_input']}"


def build_graph(
    *,
    retriever=None,
    router_history_window=6,
    agent_history_window=10,
):
    documents = [
        Document(
            page_content="서울 청년 월세 지원 정책입니다.",
            metadata={
                "plcyNo": "POLICY-001",
                "agePolicy": "all",
                "incomePolicy": "all",
            },
        )
    ]
    planner = FakePlanner()
    retriever = retriever or FakeRetriever(documents)
    agent = FakeAgent()
    graph = PolicyRagGraph(
        planner=planner,
        retriever=retriever,
        agent=agent,
        checkpointer=InMemorySaver(),
        router_history_window=router_history_window,
        agent_history_window=agent_history_window,
    )
    return graph, planner, retriever, agent


def test_compiled_graph_has_planner_retriever_agent_topology():
    graph, _, _, _ = build_graph()

    compiled = graph.graph.get_graph()
    edges = {
        (edge.source, edge.target, edge.conditional)
        for edge in compiled.edges
    }

    assert set(compiled.nodes) == {
        "__start__",
        "planner",
        "retriever",
        "agent",
        "__end__",
    }
    assert edges == {
        ("__start__", "planner", False),
        ("planner", "retriever", True),
        ("planner", "agent", True),
        ("retriever", "agent", False),
        ("agent", "__end__", False),
    }


def test_planner_node_returns_turn_metadata():
    graph, _, retriever, _ = build_graph()

    update = graph._planner_node({
        "user_input": "신청 방법은?",
        "user_profile": {},
        "exclude_expired": True,
        "messages": [HumanMessage(content="신청 방법은?")],
        "documents": retriever.documents,
    })

    assert update == {
        "route": "agent",
        "route_reason": "활성 문서를 재사용합니다.",
        "answer_strategy": "focused_followup",
        "retrieval_queries": [],
    }


def test_graph_config_skips_langfuse_when_disabled(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_PROVIDER", raising=False)
    monkeypatch.delenv("LANGFUSE_TRACING", raising=False)
    graph, _, _, _ = build_graph()

    config = graph._build_graph_config(
        "thread-1",
        trace_user_id="user-1",
    )

    assert config == {"configurable": {"thread_id": "thread-1"}}


def test_graph_config_adds_langfuse_callback_metadata(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_PROVIDER", raising=False)

    class FakeCallbackHandler:
        pass

    langfuse_module = types.ModuleType("langfuse")
    langchain_module = types.ModuleType("langfuse.langchain")
    langchain_module.CallbackHandler = FakeCallbackHandler
    langfuse_module.langchain = langchain_module
    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", langchain_module)
    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    graph, _, _, _ = build_graph()
    config = graph._build_graph_config(
        "thread-1",
        trace_user_id="user-1",
        trace_tags=["rag-test"],
        trace_metadata={"case_id": "case-1"},
    )

    assert isinstance(config["callbacks"][0], FakeCallbackHandler)
    assert config["tags"] == ["rag-test"]
    assert config["metadata"] == {
        "langfuse_user_id": "user-1",
        "langfuse_session_id": "thread-1",
        "langfuse_tags": ["rag-test"],
        "langgraph_thread_id": "thread-1",
        "case_id": "case-1",
    }


def test_generate_answer_retrieves_then_returns_public_result():
    graph, planner, retriever, agent = build_graph()
    profile = {"age": 27, "region": "서울"}

    result = graph.generate_answer(
        user_input="월세 지원 정책을 알려줘.",
        user_profile=profile,
        thread_id="sync-user",
        exclude_expired=True,
    )

    assert result.answer == "답변: 월세 지원 정책을 알려줘."
    assert result.retrieved_policy_ids == ["POLICY-001"]
    assert "서울 청년 월세 지원 정책입니다." in result.contexts[0]
    assert planner.calls[0]["documents"] == []
    assert retriever.calls == [{
        "query": "월세 지원 정책을 알려줘.",
        "user_profile": {
            "age": 27,
            "region": "서울",
        },
        "exclude_expired": True,
    }]
    assert agent.calls[0]["documents"] == retriever.documents
    assert agent.calls[0]["answer_strategy"] == "policy_recommendation"


def test_follow_up_reuses_documents_and_preserves_chat_history():
    graph, _, retriever, agent = build_graph()
    profile = {}

    graph.generate_answer(
        user_input="월세 정책을 알려줘.",
        user_profile=profile,
        thread_id="history-user",
    )
    result = graph.generate_answer(
        user_input="신청 방법은?",
        user_profile=profile,
        thread_id="history-user",
    )

    assert result.answer == "답변: 신청 방법은?"
    assert len(retriever.calls) == 1
    assert [
        (type(message), message.content)
        for message in agent.calls[1]["chat_history"]
    ] == [
        (HumanMessage, "월세 정책을 알려줘."),
        (AIMessage, "답변: 월세 정책을 알려줘."),
    ]


def test_agenerate_answer_uses_async_nodes():
    graph, _, retriever, agent = build_graph()

    result = asyncio.run(
        graph.agenerate_answer(
            user_input="비동기 질문",
            user_profile={},
            thread_id="async-user",
            exclude_expired=False,
        )
    )

    assert result.answer == "답변: 비동기 질문"
    assert result.retrieved_policy_ids == ["POLICY-001"]
    assert retriever.calls[0]["query"] == "비동기 질문"
    assert agent.calls[0]["user_input"] == "비동기 질문"


def test_policy_agent_injects_selected_answer_strategy_instruction():
    agent = PolicyAgent(RunnableLambda(lambda _: "답변"))

    chain_input = agent.build_chain_input(
        user_input="신청 기간은?",
        user_profile={},
        documents=[],
        chat_history=[],
        answer_strategy="focused_followup",
    )

    assert chain_input["answer_strategy"] == "focused_followup"
    assert (
        chain_input["answer_strategy_instruction"]
        == ANSWER_STRATEGY_INSTRUCTIONS["focused_followup"]
    )


def test_retrieve_node_uses_fallback_queries_until_documents_found():
    documents = [
        Document(
            page_content="서울 청년 주거 안정 지원 정책입니다.",
            metadata={"plcyNo": "POLICY-002"},
        )
    ]
    retriever = FakeRetriever(
        documents=[],
        documents_by_query={
            "서울 청년 월세 지원 정책": [],
            "서울 청년 주거 안정 지원 정책": documents,
        },
    )
    graph, _, _, _ = build_graph(retriever=retriever)

    update = graph._retrieve_node({
        "user_input": "월세 지원 있어?",
        "user_profile": {"region": "서울"},
        "exclude_expired": True,
        "retrieval_queries": [
            "서울 청년 월세 지원 정책",
            "서울 청년 주거 안정 지원 정책",
        ],
    })

    assert update == {"documents": documents}
    assert [call["query"] for call in retriever.calls] == [
        "서울 청년 월세 지원 정책",
        "서울 청년 주거 안정 지원 정책",
    ]


def test_router_and_agent_history_windows_are_applied_separately():
    graph, planner, _, agent = build_graph(
        router_history_window=2,
        agent_history_window=3,
    )
    messages = [
        HumanMessage(content="1"),
        AIMessage(content="2"),
        HumanMessage(content="3"),
        AIMessage(content="4"),
        HumanMessage(content="현재 질문"),
    ]

    graph._planner_node({
        "user_input": "현재 질문",
        "user_profile": {},
        "exclude_expired": True,
        "messages": messages,
    })
    graph._agent_node({
        "user_input": "현재 질문",
        "user_profile": {},
        "exclude_expired": True,
        "messages": messages,
        "documents": [],
        "answer_strategy": "brief_reply",
    })

    assert [message.content for message in planner.calls[-1]["chat_history"]] == [
        "3",
        "4",
    ]
    assert [message.content for message in agent.calls[-1]["chat_history"]] == [
        "2",
        "3",
        "4",
    ]


def test_stream_answer_emits_metadata_chunk_and_done_for_both_routes():
    graph, _, retriever, _ = build_graph()
    profile = {}

    async def collect_events(question):
        return [
            json.loads(event.removeprefix("data: ").strip())
            async for event in graph.stream_answer(
                user_input=question,
                user_profile=profile,
                thread_id="stream-user",
            )
        ]

    first_events = asyncio.run(collect_events("월세 정책을 알려줘."))
    second_events = asyncio.run(collect_events("신청 기간은?"))

    assert [event["type"] for event in first_events] == [
        "metadata",
        "chunk",
        "done",
    ]
    assert [event["type"] for event in second_events] == [
        "metadata",
        "chunk",
        "done",
    ]
    assert first_events[0]["data"]["retrieved_policy_ids"] == ["POLICY-001"]
    assert second_events[1]["data"] == "답변: 신청 기간은?"
    assert len(retriever.calls) == 1


def test_delete_conversation_removes_persisted_documents():
    graph, _, retriever, _ = build_graph()
    profile = {}

    graph.generate_answer(
        user_input="첫 질문",
        user_profile=profile,
        thread_id="delete-user",
    )
    graph.delete_conversation("delete-user")
    graph.generate_answer(
        user_input="삭제 후 질문",
        user_profile=profile,
        thread_id="delete-user",
    )

    assert len(retriever.calls) == 2
