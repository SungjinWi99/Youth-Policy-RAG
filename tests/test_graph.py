import asyncio
import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.rag.graph import PolicyRagGraph
from rag.nodes.router import RouterOutput


class FakeRouter:
    def __init__(self):
        self.calls = []

    def decide(self, *, current_question, documents, chat_history):
        self.calls.append({
            "current_question": current_question,
            "documents": list(documents),
            "chat_history": list(chat_history),
        })
        return RouterOutput(
            route="agent" if documents else "retriever",
            route_reason=(
                "활성 문서를 재사용합니다."
                if documents
                else "활성 문서가 없어 검색합니다."
            ),
        )

    async def adecide(self, **kwargs):
        return self.decide(**kwargs)


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, *, query, user_profile, exclude_expired):
        self.calls.append({
            "query": query,
            "user_profile": user_profile,
            "exclude_expired": exclude_expired,
        })
        return self.documents

    async def aretrieve(self, **kwargs):
        self.calls.append(kwargs)
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


def build_graph():
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
    router = FakeRouter()
    retriever = FakeRetriever(documents)
    agent = FakeAgent()
    graph = PolicyRagGraph(
        router=router,
        retriever=retriever,
        agent=agent,
        checkpointer=InMemorySaver(),
    )
    return graph, router, retriever, agent


def test_compiled_graph_has_router_retriever_agent_topology():
    graph, _, _, _ = build_graph()

    compiled = graph.graph.get_graph()
    edges = {
        (edge.source, edge.target, edge.conditional)
        for edge in compiled.edges
    }

    assert set(compiled.nodes) == {
        "__start__",
        "router",
        "retriever",
        "agent",
        "__end__",
    }
    assert edges == {
        ("__start__", "router", False),
        ("router", "retriever", True),
        ("router", "agent", True),
        ("retriever", "agent", False),
        ("agent", "__end__", False),
    }


def test_router_node_returns_only_route_metadata():
    graph, _, retriever, _ = build_graph()

    update = graph._router_node({
        "user_input": "신청 방법은?",
        "user_profile": {},
        "exclude_expired": True,
        "messages": [HumanMessage(content="신청 방법은?")],
        "documents": retriever.documents,
    })

    assert update == {
        "route": "agent",
        "route_reason": "활성 문서를 재사용합니다.",
    }


def test_generate_answer_retrieves_then_returns_public_result():
    graph, router, retriever, agent = build_graph()
    profile = {"age": 27, "region": "서울"}

    result = graph.generate_answer(
        user_input="월세 지원 정책을 알려줘.",
        user_profile=profile,
        user_id="sync-user",
        exclude_expired=True,
    )

    assert result.answer == "답변: 월세 지원 정책을 알려줘."
    assert result.retrieved_policy_ids == ["POLICY-001"]
    assert "서울 청년 월세 지원 정책입니다." in result.contexts[0]
    assert router.calls[0]["documents"] == []
    assert retriever.calls == [{
        "query": "월세 지원 정책을 알려줘.",
        "user_profile": {
            "age": 27,
            "region": "서울",
        },
        "exclude_expired": True,
    }]
    assert agent.calls[0]["documents"] == retriever.documents


def test_follow_up_reuses_documents_and_preserves_chat_history():
    graph, _, retriever, agent = build_graph()
    profile = {}

    graph.generate_answer(
        user_input="월세 정책을 알려줘.",
        user_profile=profile,
        user_id="history-user",
    )
    result = graph.generate_answer(
        user_input="신청 방법은?",
        user_profile=profile,
        user_id="history-user",
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
            user_id="async-user",
            exclude_expired=False,
        )
    )

    assert result.answer == "답변: 비동기 질문"
    assert result.retrieved_policy_ids == ["POLICY-001"]
    assert retriever.calls[0]["query"] == "비동기 질문"
    assert agent.calls[0]["user_input"] == "비동기 질문"


def test_stream_answer_emits_metadata_chunk_and_done_for_both_routes():
    graph, _, retriever, _ = build_graph()
    profile = {}

    async def collect_events(question):
        return [
            json.loads(event.removeprefix("data: ").strip())
            async for event in graph.stream_answer(
                user_input=question,
                user_profile=profile,
                user_id="stream-user",
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
        user_id="delete-user",
    )
    graph.delete_conversation("delete-user")
    graph.generate_answer(
        user_input="삭제 후 질문",
        user_profile=profile,
        user_id="delete-user",
    )

    assert len(retriever.calls) == 2
