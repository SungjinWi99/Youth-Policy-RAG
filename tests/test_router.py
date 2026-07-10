import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src.rag.nodes.turn_planner import TurnPlan, TurnPlanner


class FakeStructuredLlm:
    def __init__(self, response: TurnPlan):
        self.response = response
        self.schema = None
        self.prompt_values = []

    def with_structured_output(self, schema):
        self.schema = schema

        def respond(prompt_value):
            self.prompt_values.append(prompt_value)
            return self.response

        return RunnableLambda(respond)


@pytest.fixture
def policy_document():
    return Document(
        page_content="서울 청년 월세 지원 정책입니다.",
        metadata={
            "plcyNo": "POLICY-001",
            "lclsfNm": "주거",
            "region": "서울",
            "agePolicy": "range",
            "sprtTrgtMinAge": 19,
            "sprtTrgtMaxAge": 34,
            "incomePolicy": "all",
            "aplyYmd": "2026-07-01 ~ 2026-07-31",
        },
    )


def test_router_output_rejects_unknown_route():
    with pytest.raises(ValidationError):
        TurnPlan(
            route="reuse",
            answer_strategy="brief_reply",
            retrieval_queries=[],
            route_reason="지원하지 않는 분기입니다.",
        )


def test_turn_plan_rejects_unknown_answer_strategy():
    with pytest.raises(ValidationError):
        TurnPlan(
            route="agent",
            answer_strategy="unknown",
            retrieval_queries=[],
            route_reason="지원하지 않는 전략입니다.",
        )


def test_build_chain_input_formats_documents_and_copies_history(policy_document):
    planner = TurnPlanner(
        FakeStructuredLlm(
            TurnPlan(
                route="agent",
                answer_strategy="focused_followup",
                retrieval_queries=[],
                route_reason="현재 문서로 답변할 수 있습니다.",
            )
        )
    )
    history = [HumanMessage(content="서울 주거 정책을 찾아줘.")]

    chain_input = planner._build_chain_input(
        current_question="신청 기간은 언제야?",
        user_profile={"age": 27, "region": "서울"},
        documents=[policy_document],
        chat_history=history,
    )

    assert chain_input["current_question"] == "신청 기간은 언제야?"
    assert chain_input["chat_history"] == history
    assert chain_input["chat_history"] is not history
    assert "서울 청년 월세 지원 정책입니다." in chain_input["documents"]
    assert "POLICY-001" in chain_input["documents"]
    assert "지원 연령: 19세 ~ 34세" in chain_input["documents"]
    assert "소득 조건: 제한 없음" in chain_input["documents"]
    assert "나이: 27" in chain_input["user_profile"]
    assert "주거지: 서울" in chain_input["user_profile"]


def test_build_chain_input_normalizes_missing_optional_context():
    planner = TurnPlanner(
        FakeStructuredLlm(
            TurnPlan(
                route="retriever",
                answer_strategy="policy_recommendation",
                retrieval_queries=["청년 교통비 지원 정책"],
                route_reason="활성 문서가 없습니다.",
            )
        )
    )

    chain_input = planner._build_chain_input(
        current_question="교통비 지원 정책을 찾아줘.",
        user_profile={},
    )

    assert chain_input == {
        "documents": [],
        "current_question": "교통비 지원 정책을 찾아줘.",
        "user_profile": (
            "나이: 미입력\n"
            "성별: 미입력\n"
            "소득수준: 미입력\n"
            "주거지: 미입력\n"
            "직업: 미입력"
        ),
        "chat_history": [],
    }


def test_decide_uses_structured_output_and_renders_router_context(policy_document):
    llm = FakeStructuredLlm(
        TurnPlan(
            route="agent",
            answer_strategy="focused_followup",
            retrieval_queries=[],
            route_reason="현재 월세 지원 문서에 신청 기간이 포함되어 있습니다.",
        )
    )
    planner = TurnPlanner(llm)
    history = [
        HumanMessage(content="서울 월세 지원 정책을 알려줘."),
        AIMessage(content="서울 청년 월세 지원 정책이 있습니다."),
    ]

    result = planner.decide(
        current_question="신청 기간은 언제야?",
        user_profile={"region": "서울"},
        documents=[policy_document],
        chat_history=history,
    )

    assert result.route == "agent"
    assert result.answer_strategy == "focused_followup"
    assert llm.schema is TurnPlan
    assert len(llm.prompt_values) == 1

    messages = llm.prompt_values[0].to_messages()
    assert len(messages) == 4
    assert messages[1:3] == history
    assert "과거 대화:" not in messages[-1].content
    assert "신청 기간은 언제야?" in messages[-1].content
    assert "주거지: 서울" in messages[-1].content
    assert "서울 청년 월세 지원 정책입니다." in messages[-1].content
    assert "POLICY-001" in messages[-1].content


def test_decide_does_not_force_retriever_when_documents_are_empty():
    llm = FakeStructuredLlm(
        TurnPlan(
            route="agent",
            answer_strategy="brief_reply",
            retrieval_queries=[],
            route_reason="인사 발화라 짧게 응답합니다.",
        )
    )
    planner = TurnPlanner(llm)

    result = planner.decide(
        current_question="안녕",
        user_profile={},
        documents=[],
        chat_history=[],
    )

    assert result == TurnPlan(
        route="agent",
        answer_strategy="brief_reply",
        retrieval_queries=[],
        route_reason="인사 발화라 짧게 응답합니다.",
    )
    assert len(llm.prompt_values) == 1


def test_decide_guards_missing_retrieval_queries_for_retriever():
    llm = FakeStructuredLlm(
        TurnPlan(
            route="retriever",
            answer_strategy="policy_recommendation",
            retrieval_queries=[],
            route_reason="새 검색이 필요합니다.",
        )
    )
    planner = TurnPlanner(llm)

    result = planner.decide(
        current_question="서울 월세 지원 있어?",
        user_profile={},
        documents=[],
        chat_history=[],
    )

    assert result.retrieval_queries == ["서울 월세 지원 있어?"]


def test_decide_guards_impossible_agent_followup_without_documents():
    llm = FakeStructuredLlm(
        TurnPlan(
            route="agent",
            answer_strategy="focused_followup",
            retrieval_queries=["무시될 검색어"],
            route_reason="잘못된 후속 질문 판단입니다.",
        )
    )
    planner = TurnPlanner(llm)

    result = planner.decide(
        current_question="신청 기간은?",
        user_profile={},
        documents=[],
        chat_history=[],
    )

    assert result.route == "agent"
    assert result.answer_strategy == "clarifying_question"
    assert result.retrieval_queries == []


def test_adecide_invokes_the_async_chain(policy_document):
    llm = FakeStructuredLlm(
        TurnPlan(
            route="retriever",
            answer_strategy="policy_recommendation",
            retrieval_queries=["청년 교통비 지원 정책"],
            route_reason="현재 질문은 기존 월세 문서와 다른 정책을 요청합니다.",
        )
    )
    planner = TurnPlanner(llm)

    result = asyncio.run(
        planner.adecide(
            current_question="청년 교통비 지원 정책을 찾아줘.",
            user_profile={},
            documents=[policy_document],
            chat_history=[],
        )
    )

    assert result.route == "retriever"
    assert len(llm.prompt_values) == 1
    assert "청년 교통비 지원 정책을 찾아줘." in (
        llm.prompt_values[0].to_messages()[-1].content
    )
