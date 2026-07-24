import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src.rag.nodes.answer_generator import (
    GENERATOR_SYSTEM_PROMPT,
    _format_policy_assessments,
)
from src.rag.nodes.policy_checker import (
    POLICY_CHECKER_SYSTEM_PROMPT,
    PolicyCheckerOutput,
    make_policy_checker_node,
)
from src.rag.nodes.policy_selector import make_policy_selector_node
from src.rag.nodes.retrieval_planner import (
    PLANNER_SYSTEM_PROMPT,
    PlannerOutput,
    make_retrieval_planner_node,
)


class FakeStructuredLlm:
    def __init__(self, response):
        self.response = response
        self.schema = None
        self.prompt_values = []

    def with_structured_output(self, schema):
        self.schema = schema

        def respond(prompt_value):
            self.prompt_values.append(prompt_value)
            return self.response

        async def arespond(prompt_value):
            return respond(prompt_value)

        return RunnableLambda(respond, afunc=arespond)


@pytest.fixture
def policy_document():
    return Document(
        page_content="서울 청년 월세 지원 정책입니다.",
        metadata={
            "plcyNo": "POLICY-001",
            "zipCd": "11110",
            "agePolicy": "all",
            "incomePolicy": "all",
        },
    )


def test_planner_output_rejects_invalid_fields():
    with pytest.raises(ValidationError):
        PlannerOutput(
            user_requirement="정책 검색",
            needs_retrieval="not-a-boolean",
            retrieval_reason="검색 필요",
            retrieval_query="query",
        )


def test_retrieval_planner_renders_documents_profile_and_history(
    policy_document,
):
    llm = FakeStructuredLlm(PlannerOutput(
        user_requirement="서울 청년의 월세 신청 방법 확인",
        needs_retrieval=False,
        retrieval_reason="활성 정책으로 답할 수 있습니다.",
        retrieval_query="",
    ))
    planner = make_retrieval_planner_node(llm, history_window_size=2)
    history = [
        HumanMessage(content="월세 정책을 찾아줘."),
        HumanMessage(content="신청 방법은?"),
    ]

    result = planner.invoke({
        "user_input": "신청 방법은?",
        "user_profile": {"age": 27, "region": "서울"},
        "exclude_expired": True,
        "messages": history,
        "active_policies": [policy_document],
        "documents": [],
        "retrieval_count": 0,
        "retrieved_policies": [],
        "checked_policies": [],
    })

    assert result["needs_retrieval"] is False
    assert llm.schema is PlannerOutput
    rendered = llm.prompt_values[0].to_messages()
    assert "서울 청년 월세 지원 정책입니다." in rendered[-1].content
    assert "주거지: 서울" in rendered[-1].content


def test_retrieval_planner_preserves_same_query_on_retry():
    llm = FakeStructuredLlm(PlannerOutput(
        user_requirement="청년 주거비 지원 탐색",
        needs_retrieval=False,
        retrieval_reason="",
        retrieval_query="청년 주거비 지원",
    ))
    planner = make_retrieval_planner_node(llm, history_window_size=2)

    result = planner.invoke({
        "user_input": "주거비 지원",
        "user_profile": {},
        "exclude_expired": True,
        "messages": [HumanMessage(content="주거비 지원")],
        "documents": [],
        "retrieval_count": 1,
        "retrieval_query": "청년 주거비 지원",
        "retrieved_policies": [Document(
            page_content="직전 탈락 정책",
            metadata={"plcyNo": "REJECTED"},
        )],
        "checked_policies": [{
            "verdict": "indirect",
            "document": Document(
                page_content="직전 탈락 정책",
                metadata={"plcyNo": "REJECTED"},
            ),
            "reasoning": "요청 목적과 간접적으로만 관련됩니다.",
            "retrieval_rank": 1,
            "retrieval_round": 1,
        }],
    })

    assert result["needs_retrieval"] is True
    assert result["retrieval_query"] == "청년 주거비 지원"


def test_retrieval_planner_stops_repeating_same_empty_search():
    llm = FakeStructuredLlm(PlannerOutput(
        user_requirement="청년 주거비 지원 탐색",
        needs_retrieval=True,
        retrieval_reason="계속 검색합니다.",
        retrieval_query="청년 주거비 지원",
    ))
    planner = make_retrieval_planner_node(llm, history_window_size=2)

    result = planner.invoke({
        "user_input": "주거비 지원",
        "user_profile": {},
        "exclude_expired": True,
        "messages": [HumanMessage(content="주거비 지원")],
        "documents": [],
        "retrieval_count": 2,
        "retrieval_query": "청년 주거비 지원",
        "retrieved_policies": [],
        "checked_policies": [],
    })

    assert result["needs_retrieval"] is False
    assert result["retrieval_query"] == ""
    assert "동일 Query의 추가 검색 결과가 없어" in (
        result["retrieval_reason"]
    )


def test_retrieval_planner_allows_meaningful_query_change_after_empty_search():
    llm = FakeStructuredLlm(PlannerOutput(
        user_requirement="청년 주거비 지원 탐색",
        needs_retrieval=True,
        retrieval_reason="지원 목적을 넓혀 검색합니다.",
        retrieval_query="청년 주거 안정 지원",
    ))
    planner = make_retrieval_planner_node(llm, history_window_size=2)

    result = planner.invoke({
        "user_input": "주거비 지원",
        "user_profile": {},
        "exclude_expired": True,
        "messages": [HumanMessage(content="주거비 지원")],
        "documents": [],
        "retrieval_count": 2,
        "retrieval_query": "청년 주거비 지원",
        "retrieved_policies": [],
        "checked_policies": [],
    })

    assert result["needs_retrieval"] is True
    assert result["retrieval_query"] == "청년 주거 안정 지원"


def test_planner_does_not_retrieve_for_missing_answer_details():
    assert "상세정보 부족은 다른 문서를 검색할 이유가 아니다" in (
        PLANNER_SYSTEM_PROMPT
    )
    assert "상세정보 부족은 재검색 근거로 사용하지 않는다" in (
        PLANNER_SYSTEM_PROMPT
    )


def test_policy_checker_returns_reducer_friendly_checked_list(policy_document):
    llm = FakeStructuredLlm(PolicyCheckerOutput(
        verdict="direct_fit",
        reasoning="질문과 직접 관련됩니다.",
    ))
    checker = make_policy_checker_node(llm)

    result = checker.invoke({
        "current_question": "신청 방법까지 알려줘.",
        "user_requirement": "서울 월세 지원",
        "user_profile": {"age": 27, "income": 3200, "region": "서울"},
        "policy": policy_document,
    })

    assert llm.schema is PolicyCheckerOutput
    assert result["checked_policies"] == [{
        "verdict": "direct_fit",
        "document": policy_document,
        "reasoning": "질문과 직접 관련됩니다.",
        "retrieval_rank": 1,
        "retrieval_round": 1,
    }]
    checker_prompt = llm.prompt_values[0].to_messages()[-1].content
    assert "POLICY-001" in checker_prompt
    assert "신청 방법까지 알려줘." in checker_prompt
    assert "소득수준: 3200" in checker_prompt
    assert "주거지: 서울" in checker_prompt


def test_policy_checker_async_path(policy_document):
    checker = make_policy_checker_node(
        FakeStructuredLlm(PolicyCheckerOutput(
            verdict="fit_needs_clarification",
            reasoning="부분 조건 확인이 필요합니다.",
        ))
    )

    result = asyncio.run(checker.ainvoke({
        "current_question": "서울 월세 지원이 있어?",
        "user_requirement": "서울 월세 지원",
        "user_profile": {"region": "서울"},
        "policy": policy_document,
    }))

    assert (
        result["checked_policies"][0]["verdict"]
        == "fit_needs_clarification"
    )


def test_policy_checker_verdict_uses_policy_fit_not_answer_detail_coverage():
    assert "세부 정보가 부족해도" in POLICY_CHECKER_SYSTEM_PROMPT
    assert "정책 적합성 verdict를 낮추지 않는다" in POLICY_CHECKER_SYSTEM_PROMPT
    assert "소득 정보가 불명확하면 그것만으로 정책을 탈락시키거나" in (
        POLICY_CHECKER_SYSTEM_PROMPT
    )


def test_policy_selector_accepts_fit_verdicts_in_priority_order(
    policy_document,
):
    higher = Document(
        page_content="더 적합한 정책",
        metadata={"plcyNo": "POLICY-002"},
    )
    selector = make_policy_selector_node()

    result = selector({
        "checked_policies": [
            {
                "verdict": "fit_needs_clarification",
                "document": policy_document,
                "reasoning": "경계값",
                "retrieval_rank": 1,
            },
            {
                "verdict": "direct_fit",
                "document": higher,
                "reasoning": "높은 점수",
                "retrieval_rank": 2,
            },
            {
                "verdict": "indirect",
                "document": Document(
                    page_content="탈락",
                    metadata={"plcyNo": "POLICY-003"},
                ),
                "reasoning": "간접 관련",
                "retrieval_rank": 3,
            },
        ]
    })

    assert [
        document.metadata["plcyNo"]
        for document in result["documents"]
    ] == ["POLICY-002", "POLICY-001"]


def test_policy_checker_output_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        PolicyCheckerOutput(
            verdict="unknown",
            reasoning="알 수 없는 결과",
        )


def test_answer_generator_preserves_clarification_verdict(policy_document):
    assessments = _format_policy_assessments([
        {
            "verdict": "fit_needs_clarification",
            "document": policy_document,
            "reasoning": "가구소득 기준을 추가로 확인해야 합니다.",
            "retrieval_rank": 1,
        },
        {
            "verdict": "indirect",
            "document": Document(
                page_content="간접 정책",
                metadata={"plcyNo": "POLICY-002"},
            ),
            "reasoning": "핵심 목적과 다릅니다.",
            "retrieval_rank": 2,
        },
    ])

    assert "POLICY-001" in assessments
    assert "fit_needs_clarification" in assessments
    assert "가구소득 기준을 추가로 확인해야 합니다." in assessments
    assert "POLICY-002" not in assessments
    assert "사용자가 자격을 충족한다고 확정하지 않는다" in (
        GENERATOR_SYSTEM_PROMPT
    )


def test_clarification_verdict_survives_into_later_turn(policy_document):
    selector = make_policy_selector_node()
    result = selector({
        "checked_policies": [{
            "verdict": "fit_needs_clarification",
            "document": policy_document,
            "reasoning": "가구소득 기준을 추가로 확인해야 합니다.",
            "retrieval_rank": 1,
            "retrieval_round": 1,
        }]
    })

    assessments = _format_policy_assessments(
        [],
        result["documents"],
    )

    assert "POLICY-001" in assessments
    assert "fit_needs_clarification" in assessments
    assert "가구소득 기준을 추가로 확인해야 합니다." in assessments
