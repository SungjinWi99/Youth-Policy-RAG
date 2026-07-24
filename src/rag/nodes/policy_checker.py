from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from src.rag.state import PolicyVerdict, RAGUserProfile
from src.rag.utils import format_doc, format_user_profile


POLICY_CHECKER_SYSTEM_PROMPT = """
당신은 청년정책 상담 workflow의 Policy Checker이다.
정책 문서 하나가 사용자의 정책 목적, 프로필, 현재 상황에 적합한지를 판단한다.
이 단계는 정책 자체의 적합성을 평가하며, 문서가 질문의 모든 세부 답변을
포함하는지 평가하거나 최종 신청 자격을 확정하는 단계가 아니다.

판단 규칙:
1. 신청 방법, 제출 서류, 일정, 지원 금액 등 사용자가 요구한 세부 정보가
   문서에 없다는 이유로 정책 적합성 verdict를 낮추지 않는다.
2. 정책 목적과 지원 대상이 사용자 요구에 맞으면 세부 정보가 부족해도
   정책 적합성을 기준으로 verdict를 정한다.
3. 소득은 검색 필터가 아닌 적합성 확인용 메타데이터다.
   사용자 소득과 정책 소득 기준의 단위, 개인/가구 기준, 산정 방식이
   명확히 같을 때만 직접 비교한다.
4. 소득 정보가 불명확하면 그것만으로 정책을 탈락시키거나 자격을 확정하지 않는다.
   정책 목적이 맞으면 확인이 필요한 조건을 reasoning에 명시한다.
5. 명백한 지역, 연령, 대상, 정책 목적 충돌만 강한 부적합 근거로 사용한다.

Verdict 기준:
- direct_fit: 정책 목적과 사용자 요구가 직접 일치하고, 알려진 프로필과 명백한 충돌이 없다.
- fit_needs_clarification: 정책 목적은 직접 일치하지만 소득 등 일부 자격 조건을 추가 확인해야 한다.
- indirect: 같은 큰 분야의 정책이지만 핵심 문제를 직접 해결하지 못하고 간접적인 도움만 준다.
- mismatch: 정책 목적이 질문과 무관하거나 알려진 프로필과 명백히 충돌한다.
""".strip()

POLICY_CHECKER_HUMAN_PROMPT = """
현재 질문:
{current_question}

사용자 요구:
{user_requirement}

사용자 프로필:
{user_profile}

검색된 정책:
{retrieved_policy}
""".strip()


prompt = ChatPromptTemplate.from_messages([
    ("system", POLICY_CHECKER_SYSTEM_PROMPT),
    ("human", POLICY_CHECKER_HUMAN_PROMPT),
])


class PolicyCheckerInput(TypedDict):
    current_question: str
    user_requirement: str
    user_profile: RAGUserProfile
    policy: Document
    retrieval_rank: int
    retrieval_round: int


class PolicyCheckerOutput(BaseModel):
    verdict: PolicyVerdict = Field(
        description="정책 적합성에 대한 네 가지 verdict 중 하나.",
    )
    reasoning: str = Field(
        description="점수의 핵심 근거를 한국어 1~2문장으로 설명."
    )


def make_policy_checker_node(llm: BaseChatModel):
    chain = prompt | llm.with_structured_output(PolicyCheckerOutput)

    def build_chain_input(state: PolicyCheckerInput) -> dict:
        return {
            "current_question": state["current_question"],
            "user_requirement": state["user_requirement"],
            "user_profile": format_user_profile(state["user_profile"]),
            "retrieved_policy": format_doc(state["policy"]),
        }

    def build_update(state: PolicyCheckerInput, result) -> dict:
        checked = PolicyCheckerOutput.model_validate(result)
        return {
            "checked_policies": [{
                "verdict": checked.verdict,
                "document": state["policy"],
                "reasoning": checked.reasoning.strip(),
                "retrieval_rank": state.get("retrieval_rank", 1),
                "retrieval_round": state.get("retrieval_round", 1),
            }]
        }

    def policy_checker_node(state: PolicyCheckerInput):
        return build_update(state, chain.invoke(build_chain_input(state)))

    async def apolicy_checker_node(state: PolicyCheckerInput):
        result = await chain.ainvoke(build_chain_input(state))
        return build_update(state, result)

    return RunnableLambda(policy_checker_node, afunc=apolicy_checker_node)
