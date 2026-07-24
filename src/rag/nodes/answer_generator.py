from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from src.rag.state import (
    CHECKER_REASONING_METADATA_KEY,
    CHECKER_VERDICT_METADATA_KEY,
    CheckedPolicy,
    RAGGraphState,
)
from src.rag.utils import format_docs, format_user_profile, window_history


GENERATOR_SYSTEM_PROMPT = """
당신은 청년정책 안내 Answer Generator다.
Policy Checker를 통과한 정책 문서와 사용자 질문에 근거해 답한다.

규칙:
- 첫 1~2문장에서 사용자가 물은 세부 사항에 직접 답한다.
- 정책의 일반 소개는 질문에 필요한 범위에서만 짧게 덧붙인다.
- 신청 방법 질문에는 신청 채널, 절차, 준비 또는 확인 사항을 행동 순서로 제시한다.
- 자격·추천 질문에는 사용자 프로필과 직접 관련된 조건만 연결한다.
- 문서에 없는 신청 방법, 일정, 서류, 자격 조건을 만들어내지 않는다.
- 핵심 정보가 없으면 “제공된 정책 정보만으로는 확인되지 않는다”고 분명히 말하고
  확인할 곳 또는 다음 행동을 안내한다.
- 검증된 정책 문서가 비어 있으면 관련 정책을 찾지 못했다고 알리고,
  정책 내용을 추측해서 만들지 않는다.
- 정책이 여러 개면 질문에 가장 직접적인 정책부터 제시한다.
- 검색 결과 포함과 실제 신청 자격 충족을 구분한다.
- Checker verdict가 fit_needs_clarification이면 확인이 필요한 조건을 그대로 설명하고,
  사용자가 자격을 충족한다고 확정하지 않는다.
- 소득 단위, 개인·가구 기준, 가구원 수 또는 산정 방식이 불명확하면
  소득 조건 충족 여부를 추정하지 말고 추가 확인이 필요하다고 답한다.
""".strip()

GENERATOR_HUMAN_PROMPT = """
사용자 상황과 요구:
{user_requirement}

사용자 질문:
{current_question}

사용자 프로필:
{user_profile}

Checker를 통과한 정책 문서:
<policy_context>
{accepted_policies}
</policy_context>

Checker 판단:
<policy_assessments>
{policy_assessments}
</policy_assessments>
""".strip()


prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATOR_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", GENERATOR_HUMAN_PROMPT),
])


def _format_policy_assessments(
    checked_policies: list[CheckedPolicy],
    documents=None,
) -> str:
    accepted_verdicts = {"direct_fit", "fit_needs_clarification"}
    lines = []
    seen_policy_ids = set()
    for item in checked_policies:
        if item["verdict"] not in accepted_verdicts:
            continue
        policy_id = item["document"].metadata.get("plcyNo", "정책번호 없음")
        seen_policy_ids.add(str(policy_id))
        lines.append(
            f"- {policy_id}: verdict={item['verdict']}, "
            f"reason={item['reasoning']}"
        )
    for document in documents or []:
        policy_id = str(
            document.metadata.get("plcyNo") or "정책번호 없음"
        )
        verdict = document.metadata.get(CHECKER_VERDICT_METADATA_KEY)
        reasoning = document.metadata.get(CHECKER_REASONING_METADATA_KEY)
        if (
            policy_id in seen_policy_ids
            or verdict not in accepted_verdicts
            or not reasoning
        ):
            continue
        lines.append(
            f"- {policy_id}: verdict={verdict}, reason={reasoning}"
        )
    return "\n".join(lines) or "없음"


def make_answer_generator_node(
    llm: BaseChatModel,
    history_window_size: int,
):
    chain = prompt | llm | StrOutputParser()

    def build_chain_input(state: RAGGraphState) -> dict:
        return {
            "user_requirement": state.get(
                "user_requirement",
                state["user_input"],
            ),
            "current_question": state["user_input"],
            "user_profile": format_user_profile(state["user_profile"]),
            "accepted_policies": (
                format_docs(state.get("documents", [])) or "없음"
            ),
            "policy_assessments": _format_policy_assessments(
                state.get("checked_policies", []),
                state.get("documents", []),
            ),
            "chat_history": window_history(
                state.get("messages", []),
                history_window_size,
            ),
        }

    def build_update(answer: str) -> dict:
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    def answer_generator_node(state: RAGGraphState):
        return build_update(chain.invoke(build_chain_input(state)))

    async def aanswer_generator_node(state: RAGGraphState):
        answer = await chain.ainvoke(build_chain_input(state))
        return build_update(answer)

    return RunnableLambda(
        answer_generator_node,
        afunc=aanswer_generator_node,
    )
