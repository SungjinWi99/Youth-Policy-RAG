from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from src.rag.state import CheckedPolicy, RAGGraphState
from src.rag.utils import format_docs, format_user_profile, window_history


PLANNER_SYSTEM_PROMPT = """
당신은 청년정책 상담 workflow의 Retrieval Planner이다.
사용자의 현재 질문, 최근 대화, 사용자 프로필을 분석해 사용자의 요구를 한 문장으로 정리한다.
사용자의 요구와 활성 정책 문서의 관계를 분석해 이번 턴의 검색 필요 여부를 판단한다.
검색이 필요하면 사용자에게 적합한 정책을 찾기 위한 검색 Query를 작성한다.

Retrieval 결정 규칙:
1. 현재 활성 정책이 사용자의 정책 목적과 상황에 적합하지 않은 경우에만 검색한다.
2. 현재 질문을 최우선으로 고려한다.
3. 활성 정책에 대한 신청 방법, 제출 서류, 일정, 지원 금액 등 후속 세부 질문에는
   해당 정보가 문서에 없더라도 검색하지 않는다.
4. 한 정책이 문서 하나 전체에 대응하므로, 상세정보 부족은 다른 문서를 검색할 이유가 아니다.
   부족한 정보는 Answer Generator가 미제공으로 설명하고 공식 확인 경로를 안내한다.
5. 현재 질문이 기존 정책과 다른 정책 목적을 요구하거나, 기존 정책의 대상·지역·상황이
   사용자와 맞지 않을 때만 새 정책을 검색한다.
6. 단순 인사나 이미 답변한 내용의 짧은 확인에는 검색하지 않는다.
7. 이전 검색의 정책들이 정책 적합성 Checker 기준을 통과하지 못한 재검색 상황이면 검색한다.
   탈락 사유상 검색 방향을 바꿀 필요가 있을 때만 Query를 변경하며,
   실패 정책이 검색에서 제외되므로 같은 Query를 다시 사용할 수 있다.
8. Checker reasoning의 신청 방법·서류·일정 등 상세정보 부족은 재검색 근거로 사용하지 않는다.
9. 직전 검색이 빈 결과였고 검색 방향도 바뀌지 않았다면 같은 Query를 반복하지 않고 종료한다.

Query 작성 규칙:
1. Query에는 정책명 또는 주제와 사용자의 요청 목적을 포함한다.
2. 지역은 필요한 경우에만 포함한다. 연령·소득·직업은 Query에 넣지 않는다.
3. 재검색이면 탈락 사유를 반영하되 정책 자격을 임의로 단정하지 않는다.
   단순히 이전 Query와 같다는 이유만으로 표현을 기계적으로 바꾸지 않는다.
""".strip()

PLANNER_HUMAN_PROMPT = """
현재 질문:
{current_question}

사용자 프로필:
{user_profile}

활성 정책 문서:
{active_documents}

현재 턴 검색 시도 횟수:
{retrieval_count}

직전 검색 Query:
{previous_query}

직전 Checker 결과:
{checker_feedback}
""".strip()


prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", PLANNER_HUMAN_PROMPT),
])


class PlannerOutput(BaseModel):
    user_requirement: str = Field(
        description="사용자의 상황과 요구를 한 문장으로 정리."
    )
    needs_retrieval: bool = Field(
        description="검색의 필요 여부."
    )
    retrieval_reason: str = Field(
        description="검색 필요 여부를 결정한 이유를 한 문장으로 설명."
    )
    retrieval_query: str = Field(
        description="검색에 사용할 단일 Query. 검색이 불필요하면 빈 문자열."
    )


def _format_checker_feedback(checked_policies: Sequence[CheckedPolicy]) -> str:
    if not checked_policies:
        return "없음"
    lines = []
    for item in sorted(
        checked_policies,
        key=lambda checked: (
            checked.get("retrieval_round", 1),
            checked.get("retrieval_rank", 1),
        ),
    ):
        policy_id = item["document"].metadata.get("plcyNo", "정책번호 없음")
        lines.append(
            f"- round={item.get('retrieval_round', 1)}, "
            f"{policy_id}: verdict={item['verdict']}, "
            f"reason={item['reasoning']}"
        )
    return "\n".join(lines)


def make_retrieval_planner_node(
    llm: BaseChatModel,
    history_window_size: int,
):
    chain = prompt | llm.with_structured_output(PlannerOutput)

    def build_chain_input(state: RAGGraphState) -> dict:
        return {
            "current_question": state["user_input"],
            "user_profile": format_user_profile(state["user_profile"]),
            "active_documents": (
                format_docs(state.get("active_policies", [])) or "없음"
            ),
            "retrieval_count": state.get("retrieval_count", 0),
            "previous_query": state.get("retrieval_query", "") or "없음",
            "checker_feedback": _format_checker_feedback(
                state.get("checked_policies", [])
            ),
            "chat_history": window_history(
                state.get("messages", []),
                history_window_size,
            ),
        }

    def normalize_result(state: RAGGraphState, result) -> dict:
        plan = PlannerOutput.model_validate(result)
        retrieval_count = state.get("retrieval_count", 0)
        previous_query = state.get("retrieval_query", "").strip()
        needs_retrieval = plan.needs_retrieval
        retrieval_query = plan.retrieval_query.strip()
        retrieval_reason = plan.retrieval_reason.strip()
        retrieved_policies = state.get("retrieved_policies", [])

        if (
            retrieval_count > 0
            and not state.get("documents")
            and retrieved_policies
        ):
            needs_retrieval = True
            if not retrieval_query:
                retrieval_query = (
                    previous_query or state["user_input"].strip()
                )
            if not retrieval_reason:
                retrieval_reason = "Checker 탈락 결과를 반영해 재검색합니다."
        elif retrieval_count > 0 and not state.get("documents"):
            proposed_query = retrieval_query or previous_query
            if needs_retrieval and proposed_query == previous_query:
                needs_retrieval = False
                retrieval_query = ""
                retrieval_reason = (
                    "동일 Query의 추가 검색 결과가 없어 재검색을 종료합니다."
                )

        update = {
            "user_requirement": plan.user_requirement.strip(),
            "needs_retrieval": needs_retrieval,
            "retrieval_reason": retrieval_reason,
            "retrieval_query": retrieval_query,
        }
        if needs_retrieval:
            update["documents"] = []
        else:
            update["documents"] = list(
                state.get("active_policies", [])
            )
        return update

    def retrieval_planner_node(state: RAGGraphState):
        return normalize_result(state, chain.invoke(build_chain_input(state)))

    async def aretrieval_planner_node(state: RAGGraphState):
        result = await chain.ainvoke(build_chain_input(state))
        return normalize_result(state, result)

    return RunnableLambda(
        retrieval_planner_node,
        afunc=aretrieval_planner_node,
    )
