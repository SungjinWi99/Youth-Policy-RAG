from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.rag.prompts import (
    TURN_PLANNER_SYSTEM_PROMPT,
    TURN_PLANNER_USER_PROMPT,
)
from src.rag.state import RAGUserProfile
from src.rag.utils.formatting import format_docs, format_user_profile


AnswerStrategy = Literal[
    "brief_reply",
    "policy_recommendation",
    "profile_update_response",
    "focused_followup",
    "clarifying_question",
    "summary",
]


class TurnPlan(BaseModel):
    route: Literal["retriever", "agent"] = Field(
        description=(
            "이번 턴의 실행 경로. retriever는 새 정책 검색이 필요한 경우, "
            "agent는 새 검색 없이 답변 생성이 가능한 경우."
        )
    )
    answer_strategy: AnswerStrategy = Field(
        description="최종 답변 생성기가 사용할 이번 턴의 답변 전략."
    )
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description=(
            "route가 retriever일 때 사용할 검색 질의 후보 2~3개. "
            "정책 주제, 문제/목적, 지역을 포함하고 연령/소득/성별/직업은 제외."
        ),
    )
    route_reason: str = Field(
        description="실행 경로와 답변 전략을 선택한 이유. 1문장으로 간결하게 작성."
    )


class TurnPlanner:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm.with_structured_output(TurnPlan)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", TURN_PLANNER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", TURN_PLANNER_USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm

    def _build_chain_input(
        self,
        *,
        current_question: str,
        user_profile: RAGUserProfile,
        documents: Sequence[Document] | None = None,
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> dict:
        return {
            "documents": format_docs(documents) if documents else [],
            "current_question": current_question,
            "user_profile": format_user_profile(user_profile),
            "chat_history": list(chat_history or []),
        }

    def _guard_plan(
        self,
        plan: TurnPlan,
        *,
        current_question: str,
        documents: Sequence[Document],
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> TurnPlan:
        updates: dict = {}

        if plan.route == "retriever" and not plan.retrieval_queries:
            updates["retrieval_queries"] = [current_question]

        if plan.route == "agent" and plan.retrieval_queries:
            updates["retrieval_queries"] = []

        if (
            plan.route == "agent"
            and not documents
            and plan.answer_strategy in {
                "focused_followup",
                "policy_recommendation",
            }
        ):
            updates["answer_strategy"] = "clarifying_question"

        if (
            plan.answer_strategy == "summary"
            and not documents
            and not chat_history
        ):
            updates["answer_strategy"] = "brief_reply"

        if not updates:
            return plan

        return plan.model_copy(update=updates)

    def decide(
        self,
        *,
        current_question: str,
        user_profile: RAGUserProfile,
        documents: Sequence[Document],
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> TurnPlan:
        chain_input = self._build_chain_input(
            current_question=current_question,
            user_profile=user_profile,
            documents=documents,
            chat_history=chat_history,
        )
        plan = self.chain.invoke(chain_input)
        return self._guard_plan(
            plan,
            current_question=current_question,
            documents=documents,
            chat_history=chat_history,
        )

    async def adecide(
        self,
        *,
        current_question: str,
        user_profile: RAGUserProfile,
        documents: Sequence[Document],
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> TurnPlan:
        chain_input = self._build_chain_input(
            current_question=current_question,
            user_profile=user_profile,
            documents=documents,
            chat_history=chat_history,
        )
        plan = await self.chain.ainvoke(chain_input)
        return self._guard_plan(
            plan,
            current_question=current_question,
            documents=documents,
            chat_history=chat_history,
        )
