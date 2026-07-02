from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from pydantic import BaseModel, Field

from src.rag.prompts import RETRIEVAL_DECISION_SYSTEM_PROMPT


class RetrievalDecision(BaseModel):
    """현재 질문을 기존 문서로 답할지 새로 검색할지 결정한다."""

    needs_retrieval: bool = Field(
        description=(
            "현재 질문의 주제와 필요한 사실을 기존 정책 문서가 "
            "모두 충족할 때만 false. 주제가 다르거나 근거가 "
            "부족하거나 확실하지 않으면 true"
        ),
    )
    query: str | None = Field(
        default=None,
        description=(
            "needs_retrieval이 true일 때 현재 질문만을 기준으로 "
            "작성한 독립적인 자연어 검색 문장"
        ),
    )


class RetrievalPlanRequest(TypedDict):
    current_question: str
    documents: list[Document]
    messages: list[AnyMessage]
    conversation_summary: str


def format_documents_for_planning(
    documents: list[Document],
) -> str:
    if not documents:
        return "없음"
    return "\n\n---\n\n".join(
        document.page_content
        for document in documents
    )


class RetrievalPlanner:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RETRIEVAL_DECISION_SYSTEM_PROMPT),
            MessagesPlaceholder("messages"),
            (
                "human",
                (
                    "현재 반드시 판단해야 하는 질문:\n"
                    "{current_question}\n\n"
                    "이 질문을 기준으로 검색 필요 여부를 판단하세요. "
                    "검색한다면 이전 주제가 아니라 현재 질문의 "
                    "의도를 query에 담으세요."
                ),
            ),
        ])
        self.chain = (
            self.prompt
            | llm.with_structured_output(RetrievalDecision)
        )

    def _build_chain_input(
        self,
        request: RetrievalPlanRequest,
        *,
        force_retrieval: bool,
    ) -> dict:
        return {
            "force_retrieval": str(force_retrieval).lower(),
            "current_question": request["current_question"],
            "conversation_summary": (
                request["conversation_summary"] or "없음"
            ),
            "documents": format_documents_for_planning(
                request["documents"]
            ),
            "messages": request["messages"],
        }

    def _normalize_decision(
        self,
        raw_decision,
        request: RetrievalPlanRequest,
        *,
        force_retrieval: bool,
    ) -> RetrievalDecision:
        decision = RetrievalDecision.model_validate(raw_decision)
        needs_retrieval = (
            force_retrieval
            or decision.needs_retrieval
        )
        if not needs_retrieval:
            return RetrievalDecision(
                needs_retrieval=False,
                query=None,
            )

        query = (decision.query or "").strip()
        if not query:
            query = request["current_question"].strip()
        if not query:
            raise ValueError(
                "검색이 필요하지만 검색 쿼리를 만들 수 없습니다."
            )

        return RetrievalDecision(
            needs_retrieval=True,
            query=query,
        )

    def build_prompt_messages(
        self,
        request: RetrievalPlanRequest,
        *,
        force_retrieval: bool,
    ) -> list[AnyMessage]:
        return self.prompt.format_messages(
            **self._build_chain_input(
                request,
                force_retrieval=force_retrieval,
            )
        )

    def decide(
        self,
        request: RetrievalPlanRequest,
        *,
        force_retrieval: bool,
    ) -> RetrievalDecision:
        raw_decision = self.chain.invoke(
            self._build_chain_input(
                request,
                force_retrieval=force_retrieval,
            )
        )
        return self._normalize_decision(
            raw_decision,
            request,
            force_retrieval=force_retrieval,
        )

    async def adecide(
        self,
        request: RetrievalPlanRequest,
        *,
        force_retrieval: bool,
    ) -> RetrievalDecision:
        raw_decision = await self.chain.ainvoke(
            self._build_chain_input(
                request,
                force_retrieval=force_retrieval,
            )
        )
        return self._normalize_decision(
            raw_decision,
            request,
            force_retrieval=force_retrieval,
        )
