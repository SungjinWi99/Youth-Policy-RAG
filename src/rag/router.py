from collections.abc import Sequence
from typing import Any, Literal, Protocol

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.rag.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT


RouteName = Literal["reuse", "search", "clarify"]


class RoutingDecision(BaseModel):
    route: RouteName = Field(
        description="reuse, search, clarify 중 선택한 경로",
    )
    reason: str = Field(
        min_length=1,
        description="현재 질문과 활성 문서를 근거로 한 한 문장 판단 이유",
    )


class ContextRouter(Protocol):
    def decide(
        self,
        *,
        current_question: str,
        documents: list[Document],
    ) -> RoutingDecision:
        """현재 질문과 활성 문서의 관계를 분류한다."""
        ...


def format_routing_documents(
    documents: Sequence[Document],
    *,
    max_content_chars: int = 1500,
) -> str:
    if not documents:
        return "(활성 정책 문서 없음)"

    formatted = []
    for index, document in enumerate(documents, start=1):
        policy_id = str(
            document.metadata.get("plcyNo") or "미제공"
        )
        content = document.page_content.strip()
        if len(content) > max_content_chars:
            content = f"{content[:max_content_chars].rstrip()}…"
        formatted.append(
            f"[활성 정책 {index}]\n"
            f"정책번호: {policy_id}\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(formatted)


class LLMContextRouter:
    def __init__(self, llm: Any):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_USER_PROMPT),
        ])
        self.chain = (
            self.prompt
            | llm.with_structured_output(RoutingDecision)
        )

    def decide(
        self,
        *,
        current_question: str,
        documents: list[Document],
    ) -> RoutingDecision:
        result = self.chain.invoke({
            "current_question": current_question,
            "documents": format_routing_documents(documents),
        })
        return RoutingDecision.model_validate(result)
