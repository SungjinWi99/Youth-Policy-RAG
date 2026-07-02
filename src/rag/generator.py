from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from src.rag.prompts import ANSWER_SYSTEM_PROMPT
from src.rag.state import PolicySearchProfile
from src.rag.utils.formatting import (
    format_docs,
    format_user_profile,
)


class GenerationRequest(TypedDict):
    user_profile: PolicySearchProfile
    documents: list[Document]
    messages: list[AnyMessage]
    conversation_summary: str
    retrieval_error: str


class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("messages"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()

    def _build_chain_input(
        self,
        request: GenerationRequest,
    ) -> dict:
        return {
            "user_profile": format_user_profile(
                request["user_profile"]
            ),
            "context": format_docs(request["documents"]) or "없음",
            "messages": request["messages"],
            "conversation_summary": (
                request["conversation_summary"] or "없음"
            ),
            "retrieval_error": (
                request["retrieval_error"] or "없음"
            ),
        }

    def build_prompt_messages(
        self,
        request: GenerationRequest,
    ) -> list[AnyMessage]:
        return self.prompt.format_messages(
            **self._build_chain_input(request)
        )

    def generate(
        self,
        request: GenerationRequest,
    ) -> str:
        return self.chain.invoke(
            self._build_chain_input(request)
        )

    async def agenerate(
        self,
        request: GenerationRequest,
    ) -> str:
        return await self.chain.ainvoke(
            self._build_chain_input(request)
        )
