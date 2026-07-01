from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.rag.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.rag.state import PolicySearchProfile
from src.rag.utils.formatting import format_user_profile, format_docs


class GenerationRequest(TypedDict):
    question: str
    user_profile: PolicySearchProfile
    documents: list[Document]
    chat_history: list[AnyMessage]
    conversation_summary: str


class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            (
                "system",
                "이전 대화 요약:\n{conversation_summary}",
            ),
            MessagesPlaceholder(
                "chat_history",
                optional=True,
            ),
            ("human", USER_PROMPT),
        ])

        self.chain = self.prompt | llm | StrOutputParser()

    def _build_chain_input(
        self,
        request: GenerationRequest,
    ) -> dict:
        return {
            "question": request["question"],
            "user_profile": format_user_profile(
                request["user_profile"]
            ),
            "context": format_docs(
                request["documents"]
            ),
            "chat_history": request["chat_history"],
            "conversation_summary": (
                request["conversation_summary"] or "없음"
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
