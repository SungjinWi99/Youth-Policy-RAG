from typing import TypedDict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from src.rag.prompts import (
    SYSTEM_PROMPT,
    TOOL_CALLING_SYSTEM_PROMPT,
)
from src.rag.state import PolicySearchProfile, RetrievalMode
from src.rag.utils.formatting import format_docs, format_user_profile


class AgentRequest(TypedDict):
    user_profile: PolicySearchProfile
    documents: list[Document]
    messages: list[AnyMessage]
    conversation_summary: str
    last_retrieval_query: str


class PolicyAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.required_tool_name = "search_policies"
        if not any(
            getattr(tool, "name", None) == self.required_tool_name
            for tool in tools
        ):
            raise ValueError(
                f"{self.required_tool_name} tool이 필요합니다."
            )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"{TOOL_CALLING_SYSTEM_PROMPT}\n\n"
                    "이전 대화 요약:\n{conversation_summary}\n\n"
                    "마지막 정책 검색어:\n{last_retrieval_query}"
                ),
            ),
            MessagesPlaceholder("messages"),
        ])

        try:
            forced_search_model = llm.bind_tools(
                tools,
                tool_choice=self.required_tool_name,
            )
            auto_search_model = llm.bind_tools(
                tools,
                tool_choice="auto",
            )
        except (AttributeError, NotImplementedError):
            # 단순 Runnable 기반 단위 테스트를 위한 fallback이다.
            forced_search_model = None
            auto_search_model = llm

        self.forced_search_chain = (
            self.prompt | forced_search_model
            if forced_search_model is not None
            else None
        )
        self.agent_chain = self.prompt | auto_search_model
        self.final_chain = self.prompt | llm

    def _build_chain_input(
        self,
        request: AgentRequest,
    ) -> dict:
        return {
            "user_profile": format_user_profile(
                request["user_profile"]
            ),
            "context": format_docs(request["documents"]),
            "messages": request["messages"],
            "conversation_summary": (
                request["conversation_summary"] or "없음"
            ),
            "last_retrieval_query": (
                request["last_retrieval_query"] or "없음"
            ),
        }

    @staticmethod
    def _as_ai_message(response) -> AIMessage:
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=str(response))

    def build_prompt_messages(
        self,
        request: AgentRequest,
    ) -> list[AnyMessage]:
        return self.prompt.format_messages(
            **self._build_chain_input(request)
        )

    def _fallback_forced_tool_call(
        self,
        request: AgentRequest,
    ) -> AIMessage:
        query = next(
            (
                message.text
                for message in reversed(request["messages"])
                if isinstance(message, HumanMessage)
                and message.text
            ),
            request["last_retrieval_query"] or "청년정책",
        )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.required_tool_name,
                    "args": {"query": query},
                    "id": f"forced-search-{uuid4()}",
                    "type": "tool_call",
                }
            ],
        )

    def _select_chain(
        self,
        retrieval_mode: RetrievalMode,
    ):
        if retrieval_mode == "required":
            return self.forced_search_chain
        if retrieval_mode == "optional":
            return self.agent_chain
        if retrieval_mode == "disabled":
            return self.final_chain
        raise ValueError(
            f"지원하지 않는 retrieval mode입니다: {retrieval_mode}"
        )

    def _validate_response(
        self,
        response: AIMessage,
        retrieval_mode: RetrievalMode,
    ) -> AIMessage:
        if len(response.tool_calls) > 1:
            raise RuntimeError(
                "한 요청에서 하나의 tool만 호출할 수 있습니다."
            )
        if (
            retrieval_mode == "required"
            and (
                not response.tool_calls
                or response.tool_calls[0]["name"]
                != self.required_tool_name
            )
        ):
            raise RuntimeError(
                "필수 정책 검색 tool이 호출되지 않았습니다."
            )
        return response

    def invoke(
        self,
        request: AgentRequest,
        *,
        retrieval_mode: RetrievalMode,
    ) -> AIMessage:
        chain = self._select_chain(retrieval_mode)
        if chain is None:
            return self._fallback_forced_tool_call(request)

        response = self._as_ai_message(
            chain.invoke(self._build_chain_input(request))
        )
        return self._validate_response(response, retrieval_mode)

    async def ainvoke(
        self,
        request: AgentRequest,
        *,
        retrieval_mode: RetrievalMode,
    ) -> AIMessage:
        chain = self._select_chain(retrieval_mode)
        if chain is None:
            return self._fallback_forced_tool_call(request)

        response = await chain.ainvoke(
            self._build_chain_input(request)
        )
        return self._validate_response(
            self._as_ai_message(response),
            retrieval_mode,
        )
