from typing import TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.rag.prompts import SUMMARY_SYSTEM_PROMPT


class ConversationSummaryResult(TypedDict):
    conversation_summary: str
    remove_message_ids: list[str]


class ConversationSummarizer:
    def __init__(
        self,
        llm,
        *,
        max_input_tokens: int,
        summary_trigger_ratio: float,
        keep_recent_turns: int,
        chars_per_token: float,
    ):
        self.trigger_tokens = max(
            1,
            int(max_input_tokens * summary_trigger_ratio),
        )
        self.keep_recent_turns = keep_recent_turns
        self.chars_per_token = chars_per_token

        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    f"{SUMMARY_SYSTEM_PROMPT}\n\n"
                    "기존 요약:\n{conversation_summary}"
                ),
            ),
            MessagesPlaceholder("messages_to_summarize"),
            (
                "human",
                "위 내용을 하나의 최신 대화 요약으로 압축하세요.",
            ),
        ])

        self.summary_chain = (
            summary_prompt
            | llm
            | StrOutputParser()
        )

    def count_prompt_tokens(
        self,
        prompt_messages: list[AnyMessage],
    ) -> int:
        return count_tokens_approximately(
            prompt_messages,
            chars_per_token=self.chars_per_token,
        )

    def select_messages_to_summarize(
        self,
        messages: list[AnyMessage],
    ) -> list[AnyMessage]:
        human_message_indices = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ]

        # 최근 완료 N턴과 현재 사용자 메시지 보존
        human_messages_to_keep = self.keep_recent_turns + 1

        if len(human_message_indices) <= human_messages_to_keep:
            return []

        keep_from_index = human_message_indices[
            -human_messages_to_keep
        ]

        return messages[:keep_from_index]

    def should_summarize(
        self,
        prompt_messages: list[AnyMessage],
        conversation_messages: list[AnyMessage],
    ) -> bool:
        prompt_tokens = self.count_prompt_tokens(
            prompt_messages
        )

        messages_to_summarize = (
            self.select_messages_to_summarize(
                conversation_messages
            )
        )

        return (
            prompt_tokens >= self.trigger_tokens
            and bool(messages_to_summarize)
        )

    def _build_summary_input(
        self,
        existing_summary: str,
        messages_to_summarize: list[AnyMessage],
    ) -> dict:
        return {
            "conversation_summary": existing_summary or "없음",
            "messages_to_summarize": messages_to_summarize,
        }

    def _build_summary_output(
        self,
        summary: str,
        messages_to_summarize: list[AnyMessage],
    ) -> ConversationSummaryResult:
        return {
            "conversation_summary": summary,
            "remove_message_ids": [
                message.id
                for message in messages_to_summarize
                if message.id
            ],
        }

    def summarize(
        self,
        *,
        existing_summary: str,
        messages: list[AnyMessage],
    ) -> ConversationSummaryResult:
        messages_to_summarize = self.select_messages_to_summarize(
            messages
        )
        if not messages_to_summarize:
            return self._build_summary_output(existing_summary, [])

        summary = self.summary_chain.invoke(
            self._build_summary_input(
                existing_summary,
                messages_to_summarize,
            )
        )

        return self._build_summary_output(
            summary,
            messages_to_summarize,
        )

    async def asummarize(
        self,
        *,
        existing_summary: str,
        messages: list[AnyMessage],
    ) -> ConversationSummaryResult:
        messages_to_summarize = self.select_messages_to_summarize(
            messages
        )
        if not messages_to_summarize:
            return self._build_summary_output(existing_summary, [])

        summary = await self.summary_chain.ainvoke(
            self._build_summary_input(
                existing_summary,
                messages_to_summarize,
            )
        )

        return self._build_summary_output(
            summary,
            messages_to_summarize,
        )
