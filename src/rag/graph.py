import json
from collections.abc import AsyncIterator, Sequence
from typing import Callable, Literal
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, Send

from src.checkpointer import AsyncCompatibleSqliteSaver
from src.rag.state import (
    RAGGraphInput,
    RAGGraphOutput,
    RAGGraphState,
    RAGResult,
    RAGUserProfile,
)
from src.rag.utils import format_doc


class PolicyRagGraph:
    def __init__(
        self,
        *,
        retrieval_planner: Runnable,
        retriever: Runnable,
        policy_checker: Runnable,
        policy_selector: Runnable,
        answer_generator: Runnable,
        checkpointer: AsyncCompatibleSqliteSaver,
        max_retrieval_retries: int = 3,
        trace_config_factory: Callable[..., dict] | None = None,
    ):
        if max_retrieval_retries < 0:
            raise ValueError("max_retrieval_retries는 0 이상이어야 합니다.")
        self.retrieval_planner = retrieval_planner
        self.retriever = retriever
        self.policy_checker = policy_checker
        self.policy_selector = policy_selector
        self.answer_generator = answer_generator
        self.checkpointer = checkpointer
        self.max_retrieval_retries = max_retrieval_retries
        self.trace_config_factory = trace_config_factory
        self.graph = self._compile_graph()

    def _compile_graph(self):
        workflow = StateGraph(
            state_schema=RAGGraphState,
            input_schema=RAGGraphInput,
            output_schema=RAGGraphOutput,
        )
        workflow.add_node("retrieval_planner", self.retrieval_planner)
        workflow.add_node("retriever", self.retriever)
        workflow.add_node("policy_checker", self.policy_checker)
        workflow.add_node("policy_selector", self.policy_selector)
        workflow.add_node("answer_generator", self.answer_generator)

        workflow.add_edge(START, "retrieval_planner")
        workflow.add_conditional_edges(
            "retrieval_planner",
            self._select_after_planner,
            {
                "retriever": "retriever",
                "answer_generator": "answer_generator",
            },
        )
        workflow.add_conditional_edges(
            "retriever",
            self._dispatch_retrieved_policies,
            ["policy_checker", "policy_selector"],
        )
        workflow.add_edge("policy_checker", "policy_selector")
        workflow.add_conditional_edges(
            "policy_selector",
            self._select_after_policy_check,
            {
                "retry": "retrieval_planner",
                "answer_generator": "answer_generator",
            },
        )
        workflow.add_edge("answer_generator", END)
        return workflow.compile(
            checkpointer=self.checkpointer,
            name="youth_policy_rag",
        )

    @staticmethod
    def _select_after_planner(
        state: RAGGraphState,
    ) -> Literal["retriever", "answer_generator"]:
        return (
            "retriever"
            if state.get("needs_retrieval", False)
            else "answer_generator"
        )

    @staticmethod
    def _dispatch_retrieved_policies(
        state: RAGGraphState,
    ) -> list[Send] | Literal["policy_selector"]:
        documents = state.get("retrieved_policies", [])
        if not documents:
            return "policy_selector"
        return [
            Send(
                "policy_checker",
                {
                    "user_requirement": state["user_requirement"],
                    "current_question": state["user_input"],
                    "user_profile": state["user_profile"],
                    "policy": document,
                    "retrieval_rank": rank,
                    "retrieval_round": state.get("retrieval_count", 1),
                },
            )
            for rank, document in enumerate(documents, start=1)
        ]

    def _select_after_policy_check(
        self,
        state: RAGGraphState,
    ) -> Literal["retry", "answer_generator"]:
        if state.get("documents"):
            return "answer_generator"
        if state.get("retrieval_count", 0) <= self.max_retrieval_retries:
            return "retry"
        return "answer_generator"

    def _build_graph_config(
        self,
        thread_id: str | None,
        *,
        trace_user_id: str | None = None,
        trace_id: str | None = None,
        trace_tags: Sequence[str] | None = None,
        trace_metadata: dict | None = None,
    ) -> dict:
        resolved_thread_id = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": resolved_thread_id}}
        trace_config = (
            self.trace_config_factory(
                user_id=trace_user_id,
                session_id=resolved_thread_id,
                trace_id=trace_id,
                tags=trace_tags or ["youth-policy-rag"],
                metadata={
                    "langgraph_thread_id": resolved_thread_id,
                    **(trace_metadata or {}),
                },
            )
            if self.trace_config_factory
            else {}
        )
        for key, value in trace_config.items():
            if key == "metadata":
                config.setdefault("metadata", {}).update(value)
            elif key == "callbacks":
                config.setdefault("callbacks", []).extend(value)
            else:
                config[key] = value
        return config

    @staticmethod
    def build_graph_input(
        *,
        user_input: str,
        user_profile: RAGUserProfile,
        exclude_expired: bool,
    ) -> RAGGraphInput:
        return {
            "user_input": user_input,
            "user_profile": dict(user_profile),
            "exclude_expired": exclude_expired,
            "messages": [HumanMessage(content=user_input)],
            "retrieval_count": 0,
            "retrieved_policies": [],
            "checked_policies": Overwrite([]),
        }

    @staticmethod
    def build_result(
        *,
        answer: str,
        documents: Sequence[Document],
    ) -> RAGResult:
        return RAGResult(
            answer=answer,
            contexts=[
                format_doc(document, index)
                for index, document in enumerate(documents, start=1)
            ],
            retrieved_policy_ids=[
                str(document.metadata["plcyNo"])
                for document in documents
                if document.metadata.get("plcyNo")
            ],
        )

    def generate_answer(
        self,
        user_input: str,
        user_profile: RAGUserProfile,
        thread_id: str | None = None,
        exclude_expired: bool = True,
        *,
        trace_user_id: str | None = None,
        trace_id: str | None = None,
        trace_tags: Sequence[str] | None = None,
        trace_metadata: dict | None = None,
    ) -> RAGResult:
        output = self.graph.invoke(
            self.build_graph_input(
                user_input=user_input,
                user_profile=user_profile,
                exclude_expired=exclude_expired,
            ),
            config=self._build_graph_config(
                thread_id,
                trace_user_id=trace_user_id,
                trace_id=trace_id,
                trace_tags=trace_tags,
                trace_metadata=trace_metadata,
            ),
        )
        return self.build_result(
            answer=output["answer"],
            documents=output.get("documents", []),
        )

    async def agenerate_answer(
        self,
        user_input: str,
        user_profile: RAGUserProfile,
        thread_id: str | None = None,
        exclude_expired: bool = True,
        *,
        trace_user_id: str | None = None,
        trace_id: str | None = None,
        trace_tags: Sequence[str] | None = None,
        trace_metadata: dict | None = None,
    ) -> RAGResult:
        output = await self.graph.ainvoke(
            self.build_graph_input(
                user_input=user_input,
                user_profile=user_profile,
                exclude_expired=exclude_expired,
            ),
            config=self._build_graph_config(
                thread_id,
                trace_user_id=trace_user_id,
                trace_id=trace_id,
                trace_tags=trace_tags,
                trace_metadata=trace_metadata,
            ),
        )
        return self.build_result(
            answer=output["answer"],
            documents=output.get("documents", []),
        )

    async def stream_answer(
        self,
        user_input: str,
        user_profile: RAGUserProfile,
        thread_id: str | None = None,
        exclude_expired: bool = True,
        *,
        trace_user_id: str | None = None,
        trace_id: str | None = None,
        trace_tags: Sequence[str] | None = None,
        trace_metadata: dict | None = None,
    ) -> AsyncIterator[str]:
        graph_input = self.build_graph_input(
            user_input=user_input,
            user_profile=user_profile,
            exclude_expired=exclude_expired,
        )
        config = self._build_graph_config(
            thread_id,
            trace_user_id=trace_user_id,
            trace_id=trace_id,
            trace_tags=trace_tags,
            trace_metadata=trace_metadata,
        )
        previous_snapshot = await self.graph.aget_state(config)
        latest_documents = list(
            previous_snapshot.values.get("documents", [])
        )
        latest_retrieval_count = 0
        metadata_sent = False
        streamed_answer = ""

        async for part in self.graph.astream(
            graph_input,
            config=config,
            stream_mode=["updates", "messages"],
            version="v2",
        ):
            if part["type"] == "updates":
                update = part["data"]
                planner_update = update.get("retrieval_planner")
                if planner_update:
                    if planner_update.get("needs_retrieval"):
                        latest_documents = []
                    elif not metadata_sent:
                        latest_documents = list(
                            planner_update.get(
                                "documents",
                                latest_documents,
                            )
                        )
                        yield self._metadata_event(
                            latest_documents,
                            trace_id=trace_id,
                        )
                        metadata_sent = True

                retriever_update = update.get("retriever")
                if retriever_update:
                    latest_retrieval_count = retriever_update.get(
                        "retrieval_count",
                        latest_retrieval_count,
                    )

                selector_update = update.get("policy_selector")
                if selector_update:
                    latest_documents = list(
                        selector_update.get("documents", [])
                    )
                    final_attempt = (
                        latest_retrieval_count
                        >= self.max_retrieval_retries + 1
                    )
                    if (
                        not metadata_sent
                        and (latest_documents or final_attempt)
                    ):
                        yield self._metadata_event(
                            latest_documents,
                            trace_id=trace_id,
                        )
                        metadata_sent = True

                generator_update = update.get("answer_generator")
                if generator_update and not streamed_answer:
                    if not metadata_sent:
                        yield self._metadata_event(
                            latest_documents,
                            trace_id=trace_id,
                        )
                        metadata_sent = True
                    answer = generator_update["answer"]
                    streamed_answer = answer
                    yield self._sse_event("chunk", answer)

            if part["type"] == "messages":
                message_chunk, metadata = part["data"]
                if (
                    metadata.get("langgraph_node") != "answer_generator"
                    or metadata.get("ls_model_type") != "chat"
                ):
                    continue
                chunk = message_chunk.text
                if not chunk:
                    continue
                if not metadata_sent:
                    yield self._metadata_event(
                        latest_documents,
                        trace_id=trace_id,
                    )
                    metadata_sent = True
                streamed_answer += chunk
                yield self._sse_event("chunk", chunk)

        if not metadata_sent:
            yield self._metadata_event(
                latest_documents,
                trace_id=trace_id,
            )
        yield self._sse_event("done")

    async def get_conversation(self, thread_id: str) -> dict:
        snapshot = await self.graph.aget_state(
            self._build_graph_config(thread_id)
        )
        messages = []
        for message in snapshot.values.get("messages", []):
            message_type = getattr(message, "type", "")
            if message_type == "human":
                role = "user"
            elif message_type == "ai":
                role = "assistant"
            else:
                continue

            content = getattr(message, "content", "")
            if not isinstance(content, str):
                content = str(content)
            messages.append({"role": role, "content": content})

        documents = snapshot.values.get(
            "documents",
            snapshot.values.get("active_policies", []),
        )
        active_policy_ids = [
            str(document.metadata["plcyNo"])
            for document in documents
            if document.metadata.get("plcyNo")
        ]
        return {
            "messages": messages,
            "active_policy_ids": active_policy_ids,
        }

    def _metadata_event(
        self,
        documents: Sequence[Document],
        *,
        trace_id: str | None = None,
    ) -> str:
        result = self.build_result(answer="", documents=documents)
        data = {
            "contexts": result.contexts,
            "retrieved_policy_ids": result.retrieved_policy_ids,
        }
        if trace_id:
            data["trace_id"] = trace_id
        return self._sse_event(
            "metadata",
            data,
        )

    @staticmethod
    def _sse_event(event_type: str, data=None) -> str:
        event = {"type": event_type}
        if data is not None:
            event["data"] = data
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def delete_conversation(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        close = getattr(self.checkpointer, "close", None)
        if close:
            close()
