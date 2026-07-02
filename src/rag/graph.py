import json
from collections.abc import AsyncIterator
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from src.rag.agent import AgentRequest, PolicyAgent
from src.rag.state import (
    RAGGraphInput,
    RAGGraphOutput,
    RAGGraphState,
    RAGResult,
)
from src.rag.summarizer import ConversationSummarizer
from src.rag.utils.formatting import format_doc
from src.user.models import UserProfile


class RAGGraph:
    def __init__(
        self,
        *,
        summarizer: ConversationSummarizer,
        agent: PolicyAgent,
        tool_node,
        checkpointer,
    ):
        self.summarizer = summarizer
        self.agent = agent
        self.tool_node = tool_node
        self.checkpointer = checkpointer
        self.graph = self._compile_graph()

    def _compile_graph(self):
        workflow = StateGraph(
            state_schema=RAGGraphState,
            input_schema=RAGGraphInput,
            output_schema=RAGGraphOutput,
        )
        workflow.add_node("prepare", self._prepare_node)
        workflow.add_node(
            "summarize",
            RunnableLambda(
                self._summarize_node,
                afunc=self._asummarize_node,
            ),
        )
        workflow.add_node(
            "agent",
            RunnableLambda(
                self._agent_node,
                afunc=self._aagent_node,
            ),
        )
        workflow.add_node("tools", self.tool_node)

        workflow.add_edge(START, "prepare")
        workflow.add_conditional_edges(
            "prepare",
            self._route_by_context_size,
            {
                "summarize": "summarize",
                "agent": "agent",
            },
        )
        workflow.add_edge("summarize", "agent")
        workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "tools": "tools",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "tools",
            self._route_by_context_size,
            {
                "summarize": "summarize",
                "agent": "agent",
            },
        )
        return workflow.compile(
            checkpointer=self.checkpointer,
            name="youth_policy_rag",
        )

    def _build_agent_request(
        self,
        state: RAGGraphState,
    ) -> AgentRequest:
        return {
            "user_profile": state["user_profile"],
            "documents": state.get("documents", []),
            "messages": state["messages"],
            "conversation_summary": (
                state.get("conversation_summary") or ""
            ),
            "last_retrieval_query": (
                state.get("last_retrieval_query") or ""
            ),
        }

    def _build_result(
        self,
        answer: str,
        documents: list[Document],
    ) -> RAGResult:
        return RAGResult(
            answer=answer,
            contexts=[
                format_doc(document, index)
                for index, document in enumerate(
                    documents,
                    start=1,
                )
            ],
            retrieved_policy_ids=[
                document.metadata["plcyNo"]
                for document in documents
                if document.metadata.get("plcyNo")
            ],
        )

    def _build_graph_input(
        self,
        *,
        user_input: str,
        user_profile: UserProfile,
        exclude_expired: bool,
    ) -> RAGGraphInput:
        return {
            "user_input": user_input,
            "user_profile": user_profile.model_dump(
                include={
                    "age",
                    "gender",
                    "job",
                    "income",
                    "region",
                }
            ),
            "exclude_expired": exclude_expired,
            "messages": [HumanMessage(content=user_input)],
        }

    def _build_graph_config(
        self,
        user_profile: UserProfile,
    ) -> dict:
        if not user_profile.user_id:
            raise ValueError(
                "대화 상태를 저장하려면 user_profile.user_id가 필요합니다."
            )
        return {
            "configurable": {
                "thread_id": user_profile.user_id,
            }
        }

    def _retrieval_context_changed(
        self,
        state: RAGGraphState,
    ) -> bool:
        return (
            state.get("last_retrieval_profile")
            != state["user_profile"]
            or state.get("last_retrieval_exclude_expired")
            != state["exclude_expired"]
        )

    def _prepare_node(self, state: RAGGraphState) -> dict:
        documents = state.get("documents", [])
        retrieval_required = (
            not documents
            or self._retrieval_context_changed(state)
        )
        return {
            "answer": "",
            "documents": (
                []
                if retrieval_required
                else documents
            ),
            "retrieval_mode": (
                "required"
                if retrieval_required
                else "optional"
            ),
        }

    def _route_by_context_size(
        self,
        state: RAGGraphState,
    ) -> Literal["summarize", "agent"]:
        prompt_messages = self.agent.build_prompt_messages(
            self._build_agent_request(state)
        )
        if self.summarizer.should_summarize(
            prompt_messages=prompt_messages,
            conversation_messages=state["messages"],
        ):
            return "summarize"
        return "agent"

    @staticmethod
    def _route_after_agent(
        state: RAGGraphState,
    ) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if (
            isinstance(last_message, AIMessage)
            and last_message.tool_calls
        ):
            return "tools"
        return "end"

    def _summarize_node(self, state: RAGGraphState) -> dict:
        result = self.summarizer.summarize(
            existing_summary=state.get(
                "conversation_summary",
                "",
            ),
            messages=state["messages"],
        )
        return {
            "conversation_summary": result["conversation_summary"],
            "messages": [
                RemoveMessage(id=message_id)
                for message_id in result["remove_message_ids"]
            ],
        }

    async def _asummarize_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        result = await self.summarizer.asummarize(
            existing_summary=state.get(
                "conversation_summary",
                "",
            ),
            messages=state["messages"],
        )
        return {
            "conversation_summary": result["conversation_summary"],
            "messages": [
                RemoveMessage(id=message_id)
                for message_id in result["remove_message_ids"]
            ],
        }

    @staticmethod
    def _tool_message_ids_to_remove(
        state: RAGGraphState,
    ) -> list[str]:
        messages = state["messages"]
        last_human_index = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
            ),
            default=-1,
        )
        return [
            message.id
            for message in messages[last_human_index + 1:]
            if (
                message.id
                and (
                    isinstance(message, ToolMessage)
                    or (
                        isinstance(message, AIMessage)
                        and message.tool_calls
                    )
                )
            )
        ]

    def _agent_update(
        self,
        state: RAGGraphState,
        response: AIMessage,
    ) -> dict:
        if response.tool_calls:
            return {
                "messages": [response],
                "retrieval_mode": "disabled",
            }

        answer = response.text
        return {
            "answer": answer,
            "retrieval_mode": "disabled",
            "messages": [
                *[
                    RemoveMessage(id=message_id)
                    for message_id in self._tool_message_ids_to_remove(state)
                ],
                response,
            ],
        }

    def _agent_node(self, state: RAGGraphState) -> dict:
        response = self.agent.invoke(
            self._build_agent_request(state),
            retrieval_mode=state["retrieval_mode"],
        )
        return self._agent_update(state, response)

    async def _aagent_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        response = await self.agent.ainvoke(
            self._build_agent_request(state),
            retrieval_mode=state["retrieval_mode"],
        )
        return self._agent_update(state, response)

    def generate_answer(
        self,
        user_input: str,
        user_profile: UserProfile,
        exclude_expired: bool = True,
    ) -> RAGResult:
        graph_output = self.graph.invoke(
            self._build_graph_input(
                user_input=user_input,
                user_profile=user_profile,
                exclude_expired=exclude_expired,
            ),
            config=self._build_graph_config(user_profile),
        )
        return self._build_result(
            graph_output["answer"],
            graph_output["documents"],
        )

    async def agenerate_answer(
        self,
        user_input: str,
        user_profile: UserProfile,
        exclude_expired: bool = True,
    ) -> RAGResult:
        graph_output = await self.graph.ainvoke(
            self._build_graph_input(
                user_input=user_input,
                user_profile=user_profile,
                exclude_expired=exclude_expired,
            ),
            config=self._build_graph_config(user_profile),
        )
        return self._build_result(
            graph_output["answer"],
            graph_output["documents"],
        )

    async def stream_answer(
        self,
        user_input: str,
        user_profile: UserProfile,
        exclude_expired: bool = True,
    ) -> AsyncIterator[str]:
        graph_input = self._build_graph_input(
            user_input=user_input,
            user_profile=user_profile,
            exclude_expired=exclude_expired,
        )
        streamed_answer = ""
        suppressed_message_runs = set()

        async for part in self.graph.astream(
            graph_input,
            config=self._build_graph_config(user_profile),
            stream_mode=["updates", "messages"],
            version="v2",
        ):
            if part["type"] == "updates":
                prepare_update = part["data"].get("prepare")
                if (
                    prepare_update
                    and prepare_update["retrieval_mode"]
                    == "optional"
                ):
                    yield self._metadata_event(
                        prepare_update["documents"]
                    )

                tools_update = part["data"].get("tools")
                if tools_update and tools_update.get("documents"):
                    streamed_answer = ""
                    yield self._metadata_event(
                        tools_update["documents"]
                    )

                agent_update = part["data"].get("agent")
                if (
                    agent_update
                    and agent_update.get("answer")
                    and not streamed_answer
                ):
                    answer = agent_update["answer"]
                    streamed_answer = answer
                    yield self._sse_event("chunk", answer)

            if part["type"] == "messages":
                message_chunk, metadata = part["data"]
                if (
                    metadata.get("langgraph_node") != "agent"
                    or metadata.get("ls_model_type") != "chat"
                ):
                    continue

                run_id = (
                    metadata.get("checkpoint_ns")
                    or message_chunk.id
                )
                if getattr(
                    message_chunk,
                    "tool_call_chunks",
                    None,
                ):
                    suppressed_message_runs.add(run_id)
                    continue
                if run_id in suppressed_message_runs:
                    continue

                chunk = message_chunk.text
                if not chunk:
                    continue

                streamed_answer += chunk
                yield self._sse_event("chunk", chunk)

        yield self._sse_event("done")

    def _metadata_event(
        self,
        documents: list[Document],
    ) -> str:
        result_metadata = self._build_result("", documents)
        return self._sse_event(
            "metadata",
            {
                "contexts": result_metadata.contexts,
                "retrieved_policy_ids": (
                    result_metadata.retrieved_policy_ids
                ),
            },
        )

    @staticmethod
    def _sse_event(
        event_type: str,
        data=None,
    ) -> str:
        event = {"type": event_type}
        if data is not None:
            event["data"] = data
        return (
            f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        )

    def delete_conversation(self, user_id: str) -> None:
        self.checkpointer.delete_thread(user_id)

    def close(self) -> None:
        close = getattr(self.checkpointer, "close", None)
        if close:
            close()
