import json
from collections.abc import AsyncIterator
from typing import Literal
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from src.rag.generator import AnswerGenerator, GenerationRequest
from src.rag.retriever import PolicyRetriever
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
        retriever: PolicyRetriever,
        summarizer: ConversationSummarizer,
        generator: AnswerGenerator,
        checkpointer,
    ):
        self.retriever = retriever
        self.summarizer = summarizer
        self.generator = generator
        self.checkpointer = checkpointer
        self.graph = self._compile_graph()

    def _compile_graph(self):
        workflow = StateGraph(
            state_schema=RAGGraphState,
            input_schema=RAGGraphInput,
            output_schema=RAGGraphOutput,
        )
        workflow.add_node(
            "retrieve",
            RunnableLambda(
                self._retrieve_node,
                afunc=self._aretrieve_node,
            ),
        )
        workflow.add_node(
            "summarize",
            RunnableLambda(
                self._summarize_node,
                afunc=self._asummarize_node,
            ),
        )
        workflow.add_node(
            "generate",
            RunnableLambda(
                self._generate_node,
                afunc=self._agenerate_node,
            ),
        )
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._route_by_context_size,
            {
                "summarize": "summarize",
                "generate": "generate",
            },
        )
        workflow.add_edge("summarize", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile(
            checkpointer=self.checkpointer,
            name="youth_policy_rag",
        )

    def _build_retrieval_query(
        self,
        messages: list[AnyMessage],
    ) -> str:
        recent_questions = [
            message.text
            for message in messages
            if isinstance(message, HumanMessage) and message.text
        ]
        return "\n".join(recent_questions[-2:])

    def _build_generation_request(
        self,
        state: RAGGraphState,
    ) -> GenerationRequest:
        return {
            "question": state["user_input"],
            "user_profile": state["user_profile"],
            "documents": state.get("documents", []),
            "chat_history": state["messages"][:-1],
            "conversation_summary": (
                state.get("conversation_summary") or ""
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

    def _graph_config(
        self,
        user_id: str | None,
        user_profile: UserProfile,
    ) -> dict:
        profile_user_id = getattr(user_profile, "user_id", None)
        thread_id = user_id or profile_user_id or str(uuid4())
        return {"configurable": {"thread_id": thread_id}}

    def _route_by_context_size(
        self,
        state: RAGGraphState,
    ) -> Literal["summarize", "generate"]:
        request = self._build_generation_request(state)
        prompt_messages = self.generator.build_prompt_messages(
            request
        )
        if self.summarizer.should_summarize(
            prompt_messages=prompt_messages,
            conversation_messages=state["messages"],
        ):
            return "summarize"
        return "generate"

    def _retrieve_node(self, state: RAGGraphState) -> dict:
        documents = self.retriever.retrieve(
            query=self._build_retrieval_query(state["messages"]),
            user_profile=state["user_profile"],
            exclude_expired=state["exclude_expired"],
        )
        return {"documents": documents}

    async def _aretrieve_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        documents = await self.retriever.aretrieve(
            query=self._build_retrieval_query(state["messages"]),
            user_profile=state["user_profile"],
            exclude_expired=state["exclude_expired"],
        )
        return {"documents": documents}

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

    def _generate_node(self, state: RAGGraphState) -> dict:
        answer = self.generator.generate(
            self._build_generation_request(state)
        )
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    async def _agenerate_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        answer = await self.generator.agenerate(
            self._build_generation_request(state)
        )
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    def generate_answer(
        self,
        user_input: str,
        user_profile: UserProfile,
        exclude_expired: bool = True,
        user_id: str | None = None,
    ) -> RAGResult:
        graph_output = self.graph.invoke(
            self._build_graph_input(
                user_input=user_input,
                user_profile=user_profile,
                exclude_expired=exclude_expired,
            ),
            config=self._graph_config(user_id, user_profile),
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
        user_id: str | None = None,
    ) -> RAGResult:
        graph_output = await self.graph.ainvoke(
            self._build_graph_input(
                user_input=user_input,
                user_profile=user_profile,
                exclude_expired=exclude_expired,
            ),
            config=self._graph_config(user_id, user_profile),
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
        user_id: str | None = None,
    ) -> AsyncIterator[str]:
        graph_input = self._build_graph_input(
            user_input=user_input,
            user_profile=user_profile,
            exclude_expired=exclude_expired,
        )
        streamed_answer = ""

        async for part in self.graph.astream(
            graph_input,
            config=self._graph_config(user_id, user_profile),
            stream_mode=["updates", "messages"],
            version="v2",
        ):
            if part["type"] == "updates":
                retrieve_update = part["data"].get("retrieve")
                if retrieve_update:
                    result_metadata = self._build_result(
                        "",
                        retrieve_update["documents"],
                    )
                    yield self._sse_event(
                        "metadata",
                        {
                            "contexts": result_metadata.contexts,
                            "retrieved_policy_ids": (
                                result_metadata.retrieved_policy_ids
                            ),
                        },
                    )

                generate_update = part["data"].get("generate")
                if generate_update and not streamed_answer:
                    answer = generate_update["answer"]
                    streamed_answer = answer
                    yield self._sse_event("chunk", answer)

            if part["type"] == "messages":
                message_chunk, metadata = part["data"]
                if (
                    metadata.get("langgraph_node") != "generate"
                    or metadata.get("ls_model_type") != "chat"
                ):
                    continue

                chunk = message_chunk.text
                if not chunk:
                    continue

                streamed_answer += chunk
                yield self._sse_event("chunk", chunk)

        yield self._sse_event("done")

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
