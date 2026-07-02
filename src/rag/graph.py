import json
from collections.abc import AsyncIterator
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from src.rag.generator import (
    AnswerGenerator,
    GenerationRequest,
)
from src.rag.planner import (
    RetrievalPlanner,
    RetrievalPlanRequest,
)
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
        summarizer: ConversationSummarizer,
        planner: RetrievalPlanner,
        retriever: PolicyRetriever,
        generator: AnswerGenerator,
        checkpointer,
    ):
        self.summarizer = summarizer
        self.planner = planner
        self.retriever = retriever
        self.generator = generator
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
            "summarize_for_planning",
            RunnableLambda(
                self._summarize_node,
                afunc=self._asummarize_node,
            ),
        )
        workflow.add_node(
            "plan_retrieval",
            RunnableLambda(
                self._plan_retrieval_node,
                afunc=self._aplan_retrieval_node,
            ),
        )
        workflow.add_node(
            "retrieve",
            RunnableLambda(
                self._retrieve_node,
                afunc=self._aretrieve_node,
            ),
        )
        workflow.add_node(
            "summarize_for_answer",
            RunnableLambda(
                self._summarize_node,
                afunc=self._asummarize_node,
            ),
        )
        workflow.add_node(
            "generate_answer",
            RunnableLambda(
                self._generate_answer_node,
                afunc=self._agenerate_answer_node,
            ),
        )

        workflow.add_edge(START, "prepare")
        workflow.add_conditional_edges(
            "prepare",
            self._route_before_planning,
            {
                "summarize": "summarize_for_planning",
                "plan": "plan_retrieval",
            },
        )
        workflow.add_edge(
            "summarize_for_planning",
            "plan_retrieval",
        )
        workflow.add_conditional_edges(
            "plan_retrieval",
            self._route_after_planning,
            {
                "retrieve": "retrieve",
                "summarize": "summarize_for_answer",
                "generate": "generate_answer",
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            self._route_before_generation,
            {
                "summarize": "summarize_for_answer",
                "generate": "generate_answer",
            },
        )
        workflow.add_edge(
            "summarize_for_answer",
            "generate_answer",
        )
        workflow.add_edge("generate_answer", END)
        return workflow.compile(
            checkpointer=self.checkpointer,
            name="youth_policy_rag",
        )

    def _build_planner_request(
        self,
        state: RAGGraphState,
    ) -> RetrievalPlanRequest:
        messages = state["messages"]
        previous_messages = (
            messages[:-1]
            if (
                messages
                and isinstance(messages[-1], HumanMessage)
                and messages[-1].text == state["user_input"]
            )
            else messages
        )
        user_history = [
            message
            for message in previous_messages
            if isinstance(message, HumanMessage)
        ]
        return {
            "current_question": state["user_input"],
            "documents": state.get("documents", []),
            "messages": user_history,
            "conversation_summary": (
                state.get("conversation_summary") or ""
            ),
        }

    def _build_generation_request(
        self,
        state: RAGGraphState,
    ) -> GenerationRequest:
        return {
            "user_profile": state["user_profile"],
            "documents": state.get("documents", []),
            "messages": state["messages"],
            "conversation_summary": (
                state.get("conversation_summary") or ""
            ),
            "retrieval_error": (
                state.get("retrieval_error") or ""
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
        force_retrieval = (
            not documents
            or self._retrieval_context_changed(state)
        )
        return {
            "answer": "",
            "documents": (
                []
                if force_retrieval
                else documents
            ),
            "force_retrieval": force_retrieval,
            "needs_retrieval": False,
            "retrieval_query": "",
            "retrieval_error": "",
        }

    def _should_summarize(
        self,
        *,
        prompt_messages,
        state: RAGGraphState,
    ) -> bool:
        return self.summarizer.should_summarize(
            prompt_messages=prompt_messages,
            conversation_messages=state["messages"],
        )

    def _route_before_planning(
        self,
        state: RAGGraphState,
    ) -> Literal["summarize", "plan"]:
        prompt_messages = self.planner.build_prompt_messages(
            self._build_planner_request(state),
            force_retrieval=state["force_retrieval"],
        )
        if self._should_summarize(
            prompt_messages=prompt_messages,
            state=state,
        ):
            return "summarize"
        return "plan"

    def _route_before_generation(
        self,
        state: RAGGraphState,
    ) -> Literal["summarize", "generate"]:
        prompt_messages = self.generator.build_prompt_messages(
            self._build_generation_request(state)
        )
        if self._should_summarize(
            prompt_messages=prompt_messages,
            state=state,
        ):
            return "summarize"
        return "generate"

    def _route_after_planning(
        self,
        state: RAGGraphState,
    ) -> Literal["retrieve", "summarize", "generate"]:
        if state["needs_retrieval"]:
            return "retrieve"
        return self._route_before_generation(state)

    def _summarize_node(
        self,
        state: RAGGraphState,
    ) -> dict:
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

    def _plan_retrieval_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        decision = self.planner.decide(
            self._build_planner_request(state),
            force_retrieval=state["force_retrieval"],
        )
        return {
            "needs_retrieval": decision.needs_retrieval,
            "retrieval_query": decision.query or "",
        }

    async def _aplan_retrieval_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        decision = await self.planner.adecide(
            self._build_planner_request(state),
            force_retrieval=state["force_retrieval"],
        )
        return {
            "needs_retrieval": decision.needs_retrieval,
            "retrieval_query": decision.query or "",
        }

    def _retrieval_update(
        self,
        state: RAGGraphState,
        documents: list[Document],
    ) -> dict:
        return {
            "documents": documents,
            "last_retrieval_query": state["retrieval_query"],
            "last_retrieval_profile": dict(
                state["user_profile"]
            ),
            "last_retrieval_exclude_expired": (
                state["exclude_expired"]
            ),
            "retrieval_error": "",
        }

    @staticmethod
    def _retrieval_failure_update(
        state: RAGGraphState,
    ) -> dict:
        return {
            "documents": state.get("documents", []),
            "retrieval_error": (
                "정책 검색 중 오류가 발생해 새로운 문서를 "
                "확인하지 못했습니다."
            ),
        }

    def _retrieve_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        try:
            documents = self.retriever.retrieve(
                query=state["retrieval_query"],
                user_profile=state["user_profile"],
                exclude_expired=state["exclude_expired"],
            )
        except Exception:
            return self._retrieval_failure_update(state)
        return self._retrieval_update(state, documents)

    async def _aretrieve_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        try:
            documents = await self.retriever.aretrieve(
                query=state["retrieval_query"],
                user_profile=state["user_profile"],
                exclude_expired=state["exclude_expired"],
            )
        except Exception:
            return self._retrieval_failure_update(state)
        return self._retrieval_update(state, documents)

    def _generate_answer_node(
        self,
        state: RAGGraphState,
    ) -> dict:
        answer = self.generator.generate(
            self._build_generation_request(state)
        )
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    async def _agenerate_answer_node(
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
                    and not prepare_update["force_retrieval"]
                ):
                    yield self._metadata_event(
                        prepare_update["documents"]
                    )

                retrieval_update = part["data"].get("retrieve")
                if retrieval_update is not None:
                    streamed_answer = ""
                    yield self._metadata_event(
                        retrieval_update.get("documents", [])
                    )

                generation_update = part["data"].get(
                    "generate_answer"
                )
                if (
                    generation_update
                    and generation_update.get("answer")
                    and not streamed_answer
                ):
                    answer = generation_update["answer"]
                    streamed_answer = answer
                    yield self._sse_event("chunk", answer)

            if part["type"] == "messages":
                message_chunk, metadata = part["data"]
                if (
                    metadata.get("langgraph_node")
                    != "generate_answer"
                    or metadata.get("ls_model_type") != "chat"
                ):
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
