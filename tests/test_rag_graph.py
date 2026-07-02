import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from src.checkpoint import create_sqlite_checkpointer
from src.rag.agent import PolicyAgent
from src.rag.graph import RAGGraph
from src.rag.prompts import SUMMARY_SYSTEM_PROMPT
from src.rag.retriever import PolicyRetriever
from src.rag.summarizer import ConversationSummarizer
from src.rag.tools import create_search_policies_tool
from src.user.models import UserProfile


class FakeRetriever:
    def __init__(self, documents, queries):
        self.documents = documents
        self.queries = queries

    def invoke(self, query):
        self.queries.append(query)
        return self.documents

    async def ainvoke(self, query):
        self.queries.append(query)
        return self.documents


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.search_kwargs = None
        self.queries = []

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return FakeRetriever(self.documents, self.queries)


class RAGGraphTest(unittest.TestCase):
    def build_graph(
        self,
        *,
        llm,
        vector_store=None,
        search_k=3,
        max_input_tokens=32768,
        summary_trigger_ratio=0.65,
        summary_keep_recent_turns=3,
        token_chars_per_token=2.0,
        checkpointer=None,
    ):
        retriever = PolicyRetriever(
            vector_store=vector_store or self.vector_store,
            search_k=search_k,
        )
        search_tool = create_search_policies_tool(retriever)
        return RAGGraph(
            summarizer=ConversationSummarizer(
                llm,
                max_input_tokens=max_input_tokens,
                summary_trigger_ratio=summary_trigger_ratio,
                keep_recent_turns=summary_keep_recent_turns,
                chars_per_token=token_chars_per_token,
            ),
            agent=PolicyAgent(llm, [search_tool]),
            tool_node=ToolNode([search_tool]),
            checkpointer=checkpointer or InMemorySaver(),
        )

    def setUp(self):
        self.documents = [
            Document(
                page_content="정책명: 테스트 정책\n지원 내용: 최대 40만원",
                metadata={
                    "plcyNo": "policy-1",
                    "sprtTrgtMinAge": 19,
                    "sprtTrgtMaxAge": 39,
                },
            )
        ]
        self.vector_store = FakeVectorStore(self.documents)
        self.graph = self.build_graph(
            llm=RunnableLambda(lambda _: "테스트 생성 답변"),
        )
        self.user_profile = UserProfile(user_id="test-user")

    def test_generate_answer_returns_answer_contexts_and_policy_ids(self):
        result = self.graph.generate_answer(
            user_input="지원 내용을 알려줘",
            user_profile=self.user_profile,
            exclude_expired=False,
        )

        self.assertEqual(result.answer, "테스트 생성 답변")
        self.assertEqual(result.retrieved_policy_ids, ["policy-1"])
        self.assertEqual(len(result.contexts), 1)
        self.assertIn("테스트 정책", result.contexts[0])
        self.assertEqual(self.vector_store.search_kwargs, {"k": 3})

    def test_generate_answer_requires_user_profile_id(self):
        with self.assertRaisesRegex(
            ValueError,
            "user_profile.user_id",
        ):
            self.graph.generate_answer(
                user_input="지원 내용을 알려줘",
                user_profile=UserProfile(),
                exclude_expired=False,
            )

    def test_compiled_graph_has_conditional_summarization_topology(self):
        compiled_graph = self.graph.graph.get_graph()

        self.assertEqual(
            set(compiled_graph.nodes),
            {
                "__start__",
                "prepare",
                "summarize",
                "agent",
                "tools",
                "__end__",
            },
        )
        self.assertEqual(
            {
                (edge.source, edge.target, edge.conditional)
                for edge in compiled_graph.edges
            },
            {
                ("__start__", "prepare", False),
                ("prepare", "agent", True),
                ("prepare", "summarize", True),
                ("summarize", "agent", False),
                ("agent", "tools", True),
                ("agent", "__end__", True),
                ("tools", "agent", True),
                ("tools", "summarize", True),
            },
        )

    def test_prompt_token_count_uses_configured_approximation(self):
        messages = [
            HumanMessage(content="12345678"),
        ]

        conservative_count = self.build_graph(
            llm=RunnableLambda(lambda _: "답변"),
            token_chars_per_token=1.0,
        ).summarizer.count_prompt_tokens(
            messages
        )
        relaxed_count = self.build_graph(
            llm=RunnableLambda(lambda _: "답변"),
            token_chars_per_token=4.0,
        ).summarizer.count_prompt_tokens(
            messages
        )

        self.assertGreater(conservative_count, relaxed_count)

    def test_agenerate_answer_returns_same_result_shape(self):
        result = asyncio.run(
            self.graph.agenerate_answer(
                user_input="지원 내용을 알려줘",
                user_profile=self.user_profile,
                exclude_expired=False,
            )
        )

        self.assertEqual(result.answer, "테스트 생성 답변")
        self.assertEqual(result.retrieved_policy_ids, ["policy-1"])
        self.assertEqual(len(result.contexts), 1)

    def test_same_user_keeps_history_and_other_users_are_isolated(self):
        prompts = []

        def respond(prompt):
            messages = prompt.to_messages()
            prompts.append([message.content for message in messages])
            return f"답변-{len(prompts)}"

        graph = self.build_graph(
            llm=RunnableLambda(respond),
        )
        profile = UserProfile(user_id="user-a")

        graph.generate_answer(
            user_input="첫 번째 질문",
            user_profile=profile,
            exclude_expired=False,
        )
        graph.generate_answer(
            user_input="앞 답변을 더 설명해줘",
            user_profile=profile,
            exclude_expired=False,
        )
        graph.generate_answer(
            user_input="새 사용자 질문",
            user_profile=UserProfile(user_id="user-b"),
            exclude_expired=False,
        )

        self.assertIn("첫 번째 질문", prompts[1])
        self.assertIn("답변-1", prompts[1])
        self.assertNotIn("첫 번째 질문", prompts[2])
        self.assertNotIn("답변-1", prompts[2])
        self.assertEqual(
            self.vector_store.queries,
            [
                "첫 번째 질문",
                "새 사용자 질문",
            ],
        )

    def test_follow_up_reuses_existing_documents(self):
        graph = self.build_graph(
            llm=RunnableLambda(lambda _: "답변"),
        )
        profile = UserProfile(user_id="reuse-user")

        graph.generate_answer(
            user_input="테스트 정책 알려줘",
            user_profile=profile,
            exclude_expired=False,
        )
        result = graph.generate_answer(
            user_input="지원 내용을 더 설명해줘",
            user_profile=profile,
            exclude_expired=False,
        )

        self.assertEqual(
            self.vector_store.queries,
            ["테스트 정책 알려줘"],
        )
        self.assertEqual(
            result.retrieved_policy_ids,
            ["policy-1"],
        )

    def test_agent_calls_search_tool_for_new_topic(self):
        responses = iter([
            "월세 정책 답변",
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_policies",
                        "args": {
                            "query": "서울 청년 취업 지원 정책"
                        },
                        "id": "search-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            "취업 정책 답변",
        ])

        graph = self.build_graph(
            llm=RunnableLambda(lambda _: next(responses)),
        )
        profile = UserProfile(user_id="topic-user")

        graph.generate_answer(
            user_input="월세 정책 알려줘",
            user_profile=profile,
            exclude_expired=False,
        )
        graph.generate_answer(
            user_input="취업 지원은?",
            user_profile=profile,
            exclude_expired=False,
        )

        self.assertEqual(
            self.vector_store.queries,
            [
                "월세 정책 알려줘",
                "서울 청년 취업 지원 정책",
            ],
        )
        snapshot = graph.graph.get_state(
            {"configurable": {"thread_id": "topic-user"}}
        )
        self.assertEqual(
            snapshot.values["retrieval_mode"],
            "disabled",
        )
        self.assertFalse(
            any(
                isinstance(message, ToolMessage)
                for message in snapshot.values["messages"]
            )
        )
        self.assertFalse(
            any(
                isinstance(message, AIMessage)
                and message.tool_calls
                for message in snapshot.values["messages"]
            )
        )

    def test_profile_change_forces_retrieval(self):
        graph = self.build_graph(
            llm=RunnableLambda(lambda _: "답변"),
        )

        graph.generate_answer(
            user_input="내가 받을 정책 알려줘",
            user_profile=UserProfile(
                user_id="profile-user",
                age=24,
            ),
            exclude_expired=False,
        )
        graph.generate_answer(
            user_input="지금도 받을 수 있어?",
            user_profile=UserProfile(
                user_id="profile-user",
                age=25,
            ),
            exclude_expired=False,
        )

        self.assertEqual(
            self.vector_store.queries,
            [
                "내가 받을 정책 알려줘",
                "지금도 받을 수 있어?",
            ],
        )

    def test_long_history_is_summarized_before_generation(self):
        summaries = []
        generation_prompts = []
        answer_count = 0

        def respond(prompt):
            nonlocal answer_count
            messages = prompt.to_messages()
            if messages[0].content.startswith(SUMMARY_SYSTEM_PROMPT):
                summaries.append([message.content for message in messages])
                return "첫 질문과 첫 답변의 압축 요약"

            answer_count += 1
            generation_prompts.append(
                [message.content for message in messages]
            )
            return f"답변-{answer_count}"

        graph = self.build_graph(
            llm=RunnableLambda(respond),
            max_input_tokens=10,
            summary_trigger_ratio=0.5,
            summary_keep_recent_turns=1,
            token_chars_per_token=1.0,
        )
        profile = UserProfile(user_id="summary-user")

        for question in ["첫 질문", "두 번째 질문", "세 번째 질문"]:
            graph.generate_answer(
                user_input=question,
                user_profile=profile,
                exclude_expired=False,
            )

        self.assertEqual(len(summaries), 1)
        self.assertIn("첫 질문", summaries[0])
        self.assertIn("답변-1", summaries[0])
        self.assertIn(
            "첫 질문과 첫 답변의 압축 요약",
            generation_prompts[-1][0],
        )
        self.assertNotIn("첫 질문", generation_prompts[-1][1:])
        self.assertNotIn("답변-1", generation_prompts[-1][1:])

        snapshot = graph.graph.get_state(
            {"configurable": {"thread_id": "summary-user"}}
        )
        remaining_messages = [
            message.content for message in snapshot.values["messages"]
        ]
        self.assertNotIn("첫 질문", remaining_messages)
        self.assertNotIn("답변-1", remaining_messages)
        self.assertEqual(
            snapshot.values["conversation_summary"],
            "첫 질문과 첫 답변의 압축 요약",
        )

    def test_async_summary_is_not_emitted_as_answer_chunk(self):
        summary_calls = []
        answer_count = 0

        def respond(prompt):
            nonlocal answer_count
            messages = prompt.to_messages()
            if messages[0].content.startswith(SUMMARY_SYSTEM_PROMPT):
                summary_calls.append(messages)
                return "사용자에게 노출하면 안 되는 내부 요약"

            answer_count += 1
            return f"스트리밍 답변-{answer_count}"

        graph = self.build_graph(
            llm=RunnableLambda(respond),
            max_input_tokens=10,
            summary_trigger_ratio=0.5,
            summary_keep_recent_turns=1,
            token_chars_per_token=1.0,
        )
        profile = UserProfile(user_id="async-summary-user")

        async def run_conversation():
            all_events = []
            for question in ["첫 질문", "두 번째 질문", "세 번째 질문"]:
                events = [
                    json.loads(event.removeprefix("data: ").strip())
                    async for event in graph.stream_answer(
                        user_input=question,
                        user_profile=profile,
                        exclude_expired=False,
                    )
                ]
                all_events.append(events)
            return all_events

        all_events = asyncio.run(run_conversation())
        third_chunks = [
            event["data"]
            for event in all_events[-1]
            if event["type"] == "chunk"
        ]

        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(third_chunks, ["스트리밍 답변-3"])

    def test_delete_conversation_clears_user_history(self):
        prompts = []

        def respond(prompt):
            prompts.append(
                [message.content for message in prompt.to_messages()]
            )
            return "답변"

        graph = self.build_graph(
            llm=RunnableLambda(respond),
        )
        profile = UserProfile(user_id="reset-user")
        graph.generate_answer(
            user_input="삭제 전 질문",
            user_profile=profile,
            exclude_expired=False,
        )

        graph.delete_conversation("reset-user")
        graph.generate_answer(
            user_input="삭제 후 질문",
            user_profile=profile,
            exclude_expired=False,
        )

        self.assertNotIn("삭제 전 질문", prompts[1])

    def test_sqlite_checkpointer_restores_history_after_reopen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = str(Path(temp_dir) / "conversations.db")
            profile = UserProfile(user_id="persistent-user")

            first_graph = self.build_graph(
                llm=RunnableLambda(lambda _: "저장된 답변"),
                checkpointer=create_sqlite_checkpointer(checkpoint_path),
            )
            first_graph.generate_answer(
                user_input="재시작 전 질문",
                user_profile=profile,
                exclude_expired=False,
            )
            first_graph.close()

            restored_prompts = []

            def respond(prompt):
                restored_prompts.append(
                    [message.content for message in prompt.to_messages()]
                )
                return "재시작 후 답변"

            second_graph = self.build_graph(
                llm=RunnableLambda(respond),
                checkpointer=create_sqlite_checkpointer(checkpoint_path),
            )
            try:
                second_graph.generate_answer(
                    user_input="이전 대화 기억해?",
                    user_profile=profile,
                    exclude_expired=False,
                )
            finally:
                second_graph.close()

        self.assertIn("재시작 전 질문", restored_prompts[0])
        self.assertIn("저장된 답변", restored_prompts[0])

    def test_sqlite_checkpointer_supports_async_streaming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph = self.build_graph(
                llm=RunnableLambda(lambda _: "비동기 답변"),
                checkpointer=create_sqlite_checkpointer(
                    str(Path(temp_dir) / "conversations.db")
                ),
            )

            async def collect_events():
                return [
                    json.loads(event.removeprefix("data: ").strip())
                    async for event in graph.stream_answer(
                        user_input="비동기 질문",
                        user_profile=UserProfile(user_id="async-user"),
                        exclude_expired=False,
                    )
                ]

            try:
                events = asyncio.run(collect_events())
            finally:
                graph.close()

        self.assertEqual(events[0]["type"], "metadata")
        self.assertEqual(events[-1]["type"], "done")

    def test_contexts_keep_retrieval_order_labels(self):
        documents = self.documents + [
            Document(
                page_content="정책명: 두 번째 정책",
                metadata={"plcyNo": "policy-2"},
            )
        ]
        graph = self.build_graph(
            llm=RunnableLambda(lambda _: "테스트 생성 답변"),
            vector_store=FakeVectorStore(documents),
        )

        result = graph.generate_answer(
            user_input="정책을 알려줘",
            user_profile=self.user_profile,
            exclude_expired=False,
        )

        self.assertTrue(result.contexts[0].startswith("[검색 결과 1]"))
        self.assertTrue(result.contexts[1].startswith("[검색 결과 2]"))
        self.assertEqual(
            result.retrieved_policy_ids,
            ["policy-1", "policy-2"],
        )

    def test_stream_answer_emits_metadata_before_chunks(self):
        async def collect_events():
            return [
                event
                async for event in self.graph.stream_answer(
                    user_input="지원 내용을 알려줘",
                    user_profile=self.user_profile,
                    exclude_expired=False,
                )
            ]

        raw_events = asyncio.run(collect_events())
        events = [
            json.loads(event.removeprefix("data: ").strip())
            for event in raw_events
        ]

        self.assertEqual(events[0]["type"], "metadata")
        self.assertEqual(
            events[0]["data"]["retrieved_policy_ids"],
            ["policy-1"],
        )
        self.assertEqual(len(events[0]["data"]["contexts"]), 1)
        self.assertEqual(events[1]["type"], "chunk")
        self.assertEqual(events[1]["data"], "테스트 생성 답변")
        self.assertEqual(events[-1]["type"], "done")

    def test_stream_answer_emits_reused_document_metadata(self):
        graph = self.build_graph(
            llm=RunnableLambda(lambda _: "답변"),
        )
        profile = UserProfile(user_id="stream-reuse-user")
        graph.generate_answer(
            user_input="테스트 정책 알려줘",
            user_profile=profile,
            exclude_expired=False,
        )

        async def collect_events():
            return [
                json.loads(event.removeprefix("data: ").strip())
                async for event in graph.stream_answer(
                    user_input="지원 내용을 더 설명해줘",
                    user_profile=profile,
                    exclude_expired=False,
                )
            ]

        events = asyncio.run(collect_events())

        self.assertEqual(events[0]["type"], "metadata")
        self.assertEqual(
            events[0]["data"]["retrieved_policy_ids"],
            ["policy-1"],
        )
        self.assertEqual(
            self.vector_store.queries,
            ["테스트 정책 알려줘"],
        )

    def test_stream_answer_emits_tool_search_metadata(self):
        responses = iter([
            "월세 정책 답변",
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_policies",
                        "args": {
                            "query": "서울 청년 취업 지원 정책"
                        },
                        "id": "stream-search-call",
                        "type": "tool_call",
                    }
                ],
            ),
            "취업 정책 답변",
        ])
        graph = self.build_graph(
            llm=RunnableLambda(lambda _: next(responses)),
        )
        profile = UserProfile(user_id="stream-tool-user")
        graph.generate_answer(
            user_input="월세 정책 알려줘",
            user_profile=profile,
            exclude_expired=False,
        )

        async def collect_events():
            return [
                json.loads(event.removeprefix("data: ").strip())
                async for event in graph.stream_answer(
                    user_input="취업 지원은?",
                    user_profile=profile,
                    exclude_expired=False,
                )
            ]

        events = asyncio.run(collect_events())
        metadata_events = [
            event
            for event in events
            if event["type"] == "metadata"
        ]
        chunks = [
            event["data"]
            for event in events
            if event["type"] == "chunk"
        ]

        self.assertEqual(len(metadata_events), 2)
        self.assertEqual(chunks, ["취업 정책 답변"])
        self.assertEqual(
            self.vector_store.queries,
            [
                "월세 정책 알려줘",
                "서울 청년 취업 지원 정책",
            ],
        )

    def test_stream_answer_forwards_model_tokens_without_duplication(self):
        graph = self.build_graph(
            llm=FakeListChatModel(responses=["ABC"]),
        )

        async def collect_events():
            return [
                json.loads(event.removeprefix("data: ").strip())
                async for event in graph.stream_answer(
                    user_input="지원 내용을 알려줘",
                    user_profile=self.user_profile,
                    exclude_expired=False,
                )
            ]

        events = asyncio.run(collect_events())
        chunks = [
            event["data"]
            for event in events
            if event["type"] == "chunk"
        ]

        self.assertEqual(chunks, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
