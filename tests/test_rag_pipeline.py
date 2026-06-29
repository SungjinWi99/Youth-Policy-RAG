import asyncio
import json
import unittest

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.chat.rag import RAGPipeline
from src.user.models import UserProfile


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, _query):
        return self.documents

    async def ainvoke(self, _query):
        return self.documents


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.search_kwargs = None

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return FakeRetriever(self.documents)


class RAGPipelineTest(unittest.TestCase):
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
        self.pipeline = RAGPipeline(
            llm=RunnableLambda(lambda _: "테스트 생성 답변"),
            vector_store=self.vector_store,
            search_k=3,
        )
        self.user_profile = UserProfile()

    def test_generate_answer_returns_answer_contexts_and_policy_ids(self):
        result = self.pipeline.generate_answer(
            user_input="지원 내용을 알려줘",
            user_profile=self.user_profile,
            exclude_expired=False,
        )

        self.assertEqual(result.answer, "테스트 생성 답변")
        self.assertEqual(result.retrieved_policy_ids, ["policy-1"])
        self.assertEqual(len(result.contexts), 1)
        self.assertIn("테스트 정책", result.contexts[0])
        self.assertEqual(self.vector_store.search_kwargs, {"k": 3})

    def test_agenerate_answer_returns_same_result_shape(self):
        result = asyncio.run(
            self.pipeline.agenerate_answer(
                user_input="지원 내용을 알려줘",
                user_profile=self.user_profile,
                exclude_expired=False,
            )
        )

        self.assertEqual(result.answer, "테스트 생성 답변")
        self.assertEqual(result.retrieved_policy_ids, ["policy-1"])
        self.assertEqual(len(result.contexts), 1)

    def test_contexts_keep_retrieval_order_labels(self):
        documents = self.documents + [
            Document(
                page_content="정책명: 두 번째 정책",
                metadata={"plcyNo": "policy-2"},
            )
        ]
        pipeline = RAGPipeline(
            llm=RunnableLambda(lambda _: "테스트 생성 답변"),
            vector_store=FakeVectorStore(documents),
            search_k=3,
        )

        result = pipeline.generate_answer(
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
                async for event in self.pipeline.stream_answer(
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


if __name__ == "__main__":
    unittest.main()
