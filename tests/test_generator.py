import asyncio
import unittest

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.rag.generator import AnswerGenerator


class AnswerGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.prompts = []

        def respond(prompt):
            self.prompts.append(prompt)
            return "문서 기반 답변"

        self.generator = AnswerGenerator(
            RunnableLambda(respond)
        )
        self.request = {
            "user_profile": {
                "age": 25,
                "region": "서울특별시",
            },
            "documents": [
                Document(
                    page_content=(
                        "정책명: 테스트 정책\n"
                        "지원 내용: 최대 40만원"
                    ),
                    metadata={"plcyNo": "policy-1"},
                )
            ],
            "messages": [
                HumanMessage(content="얼마를 지원해줘?")
            ],
            "conversation_summary": "테스트 정책을 안내함",
            "retrieval_error": "",
        }

    def test_prompt_contains_answer_context_without_search_rules(self):
        messages = self.generator.build_prompt_messages(
            self.request
        )
        system_prompt = messages[0].text

        self.assertIn("최대 40만원", system_prompt)
        self.assertIn("서울특별시", system_prompt)
        self.assertIn("테스트 정책을 안내함", system_prompt)
        self.assertNotIn("search_policies", system_prompt)
        self.assertNotIn("force_retrieval", system_prompt)

    def test_generate_returns_plain_answer(self):
        answer = self.generator.generate(self.request)

        self.assertEqual(answer, "문서 기반 답변")

    def test_agenerate_returns_plain_answer(self):
        answer = asyncio.run(
            self.generator.agenerate(self.request)
        )

        self.assertEqual(answer, "문서 기반 답변")


if __name__ == "__main__":
    unittest.main()
