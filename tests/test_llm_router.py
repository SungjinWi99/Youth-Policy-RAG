import unittest

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.rag.prompts import ROUTER_SYSTEM_PROMPT
from src.rag.router import (
    LLMContextRouter,
    RoutingDecision,
    format_routing_documents,
)


class FakeStructuredLlm:
    def __init__(self, decision):
        self.decision = decision
        self.schema = None
        self.prompt_values = []

    def with_structured_output(self, schema):
        self.schema = schema

        def respond(prompt_value):
            self.prompt_values.append(prompt_value)
            return self.decision

        return RunnableLambda(respond)


class LLMContextRouterTest(unittest.TestCase):
    def test_router_uses_structured_output_and_document_context(self):
        llm = FakeStructuredLlm(
            RoutingDecision(
                route="reuse",
                reason="동일 정책의 후속 질문입니다.",
            )
        )
        router = LLMContextRouter(llm)
        document = Document(
            page_content="정책명: 청년 월세 지원",
            metadata={"plcyNo": "policy-1"},
        )

        result = router.decide(
            current_question="이 정책의 신청 기간은?",
            documents=[document],
        )

        self.assertEqual(result.route, "reuse")
        self.assertIs(llm.schema, RoutingDecision)
        messages = llm.prompt_values[0].to_messages()
        self.assertIn(ROUTER_SYSTEM_PROMPT, messages[0].content)
        self.assertIn("이 정책의 신청 기간은?", messages[1].content)
        self.assertIn("policy-1", messages[1].content)
        self.assertIn("청년 월세 지원", messages[1].content)

    def test_document_formatter_marks_empty_context(self):
        self.assertEqual(
            format_routing_documents([]),
            "(활성 정책 문서 없음)",
        )

    def test_document_formatter_limits_each_document(self):
        formatted = format_routing_documents(
            [
                Document(
                    page_content="가" * 20,
                    metadata={"plcyNo": "policy-1"},
                )
            ],
            max_content_chars=10,
        )

        self.assertIn("가" * 10 + "…", formatted)
        self.assertNotIn("가" * 11, formatted)


if __name__ == "__main__":
    unittest.main()
