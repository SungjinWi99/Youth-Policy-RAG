import asyncio
import unittest

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.rag.planner import (
    RetrievalDecision,
    RetrievalPlanner,
    format_documents_for_planning,
)


class FakeStructuredModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema

        def respond(prompt):
            self.prompts.append(prompt)
            return self.responses.pop(0)

        return RunnableLambda(respond)


class RetrievalPlannerTest(unittest.TestCase):
    def build_request(self):
        return {
            "current_question": "이 정책의 신청 방법은?",
            "documents": [
                Document(
                    page_content=(
                        "정책명: 테스트 주거 정책\n"
                        "지원 내용: 월세 지원"
                    )
                )
            ],
            "messages": [
                HumanMessage(content="주거 정책을 알려줘")
            ],
            "conversation_summary": "주거 정책을 안내함",
        }

    def test_uses_structured_decision_schema(self):
        model = FakeStructuredModel([
            RetrievalDecision(
                needs_retrieval=False,
                query=None,
            )
        ])
        planner = RetrievalPlanner(model)

        decision = planner.decide(
            self.build_request(),
            force_retrieval=False,
        )

        self.assertIs(model.schema, RetrievalDecision)
        self.assertFalse(decision.needs_retrieval)
        self.assertIsNone(decision.query)

    def test_force_retrieval_overrides_model_decision(self):
        model = FakeStructuredModel([
            RetrievalDecision(
                needs_retrieval=False,
                query=None,
            )
        ])
        planner = RetrievalPlanner(model)

        decision = planner.decide(
            self.build_request(),
            force_retrieval=True,
        )

        self.assertTrue(decision.needs_retrieval)
        self.assertEqual(
            decision.query,
            "이 정책의 신청 방법은?",
        )

    def test_non_retrieval_discards_unneeded_query(self):
        planner = RetrievalPlanner(FakeStructuredModel([
            RetrievalDecision(
                needs_retrieval=False,
                query="사용하면 안 되는 검색어",
            )
        ]))

        decision = planner.decide(
            self.build_request(),
            force_retrieval=False,
        )

        self.assertIsNone(decision.query)

    def test_prompt_uses_document_body_without_user_profile(self):
        planner = RetrievalPlanner(FakeStructuredModel([
            RetrievalDecision(
                needs_retrieval=True,
                query="월세 정책의 신청 방법",
            )
        ]))

        messages = planner.build_prompt_messages(
            self.build_request(),
            force_retrieval=False,
        )
        system_prompt = messages[0].text

        self.assertIn("정책명: 테스트 주거 정책", system_prompt)
        self.assertIn("사용자 발화에 없는 나이", system_prompt)
        self.assertNotIn("사용자 프로필:", system_prompt)
        self.assertNotIn("서울특별시", system_prompt)

    def test_current_question_is_repeated_after_conversation(self):
        request = self.build_request()
        request["messages"] = [
            HumanMessage(content="면접 정장 지원이 필요해"),
        ]
        request["current_question"] = "결혼 자금이 부족해"
        planner = RetrievalPlanner(FakeStructuredModel([
            RetrievalDecision(
                needs_retrieval=True,
                query="결혼 비용 지원 정책",
            )
        ]))

        messages = planner.build_prompt_messages(
            request,
            force_retrieval=False,
        )

        self.assertIn(
            "현재 반드시 판단해야 하는 질문",
            messages[-1].text,
        )
        self.assertIn("결혼 자금이 부족해", messages[-1].text)
        self.assertNotIn("이전 정책 검색어", messages[0].text)

    def test_prompt_treats_topic_change_as_retrieval(self):
        planner = RetrievalPlanner(FakeStructuredModel([
            RetrievalDecision(
                needs_retrieval=True,
                query="교통비 지원 정책",
            )
        ]))

        messages = planner.build_prompt_messages(
            self.build_request(),
            force_retrieval=False,
        )
        system_prompt = messages[0].text

        self.assertIn(
            '월세 지원 정책이고 현재 질문이 "교통비 지원 정책도 있어?"',
            system_prompt,
        )
        self.assertIn(
            "조금이라도 불확실하면 검색",
            system_prompt,
        )

    def test_async_decision_matches_sync_contract(self):
        planner = RetrievalPlanner(FakeStructuredModel([
            RetrievalDecision(
                needs_retrieval=True,
                query="월세 정책 신청 방법",
            )
        ]))

        decision = asyncio.run(
            planner.adecide(
                self.build_request(),
                force_retrieval=False,
            )
        )

        self.assertTrue(decision.needs_retrieval)
        self.assertEqual(decision.query, "월세 정책 신청 방법")

    def test_document_formatter_returns_only_page_content(self):
        documents = [
            Document(
                page_content="첫 번째 본문",
                metadata={"plcyNo": "policy-1"},
            ),
            Document(
                page_content="두 번째 본문",
                metadata={"plcyNo": "policy-2"},
            ),
        ]

        formatted = format_documents_for_planning(documents)

        self.assertIn("첫 번째 본문", formatted)
        self.assertIn("두 번째 본문", formatted)
        self.assertNotIn("policy-1", formatted)


if __name__ == "__main__":
    unittest.main()
