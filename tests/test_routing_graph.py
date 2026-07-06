import unittest

from langchain_core.documents import Document

from src.rag.router import RoutingDecision
from src.rag.routing_graph import RoutingGraph


class FakeRouter:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def decide(self, *, current_question, documents):
        self.calls.append({
            "current_question": current_question,
            "documents": documents,
        })
        return self.decision


class RoutingGraphTest(unittest.TestCase):
    def setUp(self):
        self.active_documents = [
            Document(
                page_content="정책명: 청년 주거 지원",
                metadata={"plcyNo": "policy-1"},
            )
        ]
        self.new_documents = [
            Document(
                page_content="정책명: 청년 교통비 지원",
                metadata={"plcyNo": "policy-2"},
            )
        ]
        self.handler_calls = []

    def build_graph(self, decision):
        router = FakeRouter(decision)

        def reuse_handler(state):
            self.handler_calls.append("reuse")
            return {"executed_route": "reuse"}

        def search_handler(state):
            self.handler_calls.append("search")
            return {
                "executed_route": "search",
                "documents": self.new_documents,
            }

        def clarify_handler(state):
            self.handler_calls.append("clarify")
            return {"executed_route": "clarify"}

        graph = RoutingGraph(
            router=router,
            reuse_handler=reuse_handler,
            search_handler=search_handler,
            clarify_handler=clarify_handler,
        )
        return graph, router

    def test_reuse_keeps_active_documents(self):
        graph, router = self.build_graph(
            RoutingDecision(
                route="reuse",
                reason="기존 정책에 대한 후속 질문입니다.",
            )
        )

        result = graph.invoke(
            current_question="이 정책은 얼마를 지원해?",
            documents=self.active_documents,
        )

        self.assertEqual(result["executed_route"], "reuse")
        self.assertEqual(result["documents"], self.active_documents)
        self.assertEqual(self.handler_calls, ["reuse"])
        self.assertEqual(len(router.calls), 1)

    def test_search_replaces_active_documents(self):
        graph, _ = self.build_graph(
            RoutingDecision(
                route="search",
                reason="새로운 정책 주제입니다.",
            )
        )

        result = graph.invoke(
            current_question="교통비 지원 정책을 알려줘",
            documents=self.active_documents,
        )

        self.assertEqual(result["executed_route"], "search")
        self.assertEqual(result["documents"], self.new_documents)
        self.assertEqual(self.handler_calls, ["search"])

    def test_clarify_keeps_documents(self):
        graph, _ = self.build_graph(
            RoutingDecision(
                route="clarify",
                reason="지칭하는 정책이 불명확합니다.",
            )
        )

        result = graph.invoke(
            current_question="그거 자세히 알려줘",
            documents=self.active_documents,
        )

        self.assertEqual(result["executed_route"], "clarify")
        self.assertEqual(result["documents"], self.active_documents)
        self.assertEqual(self.handler_calls, ["clarify"])

    def test_empty_documents_go_directly_to_search(self):
        graph, router = self.build_graph(
            RoutingDecision(
                route="reuse",
                reason="호출되면 안 되는 응답입니다.",
            )
        )

        result = graph.invoke(
            current_question="청년 주거 정책을 알려줘",
            documents=[],
        )

        self.assertEqual(result["executed_route"], "search")
        self.assertEqual(result["documents"], self.new_documents)
        self.assertEqual(self.handler_calls, ["search"])
        self.assertEqual(router.calls, [])

    def test_router_must_return_structured_decision(self):
        graph, _ = self.build_graph({
            "route": "reuse",
            "reason": "구조화되지 않은 응답",
        })

        with self.assertRaisesRegex(
            TypeError,
            "RoutingDecision",
        ):
            graph.invoke(
                current_question="이 정책을 다시 설명해줘",
                documents=self.active_documents,
            )


if __name__ == "__main__":
    unittest.main()
