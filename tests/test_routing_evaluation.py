import tempfile
import unittest
from pathlib import Path

from src.rag.router import RoutingDecision
from src.rag.routing_evaluation import (
    evaluate_routing,
    load_routing_evaluation_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data/eval/routing_eval.jsonl"


class SequenceRouter:
    def __init__(self, routes):
        self.routes = iter(routes)

    def decide(self, *, current_question, documents):
        route = next(self.routes)
        return RoutingDecision(
            route=route,
            reason="평가 테스트용 고정 판단입니다.",
        )


class RoutingEvaluationTest(unittest.TestCase):
    def test_dataset_is_balanced_and_has_unique_ids(self):
        cases = load_routing_evaluation_cases(DATASET_PATH)

        self.assertEqual(len(cases), 15)
        self.assertEqual(
            {
                route: sum(
                    case.expected_route == route
                    for case in cases
                )
                for route in ("reuse", "search", "clarify")
            },
            {"reuse": 5, "search": 5, "clarify": 5},
        )
        self.assertEqual(
            len({case.case_id for case in cases}),
            len(cases),
        )

    def test_evaluator_reports_overall_and_per_route_accuracy(self):
        cases = load_routing_evaluation_cases(DATASET_PATH)
        router = SequenceRouter(
            [case.expected_route for case in cases]
        )

        summary = evaluate_routing(router, cases)

        self.assertEqual(summary["total"], 15)
        self.assertEqual(summary["correct"], 15)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["error_count"], 0)
        for route in ("reuse", "search", "clarify"):
            self.assertEqual(
                summary["per_route"][route],
                {"correct": 5, "total": 5, "accuracy": 1.0},
            )

    def test_duplicate_case_id_is_rejected(self):
        duplicate_lines = "\n".join([
            (
                '{"case_id":"duplicate","current_question":"질문",'
                '"documents":[],"expected_route":"search",'
                '"rationale":"근거","tags":[]}'
            ),
            (
                '{"case_id":"duplicate","current_question":"다른 질문",'
                '"documents":[],"expected_route":"search",'
                '"rationale":"근거","tags":[]}'
            ),
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.jsonl"
            path.write_text(duplicate_lines, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "중복"):
                load_routing_evaluation_cases(path)


if __name__ == "__main__":
    unittest.main()
