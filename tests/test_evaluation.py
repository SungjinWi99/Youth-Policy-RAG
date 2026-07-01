import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from scripts.create_eval_dataset import get_langsmith_ui_url
from scripts.run_evaluation import build_evaluator_llm, run_rag_target
from src.evaluators import (
    build_evaluators,
    calculate_average_precision,
    calculate_context_recall,
    load_examples,
)
from src.rag.state import RAGResult


EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "data/eval/examples.jsonl"


class FakeJudge:
    def with_structured_output(self, schema):
        return RunnableLambda(
            lambda _: schema(
                score=0.75,
                reasoning="평가용 고정 점수입니다.",
            )
        )


class FakeRAG:
    def generate_answer(self, user_input, user_profile, exclude_expired):
        self.call = {
            "user_input": user_input,
            "user_profile": user_profile,
            "exclude_expired": exclude_expired,
        }
        return RAGResult(
            answer="평가 답변",
            contexts=["평가 context"],
            retrieved_policy_ids=["policy-1"],
        )


class EvaluationTest(unittest.TestCase):
    def test_sample_count_and_required_fields(self):
        examples = load_examples(EXAMPLES_PATH)
        self.assertEqual(len(examples), 16)

        for example in examples:
            self.assertIn("question", example["inputs"])
            self.assertIn("user_profile", example["inputs"])
            self.assertIn("reference_answer", example["outputs"])
            self.assertTrue(example["outputs"]["reference_contexts"])
            self.assertTrue(example["outputs"]["expected_policy_ids"])

    def test_average_precision_rewards_relevant_contexts_earlier(self):
        expected = ["policy-a", "policy-b"]
        top_ranked = calculate_average_precision(
            ["policy-a", "policy-b", "policy-x"],
            expected,
        )
        lower_ranked = calculate_average_precision(
            ["policy-x", "policy-a", "policy-b"],
            expected,
        )

        self.assertEqual(top_ranked, 1.0)
        self.assertGreater(top_ranked, lower_ranked)

    def test_average_precision_handles_no_relevant_context(self):
        self.assertEqual(
            calculate_average_precision(
                ["policy-x", "policy-y"],
                ["policy-a"],
            ),
            0.0,
        )

    def test_context_recall_counts_all_expected_policies(self):
        self.assertEqual(
            calculate_context_recall(
                ["policy-a", "policy-x"],
                ["policy-a", "policy-b"],
            ),
            0.5,
        )

    def test_all_evaluators_return_langsmith_feedback(self):
        evaluators = build_evaluators(FakeJudge())
        inputs = {
            "question": "질문",
            "user_profile": {"age": 25},
        }
        outputs = {
            "answer": "생성 답변",
            "contexts": ["첫 번째", "두 번째", "세 번째"],
            "retrieved_policy_ids": ["policy-a", "policy-x", "policy-b"],
        }
        reference_outputs = {
            "reference_answer": "기준 답변",
            "expected_policy_ids": ["policy-a", "policy-b"],
        }

        results = [
            evaluator(outputs, reference_outputs)
            if evaluator.__name__ in {"context_recall", "context_precision"}
            else evaluator(inputs, outputs)
            for evaluator in evaluators
        ]

        self.assertEqual(
            {result["key"] for result in results},
            {
                "context_recall",
                "context_precision",
                "faithfulness",
                "answer_relevance",
            },
        )
        for result in results:
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)
            self.assertTrue(result["comment"])

    def test_run_rag_target_uses_graph_generate_answer(self):
        rag = FakeRAG()
        result = run_rag_target(
            {
                "question": "질문",
                "user_profile": {"age": 25},
                "exclude_expired": False,
            },
            rag=rag,
        )

        self.assertEqual(result["answer"], "평가 답변")
        self.assertEqual(result["contexts"], ["평가 context"])
        self.assertEqual(result["retrieved_policy_ids"], ["policy-1"])
        self.assertEqual(rag.call["user_input"], "질문")
        self.assertEqual(rag.call["user_profile"].age, 25)
        self.assertFalse(rag.call["exclude_expired"])

    def test_apac_api_url_maps_to_apac_ui(self):
        self.assertEqual(
            get_langsmith_ui_url("https://apac.api.smith.langchain.com"),
            "https://apac.smith.langchain.com",
        )

    @patch("scripts.run_evaluation.create_chat_model")
    def test_evaluator_model_uses_shared_factory(self, create_chat_model):
        config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="google",
                model="generator-model",
            ),
            evaluation=SimpleNamespace(
                provider="upstage",
                model="judge-model",
            ),
        )
        create_chat_model.return_value = "judge-instance"

        result = build_evaluator_llm(config, fallback_llm="generator")

        create_chat_model.assert_called_once_with(
            provider="upstage",
            model_name="judge-model",
            temperature=0,
        )
        self.assertEqual(result, "judge-instance")


if __name__ == "__main__":
    unittest.main()
