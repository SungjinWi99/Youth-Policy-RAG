import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.evaluate_embedding_retrieval import (
    RetrievalEvaluationCase,
    create_query_embedding_model,
    evaluate_retrieval,
    load_evaluation_cases,
    rank_gold_policy_ids,
    recall_at_k,
    validate_collection_corpus,
    validate_filter_metadata,
    validate_gold_coverage,
    write_results,
)


class FakeEmbeddingModel:
    def embed_query(self, query):
        return [float(len(query))]


class FakeCollection:
    def __init__(self):
        self.stored_ids = ["p1", "p2", "p3", "p4"]
        self.where_calls = []
        self.query_results = {
            2.0: ["p2", "p1", "p3", "p4"],
            3.0: ["p4", "p3", "p2", "p1"],
        }

    def count(self):
        return len(self.stored_ids)

    def get(self, ids=None, include=None, where=None):
        self.where_calls.append(where)
        if ids is None:
            return {"ids": self.stored_ids}
        return {
            "ids": [
                policy_id
                for policy_id in ids
                if policy_id in self.stored_ids
            ]
        }

    def query(
        self,
        query_embeddings,
        n_results,
        include,
        where=None,
    ):
        self.where_calls.append(where)
        ranked_ids = self.query_results[query_embeddings[0][0]][:n_results]
        return {
            "ids": [ranked_ids],
            "distances": [
                [float(index) for index in range(len(ranked_ids))]
            ],
        }


class EmbeddingRetrievalEvaluationTest(unittest.TestCase):
    def test_recall_at_k_supports_multiple_gold_ids(self):
        retrieved = ["p1", "p2", "p3", "p4"]
        gold = ["p2", "p4"]

        self.assertEqual(recall_at_k(retrieved, gold, 1), 0.0)
        self.assertEqual(recall_at_k(retrieved, gold, 3), 0.5)
        self.assertEqual(recall_at_k(retrieved, gold, 5), 1.0)

    def test_rank_gold_policy_ids_uses_one_based_rank(self):
        self.assertEqual(
            rank_gold_policy_ids(
                ["p3", "p1", "p2"],
                ["p1", "missing"],
            ),
            {"p1": 2, "missing": None},
        )

    def test_evaluate_retrieval_calculates_recall_and_rank_metrics(self):
        cases = [
            RetrievalEvaluationCase(
                gold_policy_ids=["p1"],
                user_input="aa",
                user_profile={"age": 25, "region": "서울"},
            ),
            RetrievalEvaluationCase(
                gold_policy_ids=["p4"],
                user_input="bbb",
            ),
        ]

        summary, details = evaluate_retrieval(
            collection=FakeCollection(),
            embedding_model=FakeEmbeddingModel(),
            cases=cases,
            k_values=(1, 3),
        )

        metrics = summary["metrics"]
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)
        self.assertEqual(metrics["mean_gold_rank"], 1.5)
        self.assertEqual(metrics["gold_filter_eligibility_rate"], 1.0)
        self.assertEqual(details[0]["gold_ranks"], {"p1": 2})
        self.assertEqual(details[1]["gold_ranks"], {"p4": 1})
        self.assertEqual(
            details[0]["metadata_filter"],
            {
                "$and": [
                    {
                        "$or": [
                            {
                                "agePolicy": {
                                    "$in": ["all", "unknown"]
                                }
                            },
                            {
                                "$and": [
                                    {"agePolicy": {"$eq": "specific"}},
                                    {"sprtTrgtMinAge": {"$lte": 25}},
                                    {"sprtTrgtMaxAge": {"$gte": 25}},
                                ]
                            },
                        ]
                    },
                    {"region_11": {"$eq": True}},
                ]
            },
        )

    def test_collection_validation_detects_corpus_mismatch(self):
        with self.assertRaisesRegex(ValueError, "일치하지 않습니다"):
            validate_collection_corpus(
                FakeCollection(),
                {"p1", "p2", "p3", "missing"},
            )

    def test_gold_coverage_detects_missing_policy(self):
        cases = [
            RetrievalEvaluationCase(
                gold_policy_ids=["missing"],
                user_input="question",
            )
        ]

        with self.assertRaisesRegex(ValueError, "gold policy"):
            validate_gold_coverage(FakeCollection(), cases)

    def test_filter_metadata_validation_reports_missing_keys(self):
        collection = SimpleNamespace(
            get=lambda include: {
                "ids": ["p1"],
                "metadatas": [{"agePolicy": "all"}],
            }
        )

        with self.assertRaisesRegex(ValueError, "필수 키가 누락"):
            validate_filter_metadata(collection)

    @patch(
        "scripts.evaluate_embedding_retrieval.create_embedding_model"
    )
    def test_embedding_model_uses_shared_factory(self, create_model):
        create_model.return_value = "embedding"

        result = create_query_embedding_model(
            provider="ollama",
            model_name="bge-m3",
            ollama_base_url="http://ollama:11434",
        )

        create_model.assert_called_once_with(
            provider="ollama",
            model_name="bge-m3",
            base_url="http://ollama:11434",
        )
        self.assertEqual(result, "embedding")

    def test_load_evaluation_cases_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "gold_policy_ids": ["p1"],
                        "user_input": "question",
                        "user_profile": {"age": 25},
                        "hard_negative_ids": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cases = load_evaluation_cases(dataset_path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].gold_policy_ids, ["p1"])

    def test_write_results_creates_summary_and_details(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path, details_path = write_results(
                summary={"metrics": {"mrr": 1.0}},
                details=[{"gold_ranks": {"p1": 1}}],
                output_dir=Path(temp_dir),
                experiment_name="openai/model",
                overwrite=False,
            )

            self.assertTrue(summary_path.exists())
            self.assertTrue(details_path.exists())
            self.assertEqual(
                json.loads(summary_path.read_text())["metrics"]["mrr"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
