import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ingest_chroma import (
    build_documents,
    create_passage_embedding_model,
    parse_args,
    prepare_vector_store,
    to_policy_end_date,
)


class IngestChromaTest(unittest.TestCase):
    def test_build_documents_preserves_filter_metadata(self):
        policies = [
            {
                "plcyNo": "policy-1",
                "plcyNm": "주거 지원",
                "lclsfNm": "주거",
                "mclsfNm": "주택",
                "plcyExplnCn": "주거비를 지원합니다.",
                "plcySprtCn": "월 20만원",
                "sprtTrgtMinAge": "19",
                "sprtTrgtMaxAge": "39",
                "earnMinAmt": "0",
                "earnMaxAmt": "0",
                "aplyYmd": "20260101 ~ 20261231",
                "zipCd": "11110",
            }
        ]

        documents, ids = build_documents(policies)

        self.assertEqual(ids, ["policy-1"])
        self.assertIn("정책명: 주거 지원", documents[0].page_content)
        self.assertEqual(documents[0].metadata["agePolicy"], "specific")
        self.assertEqual(documents[0].metadata["incomePolicy"], "all")
        self.assertEqual(
            documents[0].metadata["applicationPolicy"],
            "fixed",
        )
        self.assertTrue(documents[0].metadata["region_11"])
        self.assertFalse(documents[0].metadata["region_26"])

    @patch("scripts.ingest_chroma.create_embedding_model")
    def test_ollama_embedding_uses_base_url(self, create_model):
        create_model.return_value = "embedding"

        result = create_passage_embedding_model(
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

    def test_parse_args_is_config_independent(self):
        args = parse_args(
            [
                "--provider",
                "openai",
                "--model",
                "text-embedding-3-large",
                "--chroma-dir",
                "data/chroma_openai",
            ]
        )

        self.assertEqual(args.provider, "openai")
        self.assertEqual(args.model, "text-embedding-3-large")
        self.assertEqual(args.chroma_dir, Path("data/chroma_openai"))
        self.assertEqual(args.distance_metric, "cosine")

    def test_existing_collection_requires_recreate(self):
        fake_client = unittest.mock.Mock()
        fake_client.list_collections.return_value = [
            unittest.mock.Mock(name="ignored")
        ]
        fake_client.list_collections.return_value[0].name = "policies"

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "scripts.ingest_chroma.chromadb.PersistentClient",
                return_value=fake_client,
            ):
                with self.assertRaisesRegex(
                    FileExistsError,
                    "--recreate",
                ):
                    prepare_vector_store(
                        chroma_dir=Path(temp_dir),
                        collection_name="policies",
                        embedding_model=object(),
                        distance_metric="cosine",
                        recreate=False,
                    )

    def test_policy_end_date_uses_open_ended_sentinel(self):
        self.assertEqual(to_policy_end_date(""), 99991231)
        self.assertEqual(to_policy_end_date("20261231"), 20261231)


if __name__ == "__main__":
    unittest.main()
