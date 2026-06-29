import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from src.config import EvaluationConfig
from src.factory import (
    CHAT_MODEL_CLASSES,
    EMBEDDING_MODEL_CLASSES,
    build_rag_pipeline,
    create_chat_model,
    create_embedding_model,
)


class FakeModel:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


class FactoryTest(unittest.TestCase):
    def test_evaluation_concurrency_must_be_positive(self):
        with self.assertRaises(ValidationError):
            EvaluationConfig(
                example_path="data/eval/examples.jsonl",
                provider="upstage",
                model="judge-model",
                dataset_name="dataset",
                experiment_prefix="experiment",
                max_concurrency=0,
            )

    def test_create_chat_model_passes_model_and_options(self):
        with patch.dict(CHAT_MODEL_CLASSES, {"fake": FakeModel}):
            model = create_chat_model(
                provider="fake",
                model_name="judge-model",
                temperature=0,
            )

        self.assertEqual(model.model, "judge-model")
        self.assertEqual(model.kwargs, {"temperature": 0})

    def test_create_embedding_model_passes_model_name(self):
        with patch.dict(EMBEDDING_MODEL_CLASSES, {"fake": FakeModel}):
            model = create_embedding_model(
                provider="fake",
                model_name="passage-model",
            )

        self.assertEqual(model.model, "passage-model")

    def test_unknown_provider_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            create_chat_model("unknown", "model")

        with self.assertRaisesRegex(ValueError, "unknown"):
            create_embedding_model("unknown", "model")

    @patch("src.factory.RAGPipeline")
    @patch("src.factory.Chroma")
    @patch("src.factory.create_chat_model")
    @patch("src.factory.create_embedding_model")
    def test_build_rag_pipeline_uses_query_embedding(
        self,
        create_embedding,
        create_chat,
        chroma,
        rag_pipeline,
    ):
        config = SimpleNamespace(
            retriever=SimpleNamespace(
                provider="upstage",
                query_model="query-model",
                passage_model="passage-model",
                search_k=3,
            ),
            llm=SimpleNamespace(
                provider="google",
                model="generator-model",
            ),
            data=SimpleNamespace(
                chroma_collection_name="policies",
                chroma_dir="data/chroma",
            ),
            path=lambda value: f"/project/{value}",
        )
        create_embedding.return_value = "embedding-instance"
        create_chat.return_value = "chat-instance"
        chroma.return_value = "vector-store"

        result = build_rag_pipeline(config)

        create_embedding.assert_called_once_with(
            provider="upstage",
            model_name="query-model",
        )
        create_chat.assert_called_once_with(
            provider="google",
            model_name="generator-model",
        )
        chroma.assert_called_once_with(
            collection_name="policies",
            persist_directory="/project/data/chroma",
            embedding_function="embedding-instance",
        )
        rag_pipeline.assert_called_once_with(
            llm="chat-instance",
            vector_store="vector-store",
            search_k=3,
        )
        self.assertEqual(result, rag_pipeline.return_value)


if __name__ == "__main__":
    unittest.main()
