import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from src.config import EvaluationConfig, LLMConfig
from src.factory import (
    CHAT_MODEL_CLASSES,
    EMBEDDING_MODEL_CLASSES,
    build_rag_graph,
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

    def test_summary_trigger_ratio_must_be_between_zero_and_one(self):
        with self.assertRaises(ValidationError):
            LLMConfig(
                provider="upstage",
                model="model",
                max_input_tokens=32768,
                summary_trigger_ratio=1.0,
                summary_keep_recent_turns=3,
                token_chars_per_token=2.0,
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

    @patch("src.factory.RAGGraph")
    @patch("src.factory.AnswerGenerator")
    @patch("src.factory.ConversationSummarizer")
    @patch("src.factory.PolicyRetriever")
    @patch("src.factory.create_sqlite_checkpointer")
    @patch("src.factory.Chroma")
    @patch("src.factory.create_chat_model")
    @patch("src.factory.create_embedding_model")
    def test_build_rag_graph_uses_query_embedding(
        self,
        create_embedding,
        create_chat,
        chroma,
        create_checkpointer,
        policy_retriever,
        conversation_summarizer,
        answer_generator,
        rag_graph,
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
                max_input_tokens=32768,
                summary_trigger_ratio=0.65,
                summary_keep_recent_turns=3,
                token_chars_per_token=2.0,
            ),
            data=SimpleNamespace(
                chroma_collection_name="policies",
                chroma_dir="data/chroma",
                conversation_db="data/sqlite/conversations.db",
            ),
            path=lambda value: f"/project/{value}",
        )
        create_embedding.return_value = "embedding-instance"
        create_chat.return_value = "chat-instance"
        chroma.return_value = "vector-store"
        create_checkpointer.return_value = "checkpointer"
        policy_retriever.return_value = "retriever"
        conversation_summarizer.return_value = "summarizer"
        answer_generator.return_value = "generator"

        result = build_rag_graph(config)

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
        create_checkpointer.assert_called_once_with(
            "/project/data/sqlite/conversations.db"
        )
        policy_retriever.assert_called_once_with(
            vector_store="vector-store",
            search_k=3,
        )
        conversation_summarizer.assert_called_once_with(
            "chat-instance",
            max_input_tokens=32768,
            summary_trigger_ratio=0.65,
            keep_recent_turns=3,
            chars_per_token=2.0,
        )
        answer_generator.assert_called_once_with("chat-instance")
        rag_graph.assert_called_once_with(
            retriever="retriever",
            summarizer="summarizer",
            generator="generator",
            checkpointer="checkpointer",
        )
        self.assertEqual(result, rag_graph.return_value)


if __name__ == "__main__":
    unittest.main()
