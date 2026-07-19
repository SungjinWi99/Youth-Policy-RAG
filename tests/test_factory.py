import pytest
from types import SimpleNamespace

import src.factory as factory
from src.config import load_config
from src.factory import create_chat_model
from src.rag.nodes.retriever import EnsemblePolicyRetriever


@pytest.fixture(autouse=True)
def deepseek_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")


def test_create_deepseek_chat_model_disables_thinking_mode_by_default():
    model = create_chat_model(
        provider="deepseek",
        model_name="deepseek-v4-flash",
    )

    assert model.extra_body == {
        "thinking": {
            "type": "disabled",
        }
    }


def test_create_deepseek_chat_model_preserves_explicit_extra_body():
    model = create_chat_model(
        provider="deepseek",
        model_name="deepseek-v4-flash",
        extra_body={
            "thinking": {
                "type": "enabled",
            },
            "custom": "value",
        },
    )

    assert model.extra_body == {
        "thinking": {
            "type": "enabled",
        },
        "custom": "value",
    }


def test_build_rag_graph_constructs_configured_ensemble(monkeypatch):
    config = load_config().model_copy(deep=True)
    vector_store = SimpleNamespace()

    monkeypatch.setattr(factory, "create_embedding_model", lambda **kwargs: object())
    monkeypatch.setattr(factory, "Chroma", lambda **kwargs: vector_store)
    monkeypatch.setattr(
        factory,
        "BM25PolicyRetriever",
        lambda collection, search_k: SimpleNamespace(
            collection=collection,
            search_k=search_k,
        ),
    )
    monkeypatch.setattr(factory, "create_chat_model", lambda **kwargs: object())
    monkeypatch.setattr(factory, "PolicyAgent", lambda llm: object())
    monkeypatch.setattr(factory, "TurnPlanner", lambda llm: object())
    monkeypatch.setattr(factory, "create_sqlite_checkpointer", lambda path: object())
    monkeypatch.setattr(
        factory,
        "PolicyRagGraph",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    graph = factory.build_rag_graph(config)

    assert isinstance(graph.retriever, EnsemblePolicyRetriever)
    assert graph.retriever.weights == [0.65, 0.35]
    assert graph.retriever.search_k == 3
    assert graph.retriever.rrf_k == 1
    assert [
        retriever.search_k for retriever in graph.retriever.retrievers
    ] == [10, 50]
