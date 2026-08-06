import logging

import pytest
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

import src.factory as factory
from src.config import LLMConfig, LLMProviderConfig, load_config
from src.factory import (
    EMBEDDING_PASSAGE_MODEL_KEY,
    EMBEDDING_PROVIDER_KEY,
    create_chat_model,
    verify_embedding_consistency,
)
from src.rag.retrievers import EnsemblePolicyRetriever


def make_vector_store(provider, passage_model):
    return SimpleNamespace(
        _collection=SimpleNamespace(
            metadata={
                EMBEDDING_PROVIDER_KEY: provider,
                EMBEDDING_PASSAGE_MODEL_KEY: passage_model,
            }
        )
    )


def patch_factory_dependencies(monkeypatch, vector_store):
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
    monkeypatch.setattr(
        factory, "create_chat_model_with_fallback", lambda config: object()
    )
    monkeypatch.setattr(
        factory,
        "make_retrieval_planner_node",
        lambda llm, history_window: SimpleNamespace(
            llm=llm,
            history_window=history_window,
        ),
    )
    monkeypatch.setattr(
        factory,
        "make_retriever_node",
        lambda retriever: retriever,
    )
    monkeypatch.setattr(
        factory,
        "make_policy_checker_node",
        lambda llm: SimpleNamespace(llm=llm),
    )
    monkeypatch.setattr(
        factory,
        "make_policy_selector_node",
        lambda: SimpleNamespace(mode="verdict"),
    )
    monkeypatch.setattr(
        factory,
        "make_answer_generator_node",
        lambda llm, history_window: SimpleNamespace(
            llm=llm,
            history_window=history_window,
        ),
    )
    monkeypatch.setattr(factory, "create_sqlite_checkpointer", lambda path: object())
    monkeypatch.setattr(
        factory,
        "PolicyRagGraph",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )


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
    vector_store = make_vector_store(
        config.retriever.provider,
        config.retriever.passage_model,
    )
    patch_factory_dependencies(monkeypatch, vector_store)

    graph = factory.build_rag_graph(config)

    assert isinstance(graph.retriever, EnsemblePolicyRetriever)
    assert graph.retriever.weights == [0.65, 0.35]
    assert graph.retriever.search_k == 3
    assert graph.retriever.rrf_k == 1
    assert [
        retriever.search_k for retriever in graph.retriever.retrievers
    ] == [10, 50]
    assert graph.policy_selector.mode == "verdict"
    assert graph.max_retrieval_retries == 3
    assert graph.retrieval_planner.history_window == 6
    assert graph.answer_generator.history_window == 10


def test_build_rag_graph_refuses_index_built_with_another_model(monkeypatch):
    config = load_config().model_copy(deep=True)
    vector_store = make_vector_store(
        config.retriever.provider,
        "some-other-passage-model",
    )
    patch_factory_dependencies(monkeypatch, vector_store)

    with pytest.raises(RuntimeError) as error:
        factory.build_rag_graph(config)

    assert "some-other-passage-model" in str(error.value)


def test_build_rag_graph_compares_passage_model_not_query_model(monkeypatch):
    # 적재는 passage 모델로 한다. query_model과 대조하면 정상 인덱스가
    # 항상 불일치로 거부된다.
    config = load_config().model_copy(deep=True)
    vector_store = make_vector_store(
        config.retriever.provider,
        config.retriever.query_model,
    )
    patch_factory_dependencies(monkeypatch, vector_store)

    with pytest.raises(RuntimeError):
        factory.build_rag_graph(config)


def test_same_dimension_different_model_is_rejected():
    # ISSUE-002의 핵심 사고: 차원이 같아 Chroma는 오류를 내지 않지만
    # 검색 결과는 무의미해지는 조합.
    vector_store = make_vector_store("ollama", "qwen3-embedding")

    with pytest.raises(RuntimeError) as error:
        verify_embedding_consistency(
            vector_store,
            provider="upstage",
            passage_model="solar-embedding-1-large-passage",
        )

    message = str(error.value)
    assert "ollama/qwen3-embedding" in message
    assert "upstage/solar-embedding-1-large-passage" in message


def test_legacy_index_without_metadata_warns_and_passes(caplog):
    vector_store = SimpleNamespace(
        _collection=SimpleNamespace(metadata={"hnsw:space": "cosine"})
    )

    with caplog.at_level(logging.WARNING):
        verify_embedding_consistency(
            vector_store,
            provider="upstage",
            passage_model="solar-embedding-1-large-passage",
        )

    # 문구가 아니라 보간된 설정값으로 확인한다. 경고 문구는 바뀔 수 있다.
    assert len(caplog.records) == 1
    assert "upstage" in caplog.text
    assert "solar-embedding-1-large-passage" in caplog.text


def test_unreachable_internal_collection_warns_and_passes(caplog):
    with caplog.at_level(logging.WARNING):
        verify_embedding_consistency(
            SimpleNamespace(),
            provider="upstage",
            passage_model="solar-embedding-1-large-passage",
        )

    # 레거시 인덱스 경고와 달리 설정값을 싣지 않는다 = 다른 경고 경로.
    assert len(caplog.records) == 1
    assert "upstage" not in caplog.text


class _RetryableStub(Exception):
    """RETRYABLE_EXCEPTIONS에 등록되는, timeout/5xx/rate-limit을 흉내낸 예외."""


class _FatalStub(Exception):
    """400 같은, 재시도해봤자 소용없는 예외를 흉내낸다."""


def make_stub_chat_model(fail_times, exception_cls):
    """fail_times번 exception_cls를 던진 뒤 성공하는 가짜 ChatModel.

    실제 Runnable(RunnableLambda)이라 .with_retry()/.with_fallbacks()가
    그대로 적용된다 - 진짜 API를 부르지 않고 재시도/폴백 배선만 검증한다.
    """
    state = {"calls": 0}

    def _invoke(_input):
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exception_cls("stub failure")
        return "ok"

    return RunnableLambda(_invoke), state


def _fallback_config(main_max_attempts=2):
    return LLMConfig(
        providers=[
            LLMProviderConfig(
                provider="deepseek",
                model="main-model",
                max_attempts=main_max_attempts,
                retry_wait_initial=0.01,
                retry_wait_max=0.01,
            ),
            LLMProviderConfig(
                provider="anthropic",
                model="fallback-model",
                max_attempts=1,
                retry_wait_initial=0.01,
                retry_wait_max=0.01,
            ),
        ]
    )


def test_retryable_error_retries_then_falls_back_when_exhausted(monkeypatch):
    monkeypatch.setitem(factory.RETRYABLE_EXCEPTIONS, "deepseek", (_RetryableStub,))
    main_model, main_state = make_stub_chat_model(fail_times=99, exception_cls=_RetryableStub)
    fallback_model, fallback_state = make_stub_chat_model(fail_times=0, exception_cls=_RetryableStub)

    def fake_create_chat_model(provider, model_name, **kwargs):
        return main_model if provider == "deepseek" else fallback_model

    monkeypatch.setattr(factory, "create_chat_model", fake_create_chat_model)

    config = _fallback_config(main_max_attempts=2)
    llm = factory.create_chat_model_with_fallback(config)

    assert llm.invoke("hi") == "ok"
    assert main_state["calls"] == config.main.max_attempts
    assert fallback_state["calls"] == 1


def test_non_retryable_error_skips_retry_and_falls_back_immediately(monkeypatch):
    monkeypatch.setitem(factory.RETRYABLE_EXCEPTIONS, "deepseek", (_RetryableStub,))
    main_model, main_state = make_stub_chat_model(fail_times=99, exception_cls=_FatalStub)
    fallback_model, fallback_state = make_stub_chat_model(fail_times=0, exception_cls=_RetryableStub)

    def fake_create_chat_model(provider, model_name, **kwargs):
        return main_model if provider == "deepseek" else fallback_model

    monkeypatch.setattr(factory, "create_chat_model", fake_create_chat_model)

    config = _fallback_config(main_max_attempts=3)
    llm = factory.create_chat_model_with_fallback(config)

    assert llm.invoke("hi") == "ok"
    assert main_state["calls"] == 1  # retry 없이 한 번만 시도하고 바로 포기
    assert fallback_state["calls"] == 1


def test_all_providers_exhausted_raises(monkeypatch):
    monkeypatch.setitem(factory.RETRYABLE_EXCEPTIONS, "deepseek", (_RetryableStub,))
    monkeypatch.setitem(factory.RETRYABLE_EXCEPTIONS, "anthropic", (_RetryableStub,))
    main_model, main_state = make_stub_chat_model(fail_times=99, exception_cls=_RetryableStub)
    fallback_model, fallback_state = make_stub_chat_model(fail_times=99, exception_cls=_RetryableStub)

    def fake_create_chat_model(provider, model_name, **kwargs):
        return main_model if provider == "deepseek" else fallback_model

    monkeypatch.setattr(factory, "create_chat_model", fake_create_chat_model)

    config = _fallback_config(main_max_attempts=1)
    llm = factory.create_chat_model_with_fallback(config)

    with pytest.raises(_RetryableStub):
        llm.invoke("hi")

    assert main_state["calls"] == 1
    assert fallback_state["calls"] == config.fallbacks[0].max_attempts
