import hashlib
import sys
import types

from src.config import load_config
from src.observability import (
    ObservabilityRuntime,
    create_observability_runtime,
)


class FakeLangfuse:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.shutdown_calls = 0
        self.flush_calls = 0
        self.score_calls = []
        self.instances.append(self)

    def create_trace_id(self, seed=None):
        value = seed or "new-trace"
        return hashlib.sha256(value.encode()).hexdigest()[:32]

    def create_score(self, **kwargs):
        self.score_calls.append(kwargs)

    def flush(self):
        self.flush_calls += 1

    def shutdown(self):
        self.shutdown_calls += 1


def test_runtime_owns_langfuse_lifecycle(monkeypatch):
    fake_module = types.ModuleType("langfuse")
    fake_module.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    FakeLangfuse.instances.clear()

    config = load_config()
    runtime = create_observability_runtime(config)

    assert len(FakeLangfuse.instances) == 1
    assert runtime.client.kwargs == {
        "release": config.app.release,
        "environment": config.app.environment,
    }

    client = runtime.client
    runtime.flush()
    runtime.shutdown()
    runtime.shutdown()

    assert client.flush_calls == 1
    assert client.shutdown_calls == 1
    assert runtime.client is None


def test_runtime_is_noop_when_tracing_is_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING", raising=False)

    runtime = create_observability_runtime(load_config())

    assert isinstance(runtime, ObservabilityRuntime)
    assert runtime.client is None
    assert runtime.build_trace_config(user_id="user") == {}


def test_runtime_builds_request_trace_config(monkeypatch):
    class FakeCallbackHandler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    langchain_module = types.ModuleType("langfuse.langchain")
    langchain_module.CallbackHandler = FakeCallbackHandler
    monkeypatch.setitem(sys.modules, "langfuse.langchain", langchain_module)
    runtime = ObservabilityRuntime(client=FakeLangfuse())

    config = runtime.build_trace_config(
        user_id="user-1",
        session_id="session-1",
        trace_id="a" * 32,
        tags=["rag"],
        metadata={"case_id": "case-1"},
    )

    assert isinstance(config["callbacks"][0], FakeCallbackHandler)
    assert config["callbacks"][0].kwargs == {
        "trace_context": {"trace_id": "a" * 32}
    }
    assert config["tags"] == ["rag"]
    assert config["metadata"] == {
        "langfuse_user_id": "user-1",
        "langfuse_session_id": "session-1",
        "langfuse_tags": ["rag"],
        "case_id": "case-1",
    }


def test_runtime_records_trace_linked_user_feedback():
    client = FakeLangfuse()
    runtime = ObservabilityRuntime(client=client)

    runtime.record_user_feedback(
        trace_id="a" * 32,
        helpful=False,
        reason="outdated-information",
        comment="신청 기간이 지났어요.",
        anonymous_user_id="anon-1",
    )

    assert [score["name"] for score in client.score_calls] == [
        "user-thumbs",
        "user-feedback-reason",
    ]
    assert client.score_calls[0]["value"] == 0
    assert client.score_calls[0]["data_type"] == "BOOLEAN"
    assert client.score_calls[1]["value"] == "outdated-information"
    assert client.score_calls[1]["data_type"] == "CATEGORICAL"
    assert all(
        score["trace_id"] == "a" * 32
        for score in client.score_calls
    )
    assert client.flush_calls == 1
