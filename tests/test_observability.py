import sys
import types

from src.config import load_config
import src.observability as observability


class FakeLangfuse:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.shutdown_calls = 0
        self.flush_calls = 0
        self.instances.append(self)

    def flush(self):
        self.flush_calls += 1

    def shutdown(self):
        self.shutdown_calls += 1


def test_initialize_langfuse_uses_app_release_and_environment(monkeypatch):
    fake_module = types.ModuleType("langfuse")
    fake_module.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(observability, "_langfuse_client", None)
    FakeLangfuse.instances.clear()

    config = load_config()
    first = observability.initialize_langfuse(config)
    second = observability.initialize_langfuse(config)

    assert first is second
    assert len(FakeLangfuse.instances) == 1
    assert first.kwargs == {
        "release": config.app.release,
        "environment": config.app.environment,
    }

    observability.flush_langfuse()
    observability.shutdown_langfuse()

    assert first.flush_calls == 1
    assert first.shutdown_calls == 1
    assert observability._langfuse_client is None


def test_initialize_langfuse_is_noop_when_tracing_is_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING", raising=False)
    monkeypatch.setattr(observability, "_langfuse_client", None)

    assert observability.initialize_langfuse(load_config()) is None
