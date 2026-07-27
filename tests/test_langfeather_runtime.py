import json
import sys
import types
from datetime import datetime

from src.langfeather_runtime import (
    LangFeatherRuntime,
    create_langfeather_runtime,
)


class FakeLangFeather:
    def __init__(self):
        self.configure_calls = 0
        self.wrap_calls = []
        self.shutdown_calls = 0

    def configure(self):
        self.configure_calls += 1

    def wrap_runnable(self, graph, *, name):
        self.wrap_calls.append((graph, name))
        return {"wrapped": graph, "name": name}

    def shutdown(self):
        self.shutdown_calls += 1


class FakeResponse:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_runtime_is_noop_when_langfeather_toggle_is_disabled(monkeypatch):
    monkeypatch.delenv("LANGFEATHER_TRACING", raising=False)

    runtime = create_langfeather_runtime()
    graph = object()

    assert isinstance(runtime, LangFeatherRuntime)
    assert runtime.sdk is None
    assert runtime.wrap_graph(graph) is graph


def test_runtime_configures_wraps_and_stops_optional_sdk(monkeypatch):
    fake_sdk = FakeLangFeather()
    fake_module = types.ModuleType("langfeather")
    fake_module.configure = fake_sdk.configure
    fake_module.wrap_runnable = fake_sdk.wrap_runnable
    fake_module.shutdown = fake_sdk.shutdown
    monkeypatch.setitem(sys.modules, "langfeather", fake_module)
    monkeypatch.setenv("LANGFEATHER_TRACING", "true")

    runtime = create_langfeather_runtime()
    graph = object()

    assert fake_sdk.configure_calls == 1
    assert runtime.wrap_graph(graph) == {
        "wrapped": graph,
        "name": "youth-policy-rag",
    }
    runtime.shutdown()
    assert fake_sdk.shutdown_calls == 1


def test_runtime_uses_shared_trace_id_and_posts_feedback(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = request.data
        return FakeResponse()

    monkeypatch.setattr("src.langfeather_runtime.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("LANGFEATHER_ENDPOINT", "http://127.0.0.1:4319")
    runtime = LangFeatherRuntime(sdk=object())

    trace_id = runtime.create_trace_id()
    assert trace_id is not None
    assert runtime.trace_metadata(trace_id) == {"langfeather_trace_id": trace_id}
    runtime.record_user_feedback(
        trace_id=trace_id,
        helpful=False,
        reason="missing-details",
        comment="신청 방법이 더 필요해요.",
        anonymous_user_id="anon_test",
    )

    assert captured["url"] == "http://127.0.0.1:4319/api/v1/feedback"
    assert captured["timeout"] == 0.5
    payload = json.loads(captured["payload"])
    assert payload["trace_id"] == trace_id
    assert payload["value"] is False
    assert payload["metadata"]["reason"] == "missing-details"
    assert payload["metadata"]["anonymous_user_id"] == "anon_test"
    assert datetime.fromisoformat(payload["created_at"]) == datetime.fromisoformat(
        payload["updated_at"]
    )
