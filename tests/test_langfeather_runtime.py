import json
import sys
import types
import urllib.error

from src.langfeather_runtime import (
    FEEDBACK_REASONS,
    LangFeatherRuntime,
    create_langfeather_runtime,
)


class FakeLangFeather:
    def __init__(self):
        self.configure_calls = 0
        self.configure_endpoints = []
        self.wrap_calls = []
        self.shutdown_calls = 0

    def configure(self, endpoint=None):
        self.configure_calls += 1
        self.configure_endpoints.append(endpoint)

    def wrap_runnable(self, graph, *, name):
        self.wrap_calls.append((graph, name))
        return {"wrapped": graph, "name": name}

    def shutdown(self):
        self.shutdown_calls += 1


class FakeResponse:
    def __init__(self, body: dict | None):
        self._body = json.dumps(body).encode("utf-8") if body is not None else b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self._body


class FakeCollector:
    """Emulates the subset of the LangFeather 0.2.0 REST API the runtime uses."""

    def __init__(self):
        self.requests: list[tuple[str, str, dict | None]] = []
        self._score_configs: dict[str, dict] = {}

    def urlopen(self, request, timeout):
        method = request.get_method()
        path = request.full_url.split("4319", 1)[1]
        payload = json.loads(request.data) if request.data else None
        self.requests.append((method, path, payload))

        if method == "POST" and path == "/api/v1/scores":
            if payload["name"] in self._score_configs:
                raise urllib.error.HTTPError(request.full_url, 409, "conflict", None, None)
            options = [
                {"score_option_id": f"so_{option['label']}", **option}
                for option in payload.get("options", [])
            ]
            config = {**payload, "score_config_id": f"sc_{payload['name']}", "options": options}
            self._score_configs[payload["name"]] = config
            return FakeResponse(config)

        if method == "GET" and path == "/api/v1/scores":
            return FakeResponse({"items": list(self._score_configs.values())})

        if method == "PUT" and "/annotations/" in path:
            return FakeResponse({"annotation_id": "an_1"})

        if method == "PUT" and path.endswith("/memo"):
            return FakeResponse({"trace_id": "tr_1"})

        raise AssertionError(f"unexpected request: {method} {path}")


def test_runtime_is_noop_when_langfeather_toggle_is_disabled(monkeypatch):
    monkeypatch.setenv("LANGFEATHER_TRACING", "false")

    runtime = create_langfeather_runtime()
    graph = object()

    assert isinstance(runtime, LangFeatherRuntime)
    assert runtime.sdk is None
    assert runtime.wrap_graph(graph) is graph


def test_runtime_configures_wraps_and_stops_sdk_by_default(monkeypatch):
    fake_sdk = FakeLangFeather()
    fake_module = types.ModuleType("langfeather")
    fake_module.configure = fake_sdk.configure
    fake_module.wrap_runnable = fake_sdk.wrap_runnable
    fake_module.shutdown = fake_sdk.shutdown
    monkeypatch.setitem(sys.modules, "langfeather", fake_module)
    monkeypatch.delenv("LANGFEATHER_TRACING", raising=False)

    runtime = create_langfeather_runtime()
    graph = object()

    assert fake_sdk.configure_calls == 1
    assert runtime.wrap_graph(graph) == {
        "wrapped": graph,
        "name": "youth-policy-rag",
    }
    runtime.shutdown()
    assert fake_sdk.shutdown_calls == 1


def test_runtime_uses_shared_trace_id_and_records_feedback_as_annotations(monkeypatch):
    collector = FakeCollector()
    monkeypatch.setattr("src.langfeather_runtime.urllib.request.urlopen", collector.urlopen)
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

    methods_and_paths = [(m, p) for m, p, _ in collector.requests]
    assert ("POST", "/api/v1/scores") in methods_and_paths
    assert (
        "PUT",
        f"/api/v1/traces/{trace_id}/annotations/sc_user_helpful",
    ) in methods_and_paths
    assert ("PUT", f"/api/v1/traces/{trace_id}/memo") in methods_and_paths

    helpful_put = next(
        p for m, path, p in collector.requests if path.endswith("annotations/sc_user_helpful")
    )
    assert helpful_put == {"value": False}

    reason_config = collector._score_configs["user_feedback_reason"]
    assert [option["label"] for option in reason_config["options"]] == FEEDBACK_REASONS
    reason_put = next(
        p
        for m, path, p in collector.requests
        if path.endswith("annotations/sc_user_feedback_reason")
    )
    assert reason_put == {"value": ["so_missing-details"]}

    memo_put = next(p for m, path, p in collector.requests if path.endswith("/memo"))
    assert memo_put["content"] == "[anon_test] 신청 방법이 더 필요해요."


def test_record_user_feedback_skips_memo_when_no_comment(monkeypatch):
    collector = FakeCollector()
    monkeypatch.setattr("src.langfeather_runtime.urllib.request.urlopen", collector.urlopen)
    monkeypatch.setenv("LANGFEATHER_ENDPOINT", "http://127.0.0.1:4319")
    runtime = LangFeatherRuntime(sdk=object())

    runtime.record_user_feedback(
        trace_id="trace-1",
        helpful=True,
        reason=None,
        comment=None,
        anonymous_user_id="anon_test",
    )

    assert not any(path.endswith("/memo") for _, path, _ in collector.requests)


def test_record_user_feedback_reuses_existing_score_config_id_after_409(monkeypatch):
    collector = FakeCollector()
    collector._score_configs["user_helpful"] = {
        "score_config_id": "sc_existing_helpful",
        "name": "user_helpful",
        "options": [],
    }
    collector._score_configs["user_feedback_reason"] = {
        "score_config_id": "sc_existing_reason",
        "name": "user_feedback_reason",
        "options": [{"score_option_id": f"so_{r}", "label": r} for r in FEEDBACK_REASONS],
    }
    monkeypatch.setattr("src.langfeather_runtime.urllib.request.urlopen", collector.urlopen)
    monkeypatch.setenv("LANGFEATHER_ENDPOINT", "http://127.0.0.1:4319")
    runtime = LangFeatherRuntime(sdk=object())

    runtime.record_user_feedback(
        trace_id="trace-1",
        helpful=True,
        reason=None,
        comment=None,
        anonymous_user_id="anon_test",
    )

    assert runtime._helpful_score_config_id == "sc_existing_helpful"
    assert (
        "PUT",
        "/api/v1/traces/trace-1/annotations/sc_existing_helpful",
    ) in [(m, p) for m, p, _ in collector.requests]
