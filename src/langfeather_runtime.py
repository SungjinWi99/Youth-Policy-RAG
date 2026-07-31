"""LangFeather lifecycle integration for the FastAPI app."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


FALSE_VALUES = {"0", "false", "no", "off"}
DEFAULT_ENDPOINT = "http://127.0.0.1:4319"

# LangFeather 0.2.0 records feedback as score annotations on a trace rather
# than a free-form feedback API. These score configs are created lazily on
# first use and reused by name across process restarts.
HELPFUL_SCORE_NAME = "user_helpful"
REASON_SCORE_NAME = "user_feedback_reason"
FEEDBACK_REASONS = [
    "policy-mismatch",
    "outdated-information",
    "missing-details",
    "unclear-answer",
    "other",
]


def langfeather_tracing_enabled() -> bool:
    """Return whether this process should emit LangFeather traces.

    Tracing is on by default; set LANGFEATHER_TRACING=false to opt out
    (e.g. for tests or environments without a reachable collector).
    """
    return str(os.getenv("LANGFEATHER_TRACING", "")).strip().lower() not in FALSE_VALUES


def _collector_request(
    endpoint: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{endpoint}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=0.5) as response:
        body = response.read()
        return json.loads(body) if body else None


@dataclass
class LangFeatherRuntime:
    """Own the optional SDK sender without coupling graph code to FastAPI."""

    sdk: Any | None = None
    _helpful_score_config_id: str | None = field(default=None, init=False, repr=False)
    _reason_score_config_id: str | None = field(default=None, init=False, repr=False)
    _reason_option_ids: dict[str, str] | None = field(default=None, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return self.sdk is not None

    def create_trace_id(self) -> str | None:
        return uuid4().hex if self.sdk is not None else None

    def trace_metadata(self, trace_id: str | None) -> dict[str, str]:
        if self.sdk is None or trace_id is None:
            return {}
        return {"langfeather_trace_id": trace_id}

    def wrap_graph(self, graph: Any) -> Any:
        if self.sdk is None:
            return graph
        return self.sdk.wrap_runnable(graph, name="youth-policy-rag")

    def shutdown(self) -> None:
        if self.sdk is not None:
            self.sdk.shutdown()

    def _endpoint(self) -> str:
        return os.getenv("LANGFEATHER_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")

    def _get_or_create_score_config(
        self, endpoint: str, name: str, **fields: Any
    ) -> dict[str, Any]:
        try:
            return _collector_request(
                endpoint, "POST", "/api/v1/scores", {"name": name, **fields}
            )
        except urllib.error.HTTPError as error:
            if error.code != 409:
                raise
        configs = _collector_request(endpoint, "GET", "/api/v1/scores")["items"]
        for config in configs:
            if config["name"] == name:
                return config
        raise RuntimeError(f"LangFeather score config '{name}'을(를) 찾을 수 없습니다.")

    def _ensure_score_configs(self, endpoint: str) -> None:
        if self._helpful_score_config_id is None:
            config = self._get_or_create_score_config(
                endpoint, HELPFUL_SCORE_NAME, data_type="boolean"
            )
            self._helpful_score_config_id = config["score_config_id"]

        if self._reason_option_ids is None:
            config = self._get_or_create_score_config(
                endpoint,
                REASON_SCORE_NAME,
                data_type="categorical",
                categorical_selection_mode="single",
                options=[{"label": reason} for reason in FEEDBACK_REASONS],
            )
            self._reason_score_config_id = config["score_config_id"]
            self._reason_option_ids = {
                option["label"]: option["score_option_id"] for option in config["options"]
            }

    def record_user_feedback(
        self,
        *,
        trace_id: str,
        helpful: bool,
        reason: str | None,
        comment: str | None,
        anonymous_user_id: str,
    ) -> None:
        if self.sdk is None:
            raise RuntimeError("LangFeather 피드백 수집이 비활성화되어 있습니다.")

        endpoint = self._endpoint()
        try:
            self._ensure_score_configs(endpoint)

            _collector_request(
                endpoint,
                "PUT",
                f"/api/v1/traces/{trace_id}/annotations/{self._helpful_score_config_id}",
                {"value": helpful},
            )

            if reason is not None:
                option_id = self._reason_option_ids.get(reason)
                if option_id is not None:
                    _collector_request(
                        endpoint,
                        "PUT",
                        f"/api/v1/traces/{trace_id}/annotations/{self._reason_score_config_id}",
                        {"value": [option_id]},
                    )

            # The trace memo is a single shared, human-editable field in the
            # LangFeather UI (not per-feedback storage) — only touch it when
            # there's an actual comment to add, so we don't clobber a
            # reviewer's note or write noise for every plain thumbs-up.
            if comment:
                _collector_request(
                    endpoint,
                    "PUT",
                    f"/api/v1/traces/{trace_id}/memo",
                    {"content": f"[{anonymous_user_id}] {comment}"},
                )
        except OSError as error:
            raise RuntimeError("LangFeather 피드백 저장에 실패했습니다.") from error


def create_langfeather_runtime() -> LangFeatherRuntime:
    """Configure the LangFeather SDK unless tracing is explicitly disabled."""
    if not langfeather_tracing_enabled():
        return LangFeatherRuntime()

    try:
        import langfeather
    except ImportError as error:
        raise RuntimeError(
            "langfeather를 import할 수 없습니다. langfeather[langchain] 의존성이 "
            "설치되어 있는지 확인하세요."
        ) from error

    langfeather.configure(endpoint=os.getenv("LANGFEATHER_ENDPOINT", DEFAULT_ENDPOINT))
    return LangFeatherRuntime(sdk=langfeather)
