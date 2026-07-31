"""Optional local LangFeather lifecycle integration for the demo app."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_ENDPOINT = "http://127.0.0.1:4319"


def langfeather_tracing_enabled() -> bool:
    """Return whether this process should emit local LangFeather traces."""
    return str(os.getenv("LANGFEATHER_TRACING", "")).strip().lower() in TRUE_VALUES


@dataclass
class LangFeatherRuntime:
    """Own the optional SDK sender without coupling graph code to FastAPI."""

    sdk: Any | None = None

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

    def record_user_feedback(
        self,
        *,
        trace_id: str,
        helpful: bool,
        reason: str | None,
        comment: str | None,
        anonymous_session_id: str,
    ) -> None:
        if self.sdk is None:
            raise RuntimeError("LangFeather 피드백 수집이 비활성화되어 있습니다.")

        timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        endpoint = os.getenv("LANGFEATHER_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
        payload = {
            "feedback_id": f"fb_{uuid4().hex}",
            "trace_id": trace_id,
            "name": "user_feedback",
            "value": helpful,
            "comment": comment,
            "metadata": {
                "source": "youth-policy-rag",
                "reason": reason,
                "anonymous_session_id": anonymous_session_id,
            },
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        request = urllib.request.Request(
            f"{endpoint}/api/v1/feedback",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=0.5) as response:
                if response.status not in (200, 201):
                    raise RuntimeError("LangFeather 피드백 저장에 실패했습니다.")
        except OSError as error:
            raise RuntimeError("LangFeather 피드백 저장에 실패했습니다.") from error


def create_langfeather_runtime() -> LangFeatherRuntime:
    """Configure the local SDK only when the explicit demo toggle is enabled."""
    if not langfeather_tracing_enabled():
        return LangFeatherRuntime()

    try:
        import langfeather
    except ImportError as error:
        raise RuntimeError(
            "LANGFEATHER_TRACING=true 이지만 langfeather를 import할 수 없습니다. "
            "시연에서는 LANGFEATHER_ENDPOINT와 함께 local SDK 경로를 제공하세요."
        ) from error

    langfeather.configure()
    return LangFeatherRuntime(sdk=langfeather)
