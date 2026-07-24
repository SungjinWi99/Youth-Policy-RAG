import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config import AppConfig


TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def langfuse_tracing_enabled() -> bool:
    return (
        _is_enabled(os.getenv("LANGFUSE_TRACING"))
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


@dataclass
class ObservabilityRuntime:
    """Own the process-level observability client and its lifecycle."""

    client: Any | None = None

    def build_trace_config(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.client is None:
            return {}

        try:
            from langfuse.langchain import CallbackHandler
        except ImportError as error:
            raise RuntimeError(
                "LANGFUSE_TRACING=true 이지만 langfuse 패키지를 import할 수 없습니다. "
                "uv sync로 의존성을 설치한 뒤 다시 실행하세요."
            ) from error

        langfuse_metadata: dict[str, Any] = {}
        if user_id:
            langfuse_metadata["langfuse_user_id"] = user_id
        if session_id:
            langfuse_metadata["langfuse_session_id"] = session_id
        if tags:
            langfuse_metadata["langfuse_tags"] = list(tags)
        if metadata:
            langfuse_metadata.update({
                key: value
                for key, value in metadata.items()
                if value is not None
            })

        callback_kwargs = (
            {"trace_context": {"trace_id": trace_id}}
            if trace_id
            else {}
        )
        config: dict[str, Any] = {
            "callbacks": [CallbackHandler(**callback_kwargs)]
        }
        if langfuse_metadata:
            config["metadata"] = langfuse_metadata
        if tags:
            config["tags"] = list(tags)
        return config

    def create_trace_id(self) -> str | None:
        if self.client is None:
            return None
        return self.client.create_trace_id()

    def record_user_feedback(
        self,
        *,
        trace_id: str,
        helpful: bool,
        reason: str | None,
        comment: str | None,
        anonymous_user_id: str,
    ) -> None:
        if self.client is None:
            raise RuntimeError("Langfuse 피드백 수집이 비활성화되어 있습니다.")

        score_metadata = {
            "source": "lan-user-feedback",
            "anonymous_user_id": anonymous_user_id,
        }
        self.client.create_score(
            name="user-thumbs",
            value=1 if helpful else 0,
            trace_id=trace_id,
            score_id=self.client.create_trace_id(
                seed=f"{trace_id}:user-thumbs"
            ),
            data_type="BOOLEAN",
            comment=comment,
            metadata=score_metadata,
        )
        if reason:
            self.client.create_score(
                name="user-feedback-reason",
                value=reason,
                trace_id=trace_id,
                score_id=self.client.create_trace_id(
                    seed=f"{trace_id}:user-feedback-reason"
                ),
                data_type="CATEGORICAL",
                metadata=score_metadata,
            )
        self.client.flush()

    def flush(self) -> None:
        if self.client is not None:
            self.client.flush()

    def shutdown(self) -> None:
        if self.client is None:
            return
        client = self.client
        self.client = None
        client.shutdown()


def create_observability_runtime(config: AppConfig) -> ObservabilityRuntime:
    if not langfuse_tracing_enabled():
        return ObservabilityRuntime()

    from langfuse import Langfuse

    return ObservabilityRuntime(
        client=Langfuse(
            release=config.app.release,
            environment=config.app.environment,
        )
    )
