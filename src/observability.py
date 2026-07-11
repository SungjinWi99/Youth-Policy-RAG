import os
from collections.abc import Sequence
from typing import Any

from src.config import AppConfig


TRUE_VALUES = {"1", "true", "yes", "on"}
_langfuse_client: Any | None = None


def _is_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def langfuse_tracing_enabled() -> bool:
    return (
        _is_enabled(os.getenv("LANGFUSE_TRACING"))
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


def initialize_langfuse(config: AppConfig) -> Any | None:
    """Initialize the process-wide Langfuse client before callbacks are built."""
    global _langfuse_client

    if not langfuse_tracing_enabled():
        return None
    if _langfuse_client is not None:
        return _langfuse_client

    from langfuse import Langfuse

    _langfuse_client = Langfuse(
        release=config.app.release,
        environment=config.app.environment,
    )
    return _langfuse_client


def build_langfuse_config(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not langfuse_tracing_enabled():
        return {}

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as error:
        raise RuntimeError(
            "LANGFUSE_TRACING=true 이지만 langfuse 패키지를 import할 수 없습니다. "
            "requirements.txt를 설치한 뒤 다시 실행하세요."
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

    config: dict[str, Any] = {"callbacks": [CallbackHandler()]}
    if langfuse_metadata:
        config["metadata"] = langfuse_metadata
    if tags:
        config["tags"] = list(tags)
    return config


def flush_langfuse() -> None:
    if _langfuse_client is None:
        return
    _langfuse_client.flush()


def shutdown_langfuse() -> None:
    global _langfuse_client

    if _langfuse_client is None:
        return
    _langfuse_client.shutdown()
    _langfuse_client = None
