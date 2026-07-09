import os
from collections.abc import Sequence
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def langfuse_tracing_enabled() -> bool:
    return (
        _is_enabled(os.getenv("LANGFUSE_TRACING"))
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


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
    if not langfuse_tracing_enabled():
        return

    from langfuse import get_client

    get_client().flush()


def shutdown_langfuse() -> None:
    if not langfuse_tracing_enabled():
        return

    from langfuse import get_client

    get_client().shutdown()
