from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def policy_id(policy: dict[str, Any]) -> str:
    return str(policy.get("plcyNo") or "").strip()


def validate_policies(
    policies: Any,
    *,
    source: str,
    require_non_empty: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(policies, list):
        raise ValueError(f"{source} 정책 목록이 JSON 배열이 아닙니다.")
    if require_non_empty and not policies:
        raise ValueError(f"{source}에는 비어 있지 않은 JSON 배열이 필요합니다.")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(policies):
        if not isinstance(item, dict):
            raise ValueError(f"{source}[{index}] 정책이 JSON 객체가 아닙니다.")
        item_id = policy_id(item)
        if not item_id:
            raise ValueError(f"{source}[{index}]에 plcyNo가 없습니다.")
        if item_id in seen:
            raise ValueError(f"{source}에 중복 plcyNo가 있습니다: {item_id}")
        seen.add(item_id)
        validated.append(item)
    return validated


def load_policy_snapshot(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as policy_file:
        policies = json.load(policy_file)
    return validate_policies(
        policies,
        source=str(path),
        require_non_empty=True,
    )


def find_new_policies(
    existing: list[dict[str, Any]],
    fetched: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_ids = {policy_id(item) for item in existing}
    return [item for item in fetched if policy_id(item) not in existing_ids]


def write_policy_snapshot_atomically(
    path: Path,
    policies: list[dict[str, Any]],
) -> None:
    validated = validate_policies(
        policies,
        source="저장할 정책",
        require_non_empty=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(validated, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
