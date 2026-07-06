import json
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from src.config import load_config
from src.policy.schemas import PolicyDetail


@lru_cache(maxsize=1)
def _load_policy_index(
    source_path: str,
    _modified_at_ns: int,
) -> dict[str, PolicyDetail]:
    with Path(source_path).open(encoding="utf-8") as file:
        raw_policies = json.load(file)

    if not isinstance(raw_policies, list):
        raise ValueError("정책 원본 데이터는 JSON 배열이어야 합니다.")

    policy_index: dict[str, PolicyDetail] = {}
    for raw_policy in raw_policies:
        policy = PolicyDetail.model_validate(raw_policy)
        if policy.plcyNo in policy_index:
            raise ValueError(f"중복된 정책 ID입니다: {policy.plcyNo}")
        policy_index[policy.plcyNo] = policy

    return policy_index


class Policy:
    @classmethod
    def get(
        cls,
        policy_id: str,
        source_path: str | Path | None = None,
    ) -> PolicyDetail:
        if source_path is None:
            config = load_config()
            source_path = config.path(config.data.raw)

        resolved_path = Path(source_path).resolve()
        policy_index = _load_policy_index(
            str(resolved_path),
            resolved_path.stat().st_mtime_ns,
        )
        policy = policy_index.get(policy_id)
        if policy is None:
            raise HTTPException(
                status_code=404,
                detail="정책을 찾을 수 없습니다.",
            )
        return policy
