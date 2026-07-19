from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvaluationCase(BaseModel):
    """Canonical JSONL schema shared by retrieval experiments."""

    case_id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    user_profile: dict[str, Any] = Field(default_factory=dict)
    expected_policy_ids: list[str] = Field(min_length=1)
    exclude_expired: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_policy_ids")
    @classmethod
    def reject_blank_or_duplicate_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("정책 ID 목록에 빈 문자열을 사용할 수 없습니다.")
        if len(values) != len(set(values)):
            raise ValueError("정책 ID 목록에 중복을 사용할 수 없습니다.")
        return values

    @property
    def gold_policy_ids(self) -> list[str]:
        return self.expected_policy_ids


class PlannerQueryRecord(BaseModel):
    case_id: str = Field(min_length=1)
    raw_query: str = Field(min_length=1)
    user_profile: dict[str, Any] = Field(default_factory=dict)
    planner_route: str
    answer_strategy: str
    retrieval_queries: list[str] = Field(default_factory=list)
    route_reason: str
    planner_provider: str
    planner_model: str
    planner_prompt_sha256: str
    generated_at: str
