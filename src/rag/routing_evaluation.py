from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

from src.rag.router import ContextRouter, RouteName


class RoutingEvaluationDocument(BaseModel):
    page_content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_document(self) -> Document:
        return Document(
            page_content=self.page_content,
            metadata=self.metadata,
        )


class RoutingEvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    current_question: str = Field(min_length=1)
    documents: list[RoutingEvaluationDocument]
    expected_route: RouteName
    rationale: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def reject_blank_tags(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("tags에는 빈 문자열을 사용할 수 없습니다.")
        return values

    def to_documents(self) -> list[Document]:
        return [document.to_document() for document in self.documents]


def load_routing_evaluation_cases(
    path: Path,
) -> list[RoutingEvaluationCase]:
    cases = []
    case_ids = set()
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                case = RoutingEvaluationCase.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} 평가 데이터가 유효하지 않습니다."
                ) from error
            if case.case_id in case_ids:
                raise ValueError(
                    f"{path}:{line_number} case_id가 중복됩니다: "
                    f"{case.case_id}"
                )
            case_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}에 평가 데이터가 없습니다.")
    return cases


def evaluate_routing(
    router: ContextRouter,
    cases: list[RoutingEvaluationCase],
) -> dict[str, Any]:
    expected_counts = Counter(case.expected_route for case in cases)
    correct_counts: Counter[str] = Counter()
    results = []

    for case in cases:
        try:
            decision = router.decide(
                current_question=case.current_question,
                documents=case.to_documents(),
            )
            actual_route = decision.route
            reason = decision.reason
            error = None
        except Exception as exception:
            actual_route = None
            reason = None
            error = f"{type(exception).__name__}: {exception}"

        correct = actual_route == case.expected_route
        if correct:
            correct_counts[case.expected_route] += 1
        results.append({
            "case_id": case.case_id,
            "expected_route": case.expected_route,
            "actual_route": actual_route,
            "correct": correct,
            "reason": reason,
            "error": error,
            "rationale": case.rationale,
            "tags": case.tags,
        })

    correct_total = sum(correct_counts.values())
    per_route = {
        route: {
            "correct": correct_counts[route],
            "total": expected_counts[route],
            "accuracy": (
                correct_counts[route] / expected_counts[route]
                if expected_counts[route]
                else 0.0
            ),
        }
        for route in ("reuse", "search", "clarify")
    }
    return {
        "total": len(cases),
        "correct": correct_total,
        "accuracy": correct_total / len(cases) if cases else 0.0,
        "error_count": sum(
            result["error"] is not None
            for result in results
        ),
        "per_route": per_route,
        "results": results,
    }
