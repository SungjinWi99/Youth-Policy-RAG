import statistics
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from tqdm import tqdm

from src.evaluation.metrics import reciprocal_rank, recall_at_k
from src.evaluation.models import EvaluationCase


def item_value(item: Any, key: str, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def stable_dataset_item_id(
    dataset_name: str,
    case_id: str,
    *,
    namespace: str,
) -> str:
    namespace_prefix = f"{namespace}:" if namespace else ""
    return str(uuid5(
        NAMESPACE_URL,
        f"langfuse:{namespace_prefix}{dataset_name}:{case_id}",
    ))


def is_not_found_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    response_status_code = getattr(getattr(error, "response", None), "status_code", None)
    message = str(error).lower()
    return (
        status_code == 404
        or response_status_code == 404
        or "not found" in message
        or "404" in message
    )


def ensure_retrieval_dataset(
    langfuse,
    *,
    dataset_name: str,
    dataset_path: Path,
    cases: list[EvaluationCase],
):
    try:
        langfuse.get_dataset(dataset_name)
    except Exception as error:
        if not is_not_found_error(error):
            raise
        langfuse.create_dataset(
            name=dataset_name,
            description="청년정책 PolicyRetriever 평가 데이터셋",
            metadata={"source_path": str(dataset_path), "example_count": len(cases)},
        )

    for case in tqdm(cases, desc="Langfuse retrieval dataset sync"):
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=stable_dataset_item_id(
                dataset_name,
                case.case_id,
                namespace="retrieval",
            ),
            input={
                "user_input": case.user_input,
                "user_profile": case.user_profile,
                "exclude_expired": case.exclude_expired,
            },
            expected_output={"expected_policy_ids": case.expected_policy_ids},
            metadata={
                **case.metadata,
                "case_id": case.case_id,
                "source_path": str(dataset_path),
            },
        )
    langfuse.flush()
    return langfuse.get_dataset(dataset_name)


def _retrieved_ids(output: Any) -> list[str]:
    return list(output.get("retrieved_policy_ids") or []) if isinstance(output, dict) else []


def _expected_ids(expected_output: Any) -> list[str]:
    return (
        list(expected_output.get("expected_policy_ids") or [])
        if isinstance(expected_output, dict)
        else []
    )


def build_recall_evaluator(k: int):
    def evaluator(*, output, expected_output, **kwargs):
        from langfuse import Evaluation

        return Evaluation(
            name=f"recall_at_{k}",
            value=recall_at_k(_retrieved_ids(output), _expected_ids(expected_output), k),
        )

    evaluator.__name__ = f"recall_at_{k}_evaluator"
    return evaluator


def reciprocal_rank_evaluator(*, output, expected_output, **kwargs):
    from langfuse import Evaluation

    return Evaluation(
        name="reciprocal_rank",
        value=reciprocal_rank(_retrieved_ids(output), _expected_ids(expected_output)),
    )


def planner_retriever_route_evaluator(*, output, **kwargs):
    from langfuse import Evaluation

    return Evaluation(
        name="planner_retriever_route",
        value=1.0 if output.get("planner_route") == "retriever" else 0.0,
    )


def planner_raw_fallback_evaluator(*, output, **kwargs):
    from langfuse import Evaluation

    return Evaluation(
        name="planner_raw_fallback",
        value=1.0 if output.get("used_raw_fallback") else 0.0,
    )


def reranker_latency_evaluator(*, output, **kwargs):
    from langfuse import Evaluation

    return Evaluation(
        name="reranker_latency_ms",
        value=float(output.get("reranker_latency_ms", 0.0)),
    )


def build_mean_run_evaluator(*, item_metric_name: str, run_metric_name: str):
    def evaluator(*, item_results, **kwargs):
        from langfuse import Evaluation

        values = [
            evaluation.value
            for item_result in item_results
            for evaluation in item_result.evaluations
            if evaluation.name == item_metric_name
            and isinstance(evaluation.value, (int, float))
        ]
        return Evaluation(
            name=run_metric_name,
            value=statistics.mean(values) if values else 0.0,
        )

    evaluator.__name__ = f"{run_metric_name}_run_evaluator"
    return evaluator
