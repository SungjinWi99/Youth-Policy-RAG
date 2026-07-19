import statistics
from typing import Any


DEFAULT_K_VALUES = (3, 5, 10)


def recall_at_k(
    retrieved_policy_ids: list[str],
    gold_policy_ids: list[str],
    k: int,
) -> float:
    gold = set(gold_policy_ids)
    if not gold:
        return 0.0
    return len(set(retrieved_policy_ids[:k]) & gold) / len(gold)


def rank_gold_policy_ids(
    retrieved_policy_ids: list[str],
    gold_policy_ids: list[str],
) -> dict[str, int | None]:
    ranks = {
        policy_id: rank
        for rank, policy_id in enumerate(retrieved_policy_ids, start=1)
    }
    return {policy_id: ranks.get(policy_id) for policy_id in gold_policy_ids}


def reciprocal_rank(
    retrieved_policy_ids: list[str],
    gold_policy_ids: list[str],
) -> float:
    ranks = rank_gold_policy_ids(retrieved_policy_ids, gold_policy_ids)
    found_ranks = [rank for rank in ranks.values() if rank is not None]
    return 1.0 / min(found_ranks) if found_ranks else 0.0


def score_ranked_rows(
    rows: list[dict[str, Any]],
    *,
    result_key: str,
    expected_key: str = "expected_policy_ids",
) -> dict[str, float]:
    recalls = {k: [] for k in DEFAULT_K_VALUES}
    reciprocal_ranks = []
    for row in rows:
        retrieved = row[result_key][: max(DEFAULT_K_VALUES)]
        expected = row[expected_key]
        for k in recalls:
            recalls[k].append(recall_at_k(retrieved, expected, k))
        reciprocal_ranks.append(reciprocal_rank(retrieved, expected))
    return {
        **{
            f"recall_at_{k}": statistics.mean(values)
            for k, values in recalls.items()
        },
        "mrr": statistics.mean(reciprocal_ranks),
    }


def rank_movement(
    rows: list[dict[str, Any]],
    *,
    baseline_key: str,
    candidate_key: str,
    expected_key: str = "expected_policy_ids",
) -> dict[str, int]:
    movement = {"improved": 0, "degraded": 0, "same": 0}
    for row in rows:
        expected = row[expected_key]
        baseline_ranks = rank_gold_policy_ids(row[baseline_key], expected)
        candidate_ranks = rank_gold_policy_ids(row[candidate_key], expected)
        baseline_found = [rank for rank in baseline_ranks.values() if rank is not None]
        candidate_found = [rank for rank in candidate_ranks.values() if rank is not None]
        baseline_rank = min(baseline_found) if baseline_found else 10**9
        candidate_rank = min(candidate_found) if candidate_found else 10**9
        key = (
            "improved"
            if candidate_rank < baseline_rank
            else "degraded"
            if candidate_rank > baseline_rank
            else "same"
        )
        movement[key] += 1
    return movement


def mean_or_none(values: list[float | int]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median_or_none(values: list[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None
