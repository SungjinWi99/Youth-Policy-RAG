import json
from datetime import date
from pathlib import Path
from typing import Any

from src.evaluation.metrics import rank_movement, score_ranked_rows
from src.evaluation.models import EvaluationCase, PlannerQueryRecord
from src.rag.retrievers import (
    BM25PolicyRetriever,
    EnsemblePolicyRetriever,
    RetrievalRequest,
    tokenize_korean_legacy,
    tokenize_korean_lexical,
)


BM25_TOKENIZERS = {
    "kiwi": tokenize_korean_lexical,
    "legacy": tokenize_korean_legacy,
}


class CachedPolicyRetriever:
    """Adapter used to replay fixed ranked candidates through an ensemble."""

    def __init__(self, search_k: int):
        self.search_k = search_k
        self.documents = []

    def retrieve(self, request: RetrievalRequest):
        return list(self.documents)

    async def aretrieve(self, request: RetrievalRequest):
        return self.retrieve(request)


def load_dense_details(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as details_file:
        for line_number, line in enumerate(details_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id")
            if not case_id or case_id in rows:
                raise ValueError(f"{path}:{line_number} case_id가 잘못됐습니다.")
            rows[case_id] = row
    return rows


def evaluate_cached_hybrid_sweep(
    *,
    collection: Any,
    cases: list[EvaluationCase],
    planner_records: dict[str, PlannerQueryRecord],
    dense_details: dict[str, dict[str, Any]],
    evaluation_today: date,
    bm25_candidate_k: int,
    rrf_k: int,
    dense_weights: list[float],
    selected_dense_weight: float,
    rank_depth: int = 10,
    bm25_tokenizer: str = "kiwi",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(dense_details) != {case.case_id for case in cases}:
        raise ValueError("dense details와 dataset의 case_id가 일치하지 않습니다.")
    bm25 = BM25PolicyRetriever(
        collection=collection,
        search_k=bm25_candidate_k,
        today_provider=lambda: evaluation_today,
        tokenizer=BM25_TOKENIZERS[bm25_tokenizer],
    )
    base_rows = []
    for case in cases:
        planner_record = planner_records[case.case_id]
        dense_policy_ids = dense_details[case.case_id]["output"]["retrieved_policy_ids"]
        if planner_record.planner_route == "retriever":
            query = (planner_record.retrieval_queries or [case.user_input])[0]
            bm25_documents = bm25.retrieve(RetrievalRequest(
                query=query,
                user_profile=case.user_profile,
                exclude_expired=case.exclude_expired,
            ))
        else:
            query = None
            bm25_documents = []
        base_rows.append({
            "case_id": case.case_id,
            "query": query,
            "user_profile": case.user_profile,
            "exclude_expired": case.exclude_expired,
            "expected_policy_ids": case.expected_policy_ids,
            "dense_policy_ids": dense_policy_ids,
            "bm25_policy_ids": [
                document.metadata["plcyNo"] for document in bm25_documents
            ],
        })

    results_by_weight = {}
    selected_rows: list[dict[str, Any]] = []
    for weight in sorted(set([*dense_weights, selected_dense_weight])):
        dense_source = CachedPolicyRetriever(search_k=rank_depth)
        bm25_source = CachedPolicyRetriever(search_k=bm25_candidate_k)
        ensemble = EnsemblePolicyRetriever(
            retrievers=[dense_source, bm25_source],
            weights=[weight, 1 - weight],
            search_k=rank_depth,
            rrf_k=rrf_k,
        )
        rows = []
        for base_row in base_rows:
            dense_source.documents = [
                bm25.index.documents[policy_id]
                for policy_id in base_row["dense_policy_ids"]
            ]
            bm25_source.documents = [
                bm25.index.documents[policy_id]
                for policy_id in base_row["bm25_policy_ids"]
            ]
            hybrid_documents = (
                ensemble.retrieve(RetrievalRequest(
                    query=base_row["query"],
                    user_profile=base_row["user_profile"],
                    exclude_expired=base_row["exclude_expired"],
                ))
                if base_row["query"] is not None
                else []
            )
            rows.append({
                **base_row,
                "hybrid_policy_ids": [
                    document.metadata["plcyNo"] for document in hybrid_documents
                ],
            })
        results_by_weight[str(weight)] = {
            **score_ranked_rows(rows, result_key="hybrid_policy_ids"),
            "rank_movement_vs_dense": rank_movement(
                rows,
                baseline_key="dense_policy_ids",
                candidate_key="hybrid_policy_ids",
            ),
        }
        if weight == selected_dense_weight:
            selected_rows = rows

    return {
        "evaluation_cases": len(cases),
        "dense_candidate_k": rank_depth,
        "bm25_candidate_k": bm25_candidate_k,
        "rrf_k": rrf_k,
        "bm25_tokenizer": bm25_tokenizer,
        "baseline_dense": score_ranked_rows(base_rows, result_key="dense_policy_ids"),
        "bm25_only": score_ranked_rows(base_rows, result_key="bm25_policy_ids"),
        "hybrid_by_dense_weight": results_by_weight,
        "selected_dense_weight": selected_dense_weight,
    }, selected_rows
