import re
import statistics
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from tqdm import tqdm

from src.evaluation.metrics import (
    DEFAULT_K_VALUES,
    mean_or_none,
    median_or_none,
    rank_gold_policy_ids,
    recall_at_k,
)
from src.evaluation.langfuse import item_value
from src.evaluation.models import EvaluationCase, PlannerQueryRecord
from src.factory import create_embedding_model
from src.rag.retrievers import (
    BM25PolicyRetriever,
    DensePolicyRetriever,
    EnsemblePolicyRetriever,
    PolicyRetriever,
    RetrievalRequest,
    build_filter_from_profile,
    tokenize_korean_legacy,
    tokenize_korean_lexical,
)
from src.rag.reranker import LlamaCppReranker


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_RANK_DEPTH = 10
DEFAULT_RRF_K = 60
DEFAULT_HYBRID_BM25_CANDIDATE_K = 50
DEFAULT_HYBRID_DENSE_WEIGHT = 0.65
DEFAULT_HYBRID_RRF_K = 1
BM25_TOKENIZERS = {
    "kiwi": tokenize_korean_lexical,
    "legacy": tokenize_korean_legacy,
}


def create_query_embedding_model(
    provider: str,
    model_name: str,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
):
    kwargs = {"base_url": ollama_base_url} if provider == "ollama" else {}
    return create_embedding_model(provider=provider, model_name=model_name, **kwargs)


def build_policy_retriever(
    *,
    collection: Any,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: Any,
    retrieval_mode: str,
    rank_depth: int,
    evaluation_today: date,
    bm25_candidate_k: int = DEFAULT_HYBRID_BM25_CANDIDATE_K,
    dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
    rrf_k: int = DEFAULT_HYBRID_RRF_K,
    bm25_tokenizer: str = "kiwi",
) -> PolicyRetriever:
    if retrieval_mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"지원하지 않는 retrieval mode: {retrieval_mode}")
    try:
        tokenizer = BM25_TOKENIZERS[bm25_tokenizer]
    except KeyError as error:
        raise ValueError(f"지원하지 않는 BM25 tokenizer: {bm25_tokenizer}") from error
    if retrieval_mode == "bm25":
        return BM25PolicyRetriever(
            collection=collection,
            search_k=rank_depth,
            today_provider=lambda: evaluation_today,
            tokenizer=tokenizer,
        )
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(chroma_dir),
        embedding_function=embedding_model,
    )
    dense = DensePolicyRetriever(
        vector_store=vector_store,
        search_k=rank_depth,
        today_provider=lambda: evaluation_today,
    )
    if retrieval_mode == "dense":
        return dense
    bm25 = BM25PolicyRetriever(
        collection=collection,
        search_k=bm25_candidate_k if retrieval_mode == "hybrid" else rank_depth,
        today_provider=lambda: evaluation_today,
        tokenizer=tokenizer,
    )
    return EnsemblePolicyRetriever(
        retrievers=[dense, bm25],
        weights=[dense_weight, 1 - dense_weight],
        search_k=rank_depth,
        rrf_k=rrf_k,
    )


def reciprocal_rank_fusion(
    ranked_policy_id_lists: list[list[str]],
    *,
    rrf_k: int,
    limit: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    seen_order = 0
    for policy_ids in ranked_policy_id_lists:
        seen_in_list: set[str] = set()
        for rank, policy_id in enumerate(policy_ids, start=1):
            if not policy_id or policy_id in seen_in_list:
                continue
            seen_in_list.add(policy_id)
            if policy_id not in first_seen:
                first_seen[policy_id] = seen_order
                seen_order += 1
            scores[policy_id] = scores.get(policy_id, 0.0) + 1.0 / (rrf_k + rank)
            best_ranks[policy_id] = min(best_ranks.get(policy_id, rank), rank)
    ranked = sorted(
        scores,
        key=lambda policy_id: (
            -scores[policy_id],
            best_ranks[policy_id],
            first_seen[policy_id],
            policy_id,
        ),
    )
    return [(policy_id, scores[policy_id]) for policy_id in ranked[:limit]]


def build_retrieval_task(
    retriever: PolicyRetriever,
    planner_records: dict[str, PlannerQueryRecord] | None = None,
    *,
    planner_query_mode: str = "first",
    rrf_k: int = DEFAULT_RRF_K,
    reranker: LlamaCppReranker | None = None,
):
    if planner_query_mode not in {"first", "rrf"}:
        raise ValueError(f"지원하지 않는 Planner query mode: {planner_query_mode}")

    def task(*, item, **kwargs) -> dict[str, Any]:
        inputs = item_value(item, "input", {}) or {}
        metadata = item_value(item, "metadata", {}) or {}
        raw_query = inputs["user_input"]
        query = raw_query
        planner_output: dict[str, Any] = {}
        used_raw_fallback = False

        if planner_records is not None:
            case_id = metadata.get("case_id")
            if not case_id or case_id not in planner_records:
                raise ValueError(f"Planner cache에 없는 평가 item: {case_id}")
            record = planner_records[case_id]
            planner_output = {
                "planner_route": record.planner_route,
                "answer_strategy": record.answer_strategy,
                "planner_queries": record.retrieval_queries,
                "route_reason": record.route_reason,
                "planner_provider": record.planner_provider,
                "planner_model": record.planner_model,
            }
            if record.planner_route != "retriever":
                return {
                    **planner_output,
                    "raw_query": raw_query,
                    "executed_query": None,
                    "executed_queries": [],
                    "used_raw_fallback": False,
                    "retrieved_policy_ids": [],
                }
            queries = record.retrieval_queries or [raw_query]
            used_raw_fallback = not record.retrieval_queries
            if planner_query_mode == "rrf":
                ranked_lists = []
                for planner_query in queries:
                    documents = retriever.retrieve(RetrievalRequest(
                        query=planner_query,
                        user_profile=inputs.get("user_profile", {}),
                        exclude_expired=inputs.get("exclude_expired", False),
                    ))
                    ranked_lists.append([
                        document.metadata["plcyNo"]
                        for document in documents
                        if document.metadata.get("plcyNo")
                    ])
                fused = reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k, limit=retriever.search_k)
                return {
                    **planner_output,
                    "raw_query": raw_query,
                    "executed_query": None,
                    "executed_queries": queries,
                    "used_raw_fallback": used_raw_fallback,
                    "per_query_retrieved_policy_ids": ranked_lists,
                    "rrf_k": rrf_k,
                    "rrf_scores": dict(fused),
                    "retrieved_policy_ids": [policy_id for policy_id, _ in fused],
                }
            query = queries[0]

        request = RetrievalRequest(
            query=query,
            user_profile=inputs.get("user_profile", {}),
            exclude_expired=inputs.get("exclude_expired", False),
        )
        hybrid_output: dict[str, Any] = {}
        if isinstance(retriever, EnsemblePolicyRetriever):
            documents, source_documents = retriever.retrieve_with_sources(request)
            source_policy_ids = [[
                document.metadata["plcyNo"]
                for document in source_result
                if document.metadata.get("plcyNo")
            ] for source_result in source_documents]
            hybrid_output = {
                "source_retrieved_policy_ids": source_policy_ids,
                "hybrid_dense_weight": retriever.weights[0],
                "hybrid_rrf_k": retriever.rrf_k,
            }
            if source_policy_ids:
                hybrid_output["dense_retrieved_policy_ids"] = source_policy_ids[0]
            if len(source_policy_ids) > 1:
                hybrid_output["bm25_retrieved_policy_ids"] = source_policy_ids[1]
        else:
            documents = retriever.retrieve(request)

        dense_policy_ids = [
            document.metadata["plcyNo"]
            for document in documents
            if document.metadata.get("plcyNo")
        ]
        reranker_output: dict[str, Any] = {}
        if reranker is not None:
            started_at = time.perf_counter()
            reranked_documents = reranker.rerank(query=query, documents=documents)
            documents = [result.document for result in reranked_documents]
            reranker_output = {
                "dense_retrieved_policy_ids": dense_policy_ids,
                "reranker_model": reranker.model,
                "reranker_latency_ms": (time.perf_counter() - started_at) * 1000,
                "reranker_results": [{
                    "policy_id": result.document.metadata.get("plcyNo"),
                    "dense_rank": result.original_rank,
                    "rerank_score": result.relevance_score,
                } for result in reranked_documents],
            }
        return {
            **planner_output,
            **hybrid_output,
            **reranker_output,
            "raw_query": raw_query,
            "executed_query": query,
            "executed_queries": [query],
            "used_raw_fallback": used_raw_fallback,
            "retrieved_policy_ids": [
                document.metadata["plcyNo"]
                for document in documents
                if document.metadata.get("plcyNo")
            ],
        }

    return task


def evaluate_retrieval(
    collection: Any,
    retriever: PolicyRetriever,
    cases: list[EvaluationCase],
    rank_depth: int | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    exclude_expired: bool | None = None,
    today_yyyymmdd: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collection_count = collection.count()
    if collection_count < 1:
        raise ValueError("Chroma collection이 비어 있습니다.")
    maximum_rank_depth = min(rank_depth or collection_count, collection_count)
    if maximum_rank_depth < max(k_values):
        raise ValueError(
            f"rank_depth({maximum_rank_depth})는 최대 k({max(k_values)}) 이상이어야 합니다."
        )
    if retriever.search_k < maximum_rank_depth:
        raise ValueError(
            f"retriever.search_k({retriever.search_k})는 rank_depth({maximum_rank_depth}) 이상이어야 합니다."
        )
    effective_today_yyyymmdd = today_yyyymmdd or int(date.today().strftime("%Y%m%d"))
    effective_today = datetime.strptime(str(effective_today_yyyymmdd), "%Y%m%d").date()
    details: list[dict[str, Any]] = []
    recall_scores = {k: [] for k in k_values}
    reciprocal_ranks: list[float] = []
    first_relevant_ranks: list[int] = []
    all_gold_ranks: list[int] = []
    candidate_counts: list[int] = []
    filter_eligible_gold_count = 0
    total_gold_ids = sum(len(case.gold_policy_ids) for case in cases)

    for index, case in enumerate(tqdm(cases, desc="PolicyRetriever 평가"), start=1):
        case_exclude_expired = case.exclude_expired if exclude_expired is None else exclude_expired
        metadata_filter = build_filter_from_profile(
            case.user_profile,
            exclude_expired=case_exclude_expired,
            today=effective_today,
        )
        get_kwargs: dict[str, Any] = {"include": []}
        if metadata_filter:
            get_kwargs["where"] = metadata_filter
        eligible_ids = set(collection.get(**get_kwargs)["ids"])
        if not eligible_ids:
            raise ValueError(
                f"{index}번째 query의 metadata filter 결과가 비어 있습니다: {metadata_filter}"
            )
        candidate_counts.append(len(eligible_ids))
        filter_eligible_gold = {
            policy_id: policy_id in eligible_ids for policy_id in case.gold_policy_ids
        }
        filter_eligible_gold_count += sum(filter_eligible_gold.values())
        documents = retriever.retrieve(RetrievalRequest(
            query=case.user_input,
            user_profile=case.user_profile,
            exclude_expired=case_exclude_expired,
        ))
        retrieved_ids = [
            str(document.metadata.get("plcyNo") or "")
            for document in documents[:maximum_rank_depth]
        ]
        if any(not policy_id for policy_id in retrieved_ids):
            raise ValueError(f"{index}번째 Retriever 결과에 plcyNo가 없는 문서가 있습니다.")
        gold_ranks = rank_gold_policy_ids(retrieved_ids, case.gold_policy_ids)
        found_ranks = [rank for rank in gold_ranks.values() if rank is not None]
        first_relevant_rank = min(found_ranks) if found_ranks else None
        reciprocal = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        per_case_recall = {}
        for k in k_values:
            score = recall_at_k(retrieved_ids, case.gold_policy_ids, k)
            recall_scores[k].append(score)
            per_case_recall[f"recall_at_{k}"] = score
        reciprocal_ranks.append(reciprocal)
        if first_relevant_rank is not None:
            first_relevant_ranks.append(first_relevant_rank)
        all_gold_ranks.extend(found_ranks)
        details.append({
            "case_id": case.case_id,
            "case_index": index,
            "user_input": case.user_input,
            "expected_policy_ids": case.expected_policy_ids,
            "gold_policy_ids": case.gold_policy_ids,
            "gold_ranks": gold_ranks,
            "filter_eligible_gold": filter_eligible_gold,
            "first_relevant_rank": first_relevant_rank,
            "reciprocal_rank": reciprocal,
            "metadata_filter": metadata_filter,
            "candidate_count": len(eligible_ids),
            **per_case_recall,
            "top_retrieved_policy_ids": retrieved_ids[: max(k_values)],
        })

    metrics = {
        **{f"recall_at_{k}": statistics.mean(scores) for k, scores in recall_scores.items()},
        "mrr": statistics.mean(reciprocal_ranks),
        "mean_first_relevant_rank": mean_or_none(first_relevant_ranks),
        "median_first_relevant_rank": median_or_none(first_relevant_ranks),
        "mean_gold_rank": mean_or_none(all_gold_ranks),
        "median_gold_rank": median_or_none(all_gold_ranks),
        "gold_found_rate": len(all_gold_ranks) / total_gold_ids if total_gold_ids else 0.0,
        "found_gold_ids": len(all_gold_ranks),
        "total_gold_ids": total_gold_ids,
        "gold_filter_eligibility_rate": filter_eligible_gold_count / total_gold_ids if total_gold_ids else 0.0,
        "filter_eligible_gold_ids": filter_eligible_gold_count,
        "mean_candidate_count": mean_or_none(candidate_counts),
        "median_candidate_count": median_or_none(candidate_counts),
    }
    return {
        "metrics": metrics,
        "evaluation": {
            "example_count": len(cases),
            "collection_count": collection_count,
            "rank_depth": maximum_rank_depth,
            "retriever_class": type(retriever).__name__,
            "query_source": "user_input",
            "metadata_filtering": True,
            "exclude_expired": (
                "per_case" if exclude_expired is None else exclude_expired
            ),
            "today_yyyymmdd": effective_today_yyyymmdd,
        },
    }, details


def safe_experiment_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "retrieval-experiment"
