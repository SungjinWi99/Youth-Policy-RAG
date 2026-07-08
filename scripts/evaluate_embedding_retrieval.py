import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factory import create_embedding_model
from src.rag.nodes.retriever import build_user_filter
from src.policy.utils import REGION_CODES, region_metadata_key


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data/eval/eval_v1_500.jsonl"
)
DEFAULT_RAW_POLICY_PATH = PROJECT_ROOT / "data/raw/youth_policies.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/eval/embedding_retrieval_results"
DEFAULT_COLLECTION_NAME = "youth_policies_rag"
DEFAULT_K_VALUES = (3, 5, 10)
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
REQUIRED_FILTER_METADATA_KEYS = {
    "agePolicy",
    "sprtTrgtMinAge",
    "sprtTrgtMaxAge",
    "incomePolicy",
    "earnMinAmt",
    "earnMaxAmt",
    "applicationPolicy",
    "applicationEndYmd",
    *{
        region_metadata_key(region_code)
        for region_code in REGION_CODES
    },
}


class RetrievalEvaluationCase(BaseModel):
    expected_policy_ids: list[str] = Field(min_length=1)
    user_input: str = Field(min_length=1)
    user_profile: dict[str, Any] = Field(default_factory=dict)

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


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_evaluation_cases(
    path: Path,
) -> list[RetrievalEvaluationCase]:
    cases = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                cases.append(
                    RetrievalEvaluationCase.model_validate_json(line)
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} 평가 데이터가 유효하지 않습니다."
                ) from error
    if not cases:
        raise ValueError(f"{path}에 평가 데이터가 없습니다.")
    return cases


def load_corpus_policy_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as policy_file:
        policies = json.load(policy_file)
    if not isinstance(policies, list) or not policies:
        raise ValueError(f"{path}에는 비어 있지 않은 JSON 배열이 필요합니다.")

    policy_ids = {
        str(policy.get("plcyNo") or "").strip()
        for policy in policies
        if isinstance(policy, dict)
    }
    if "" in policy_ids:
        raise ValueError(f"{path}에 plcyNo가 없는 정책이 있습니다.")
    if len(policy_ids) != len(policies):
        raise ValueError(f"{path}에 중복 plcyNo가 있습니다.")
    return policy_ids


def create_query_embedding_model(
    provider: str,
    model_name: str,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
):
    kwargs = {}
    if provider == "ollama":
        kwargs["base_url"] = ollama_base_url
    return create_embedding_model(
        provider=provider,
        model_name=model_name,
        **kwargs,
    )


def open_collection(
    chroma_dir: Path,
    collection_name: str,
):
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        return client.get_collection(collection_name)
    except Exception as error:
        available = sorted(collection.name for collection in client.list_collections())
        raise ValueError(
            f"{chroma_dir}에서 collection '{collection_name}'을 찾지 못했습니다. "
            f"사용 가능한 collection: {available}"
        ) from error


def validate_collection_corpus(
    collection: Any,
    corpus_policy_ids: set[str],
) -> None:
    stored = collection.get(include=[])
    stored_ids = set(stored["ids"])
    missing_ids = corpus_policy_ids - stored_ids
    extra_ids = stored_ids - corpus_policy_ids
    if missing_ids or extra_ids:
        raise ValueError(
            "Chroma collection과 원천 정책 ID가 일치하지 않습니다. "
            f"collection={len(stored_ids)}, corpus={len(corpus_policy_ids)}, "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )


def validate_gold_coverage(
    collection: Any,
    cases: list[RetrievalEvaluationCase],
) -> None:
    gold_ids = sorted(
        {
            policy_id
            for case in cases
            for policy_id in case.gold_policy_ids
        }
    )
    stored_gold_ids = set(
        collection.get(ids=gold_ids, include=[])["ids"]
    )
    missing_gold_ids = set(gold_ids) - stored_gold_ids
    if missing_gold_ids:
        preview = sorted(missing_gold_ids)[:10]
        raise ValueError(
            f"Chroma collection에 gold policy {len(missing_gold_ids)}건이 "
            f"없습니다: {preview}"
        )


def validate_filter_metadata(collection: Any) -> None:
    stored = collection.get(include=["metadatas"])
    invalid_records = []
    for policy_id, metadata in zip(
        stored["ids"],
        stored["metadatas"],
        strict=True,
    ):
        missing_keys = REQUIRED_FILTER_METADATA_KEYS - set(metadata or {})
        if missing_keys:
            invalid_records.append(
                {
                    "policy_id": policy_id,
                    "missing_keys": sorted(missing_keys),
                }
            )
            if len(invalid_records) >= 5:
                break
    if invalid_records:
        raise ValueError(
            "Chroma collection에 metadata filter 필수 키가 누락됐습니다: "
            f"{invalid_records}"
        )


def recall_at_k(
    retrieved_policy_ids: list[str],
    gold_policy_ids: list[str],
    k: int,
) -> float:
    gold = set(gold_policy_ids)
    if not gold:
        return 0.0
    retrieved = set(retrieved_policy_ids[:k])
    return len(retrieved & gold) / len(gold)


def rank_gold_policy_ids(
    retrieved_policy_ids: list[str],
    gold_policy_ids: list[str],
) -> dict[str, int | None]:
    ranks = {
        policy_id: rank
        for rank, policy_id in enumerate(retrieved_policy_ids, start=1)
    }
    return {
        policy_id: ranks.get(policy_id)
        for policy_id in gold_policy_ids
    }


def _mean_or_none(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _median_or_none(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def evaluate_retrieval(
    collection: Any,
    embedding_model: Any,
    cases: list[RetrievalEvaluationCase],
    rank_depth: int | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    exclude_expired: bool = False,
    today_yyyymmdd: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collection_count = collection.count()
    if collection_count < 1:
        raise ValueError("Chroma collection이 비어 있습니다.")

    maximum_rank_depth = rank_depth or collection_count
    maximum_rank_depth = min(maximum_rank_depth, collection_count)
    if maximum_rank_depth < max(k_values):
        raise ValueError(
            f"rank_depth({maximum_rank_depth})는 최대 k({max(k_values)}) "
            "이상이어야 합니다."
        )
    effective_today = today_yyyymmdd or int(
        date.today().strftime("%Y%m%d")
    )

    details = []
    recall_scores = {k: [] for k in k_values}
    reciprocal_ranks = []
    first_relevant_ranks = []
    all_gold_ranks = []
    candidate_counts = []
    filter_eligible_gold_count = 0
    total_gold_ids = sum(len(case.gold_policy_ids) for case in cases)

    for index, case in enumerate(
        tqdm(cases, desc="Embedding retrieval 평가"),
        start=1,
    ):
        metadata_filter = build_user_filter(
            case.user_profile,
            exclude_expired=exclude_expired,
            today_yyyymmdd=effective_today,
        )
        get_kwargs: dict[str, Any] = {"include": []}
        if metadata_filter:
            get_kwargs["where"] = metadata_filter
        eligible_ids = set(collection.get(**get_kwargs)["ids"])
        candidate_count = len(eligible_ids)
        if candidate_count < 1:
            raise ValueError(
                f"{index}번째 query의 metadata filter 결과가 비어 있습니다: "
                f"{metadata_filter}"
            )
        candidate_counts.append(candidate_count)
        filter_eligible_gold = {
            policy_id: policy_id in eligible_ids
            for policy_id in case.gold_policy_ids
        }
        filter_eligible_gold_count += sum(filter_eligible_gold.values())

        query_embedding = embedding_model.embed_query(case.user_input + " " +case.user_profile["region"])
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(maximum_rank_depth, candidate_count),
            "include": ["distances"],
        }
        if metadata_filter:
            query_kwargs["where"] = metadata_filter
        query_result = collection.query(**query_kwargs)
        retrieved_ids = [
            str(policy_id)
            for policy_id in query_result["ids"][0]
        ]
        distances = [
            float(distance)
            for distance in query_result["distances"][0]
        ]
        gold_ranks = rank_gold_policy_ids(
            retrieved_ids,
            case.gold_policy_ids,
        )
        found_ranks = [
            rank
            for rank in gold_ranks.values()
            if rank is not None
        ]
        first_relevant_rank = min(found_ranks) if found_ranks else None
        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        per_case_recall = {}
        for k in k_values:
            score = recall_at_k(
                retrieved_ids,
                case.gold_policy_ids,
                k,
            )
            recall_scores[k].append(score)
            per_case_recall[f"recall_at_{k}"] = score

        reciprocal_ranks.append(reciprocal_rank)
        first_relevant_ranks.extend(
            [first_relevant_rank]
            if first_relevant_rank is not None
            else []
        )
        all_gold_ranks.extend(found_ranks)
        top_result_count = max(k_values)
        details.append(
            {
                "case_index": index,
                "user_input": case.user_input,
                "gold_policy_ids": case.gold_policy_ids,
                "gold_ranks": gold_ranks,
                "filter_eligible_gold": filter_eligible_gold,
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": reciprocal_rank,
                "metadata_filter": metadata_filter,
                "candidate_count": candidate_count,
                **per_case_recall,
                "top_retrieved_policy_ids": retrieved_ids[:top_result_count],
                "top_distances": distances[:top_result_count],
            }
        )

    found_gold_ids = len(all_gold_ranks)
    metrics = {
        **{
            f"recall_at_{k}": statistics.mean(scores)
            for k, scores in recall_scores.items()
        },
        "mrr": statistics.mean(reciprocal_ranks),
        "mean_first_relevant_rank": _mean_or_none(first_relevant_ranks),
        "median_first_relevant_rank": _median_or_none(first_relevant_ranks),
        "mean_gold_rank": _mean_or_none(all_gold_ranks),
        "median_gold_rank": _median_or_none(all_gold_ranks),
        "gold_found_rate": (
            found_gold_ids / total_gold_ids
            if total_gold_ids
            else 0.0
        ),
        "found_gold_ids": found_gold_ids,
        "total_gold_ids": total_gold_ids,
        "gold_filter_eligibility_rate": (
            filter_eligible_gold_count / total_gold_ids
            if total_gold_ids
            else 0.0
        ),
        "filter_eligible_gold_ids": filter_eligible_gold_count,
        "mean_candidate_count": _mean_or_none(candidate_counts),
        "median_candidate_count": _median_or_none(candidate_counts),
    }
    evaluation_info = {
        "example_count": len(cases),
        "collection_count": collection_count,
        "rank_depth": maximum_rank_depth,
        "metadata_filtering": True,
        "exclude_expired": exclude_expired,
        "today_yyyymmdd": effective_today,
    }
    return {
        "metrics": metrics,
        "evaluation": evaluation_info,
    }, details


def safe_experiment_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value)
    return normalized.strip("-") or "embedding-experiment"


def write_results(
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    output_dir: Path,
    experiment_name: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_experiment_name(experiment_name)
    summary_path = output_dir / f"{prefix}.summary.json"
    details_path = output_dir / f"{prefix}.details.jsonl"

    existing_paths = [
        path
        for path in (summary_path, details_path)
        if path.exists()
    ]
    if existing_paths and not overwrite:
        raise FileExistsError(
            f"결과 파일이 이미 존재합니다: {existing_paths}. "
            "덮어쓰려면 --overwrite를 사용하세요."
        )

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with details_path.open("w", encoding="utf-8") as details_file:
        for detail in details:
            details_file.write(
                json.dumps(detail, ensure_ascii=False) + "\n"
            )
    return summary_path, details_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "user_profile metadata filter를 적용한 Chroma embedding 검색의 "
            "Recall@3/5/10과 gold policy 순위를 평가합니다."
        )
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("ollama", "openai", "upstage"),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="query embedding 모델명",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        required=True,
        help="평가할 ChromaDB 디렉터리",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--raw-policy-path",
        type=Path,
        default=DEFAULT_RAW_POLICY_PATH,
        help="collection 전체 ID 일치 검증에 사용할 원천 정책 JSON",
    )
    parser.add_argument(
        "--skip-corpus-validation",
        action="store_true",
        help="원천 정책과 collection의 전체 ID 일치 검증 생략",
    )
    parser.add_argument(
        "--rank-depth",
        type=int,
        default=0,
        help="gold 순위를 찾을 검색 깊이. 0이면 collection 전체",
    )
    parser.add_argument(
        "--exclude-expired",
        action="store_true",
        help="신청 마감일이 지난 정책을 metadata filter에서 제외",
    )
    parser.add_argument(
        "--today",
        help="마감 필터 기준일 YYYYMMDD. 생략하면 실행일",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="앞에서부터 일부 query만 평가하는 smoke test 옵션",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
    )
    parser.add_argument(
        "--experiment-name",
        help="결과 파일 이름. 생략하면 provider-model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()
    if args.rank_depth < 0:
        parser.error("--rank-depth는 0 이상이어야 합니다.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit은 1 이상이어야 합니다.")
    if args.today is not None:
        try:
            datetime.strptime(args.today, "%Y%m%d")
        except ValueError:
            parser.error("--today는 유효한 YYYYMMDD 형식이어야 합니다.")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    dataset_path = project_path(args.dataset)
    chroma_dir = project_path(args.chroma_dir)
    output_dir = project_path(args.output_dir)
    cases = load_evaluation_cases(dataset_path)
    if args.limit is not None:
        cases = cases[:args.limit]

    collection = open_collection(
        chroma_dir=chroma_dir,
        collection_name=args.collection,
    )
    if not args.skip_corpus_validation:
        corpus_policy_ids = load_corpus_policy_ids(
            project_path(args.raw_policy_path)
        )
        validate_collection_corpus(collection, corpus_policy_ids)
    validate_filter_metadata(collection)
    validate_gold_coverage(collection, cases)

    embedding_model = create_query_embedding_model(
        provider=args.provider,
        model_name=args.model,
        ollama_base_url=args.ollama_base_url,
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result, details = evaluate_retrieval(
        collection=collection,
        embedding_model=embedding_model,
        cases=cases,
        rank_depth=args.rank_depth or None,
        exclude_expired=args.exclude_expired,
        today_yyyymmdd=(
            int(args.today)
            if args.today is not None
            else None
        ),
    )
    duration_seconds = time.perf_counter() - started
    experiment_name = args.experiment_name or (
        f"{args.provider}-{args.model}"
    )
    summary = {
        "experiment": {
            "name": experiment_name,
            "provider": args.provider,
            "query_model": args.model,
            "chroma_dir": str(chroma_dir),
            "collection": args.collection,
            "dataset": str(dataset_path),
            "started_at": started_at.isoformat(),
            "duration_seconds": duration_seconds,
        },
        **result,
    }
    summary_path, details_path = write_results(
        summary=summary,
        details=details,
        output_dir=output_dir,
        experiment_name=experiment_name,
        overwrite=args.overwrite,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")


if __name__ == "__main__":
    main()
