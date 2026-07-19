import json
from pathlib import Path
from typing import Any

import chromadb

from src.evaluation.models import EvaluationCase, PlannerQueryRecord
from src.policy.utils import REGION_CODES, region_metadata_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data/eval/eval_v1_200.jsonl"
DEFAULT_RAW_POLICY_PATH = PROJECT_ROOT / "data/raw/youth_policies.json"
DEFAULT_COLLECTION_NAME = "youth_policies_rag"
REQUIRED_FILTER_METADATA_KEYS = {
    "agePolicy",
    "sprtTrgtMinAge",
    "sprtTrgtMaxAge",
    "incomePolicy",
    "earnMinAmt",
    "earnMaxAmt",
    "applicationPolicy",
    "applicationEndYmd",
    *{region_metadata_key(region_code) for region_code in REGION_CODES},
}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                case = EvaluationCase.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} 평가 데이터가 유효하지 않습니다."
                ) from error
            if case.case_id in seen_case_ids:
                raise ValueError(f"{path}:{line_number} 중복 case_id: {case.case_id}")
            seen_case_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}에 평가 데이터가 없습니다.")
    return cases


def load_planner_query_records(
    path: Path,
    cases: list[EvaluationCase],
) -> dict[str, PlannerQueryRecord]:
    records: dict[str, PlannerQueryRecord] = {}
    with path.open(encoding="utf-8") as cache_file:
        for line_number, line in enumerate(cache_file, start=1):
            if not line.strip():
                continue
            try:
                record = PlannerQueryRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} Planner cache가 유효하지 않습니다."
                ) from error
            if record.case_id in records:
                raise ValueError(f"{path}:{line_number} 중복 case_id: {record.case_id}")
            records[record.case_id] = record

    cases_by_id = {case.case_id: case for case in cases}
    if set(records) != set(cases_by_id):
        missing = sorted(set(cases_by_id) - set(records))
        extra = sorted(set(records) - set(cases_by_id))
        raise ValueError(
            "Planner cache와 평가 데이터셋의 case_id가 일치하지 않습니다. "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for case_id, record in records.items():
        case = cases_by_id[case_id]
        if record.raw_query != case.user_input:
            raise ValueError(f"Planner cache raw_query 불일치: {case_id}")
        if record.user_profile != case.user_profile:
            raise ValueError(f"Planner cache user_profile 불일치: {case_id}")
    return records


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


def open_collection(chroma_dir: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        return client.get_collection(collection_name)
    except Exception as error:
        available = sorted(item.name for item in client.list_collections())
        raise ValueError(
            f"{chroma_dir}에서 collection '{collection_name}'을 찾지 못했습니다. "
            f"사용 가능한 collection: {available}"
        ) from error


def validate_collection_corpus(collection: Any, corpus_policy_ids: set[str]) -> None:
    stored_ids = set(collection.get(include=[])["ids"])
    missing_ids = corpus_policy_ids - stored_ids
    extra_ids = stored_ids - corpus_policy_ids
    if missing_ids or extra_ids:
        raise ValueError(
            "Chroma collection과 원천 정책 ID가 일치하지 않습니다. "
            f"collection={len(stored_ids)}, corpus={len(corpus_policy_ids)}, "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )


def validate_gold_coverage(collection: Any, cases: list[EvaluationCase]) -> None:
    gold_ids = sorted({policy_id for case in cases for policy_id in case.gold_policy_ids})
    stored_gold_ids = set(collection.get(ids=gold_ids, include=[])["ids"])
    missing_gold_ids = set(gold_ids) - stored_gold_ids
    if missing_gold_ids:
        raise ValueError(
            f"Chroma collection에 gold policy {len(missing_gold_ids)}건이 없습니다: "
            f"{sorted(missing_gold_ids)[:10]}"
        )


def validate_filter_metadata(collection: Any) -> None:
    stored = collection.get(include=["metadatas"])
    invalid_records = []
    for policy_id, metadata in zip(stored["ids"], stored["metadatas"], strict=True):
        missing_keys = REQUIRED_FILTER_METADATA_KEYS - set(metadata or {})
        if missing_keys:
            invalid_records.append({
                "policy_id": policy_id,
                "missing_keys": sorted(missing_keys),
            })
            if len(invalid_records) >= 5:
                break
    if invalid_records:
        raise ValueError(
            "Chroma collection에 metadata filter 필수 키가 누락됐습니다: "
            f"{invalid_records}"
        )
