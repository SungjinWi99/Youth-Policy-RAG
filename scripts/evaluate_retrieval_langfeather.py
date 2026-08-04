"""Run retrieval recall@k through LangFeather's dataset + evaluate() API.

Unlike ``scripts/evaluate_retrieval.py`` (which writes results to local JSON
files only), this script pushes the evaluation dataset and every experiment
case to a LangFeather server so runs are traced and comparable in the
LangFeather UI. Recall@3/5/10 is computed twice by design: once inside
LangFeather evaluators (so the server has authoritative per-case scores) and
once locally from the same retrieval output (so this script can print/save a
summary without a second round trip to the server).
"""

import argparse
import json
import os
import statistics
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import langfeather
from dotenv import load_dotenv

from src.evaluation.datasets import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATASET_PATH,
    DEFAULT_RAW_POLICY_PATH,
    load_corpus_policy_ids,
    load_evaluation_cases,
    open_collection,
    project_path,
    validate_collection_corpus,
    validate_filter_metadata,
    validate_gold_coverage,
)
from src.evaluation.metrics import DEFAULT_K_VALUES, recall_at_k
from src.evaluation.models import EvaluationCase
from src.evaluation.retrieval import (
    DEFAULT_HYBRID_BM25_CANDIDATE_K,
    DEFAULT_HYBRID_DENSE_WEIGHT,
    DEFAULT_HYBRID_RRF_K,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RANK_DEPTH,
    build_policy_retriever,
    safe_experiment_name,
)
from src.rag.retrievers import PolicyRetriever, RetrievalRequest


DEFAULT_LANGFEATHER_ENDPOINT = "http://127.0.0.1:4319"
DEFAULT_OUTPUT_DIR = project_path("data/eval/langfeather_retrieval_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangFeather dataset/evaluate()로 retrieval recall@3/5/10 평가"
    )
    parser.add_argument("--provider", choices=("ollama", "openai", "upstage"))
    parser.add_argument("--model", help="dense/hybrid query embedding 모델명")
    parser.add_argument("--chroma-dir", type=Path, required=True)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--raw-policy-path", type=Path, default=DEFAULT_RAW_POLICY_PATH)
    parser.add_argument("--skip-corpus-validation", action="store_true")
    parser.add_argument("--retrieval-mode", choices=("dense", "bm25", "hybrid"), default="dense")
    parser.add_argument("--rank-depth", type=int, default=DEFAULT_RANK_DEPTH)
    parser.add_argument("--today", help="마감 필터 기준일 YYYYMMDD")
    parser.add_argument("--limit", type=int, help="local smoke test에서 사용할 앞 N개 case")
    parser.add_argument("--exclude-expired", action="store_true", help="만료 정책 제외")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--bm25-candidate-k", type=int, default=DEFAULT_HYBRID_BM25_CANDIDATE_K)
    parser.add_argument("--bm25-tokenizer", choices=("kiwi", "legacy"), default="kiwi")
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_HYBRID_DENSE_WEIGHT)
    parser.add_argument("--hybrid-rrf-k", type=int, default=DEFAULT_HYBRID_RRF_K)
    parser.add_argument(
        "--langfeather-endpoint",
        default=os.getenv("LANGFEATHER_ENDPOINT", DEFAULT_LANGFEATHER_ENDPOINT),
    )
    parser.add_argument(
        "--langfeather-dataset-name",
        help="생략 시 평가 데이터 파일명에서 유도합니다.",
    )
    parser.add_argument("--experiment-name")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.rank_depth < 1:
        parser.error("--rank-depth는 1 이상이어야 합니다.")
    if args.retrieval_mode != "bm25" and not (args.provider and args.model):
        parser.error("dense/hybrid 모드에는 --provider와 --model이 필요합니다.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit은 1 이상이어야 합니다.")
    if args.hybrid_rrf_k < 1:
        parser.error("RRF k는 1 이상이어야 합니다.")
    if args.bm25_candidate_k < args.rank_depth:
        parser.error("--bm25-candidate-k는 --rank-depth 이상이어야 합니다.")
    if not 0 <= args.dense_weight <= 1:
        parser.error("--dense-weight는 0과 1 사이여야 합니다.")
    if args.rank_depth < max(DEFAULT_K_VALUES):
        parser.error(f"--rank-depth는 최대 k({max(DEFAULT_K_VALUES)}) 이상이어야 합니다.")
    try:
        if args.today:
            datetime.strptime(args.today, "%Y%m%d")
    except ValueError:
        parser.error("--today는 유효한 YYYYMMDD 형식이어야 합니다.")
    return args


def case_dataset_example_input(case: EvaluationCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "user_input": case.user_input,
        "user_profile": case.user_profile,
        "exclude_expired": case.exclude_expired,
    }


def build_or_reuse_dataset(
    *,
    name: str,
    cases: list[EvaluationCase],
    endpoint: str,
) -> "langfeather.Dataset":
    def make_example(case: EvaluationCase, index: int) -> "langfeather.DatasetExample":
        return langfeather.DatasetExample(
            input=case_dataset_example_input(case),
            expected_output=list(case.gold_policy_ids),
            metadata={"case_index": index, **case.metadata},
        )

    examples = [make_example(case, index) for index, case in enumerate(cases, start=1)]
    dataset = langfeather.get_or_create_dataset(
        name=name,
        examples=examples,
        description=f"Youth policy retrieval recall@k eval ({len(cases)} cases)",
        endpoint=endpoint,
    )

    # get_or_create_dataset() does not top up an existing dataset, so a name reused
    # from a smaller (e.g. smoke-test) run would otherwise silently under-evaluate.
    existing_case_ids = {
        example.input.get("case_id")
        for example in dataset.examples
        if isinstance(example.input, dict)
    }
    missing = [
        make_example(case, index)
        for index, case in enumerate(cases, start=1)
        if case.case_id not in existing_case_ids
    ]
    if missing:
        dataset = langfeather.add_dataset_examples(dataset.dataset_id, missing, endpoint=endpoint)
    return dataset


def build_recall_evaluator(k: int) -> "langfeather.Evaluator":
    @langfeather.evaluator(key=f"recall_at_{k}", name=f"Recall@{k}", data_type="number")
    def _recall(
        *,
        input: Any,
        output: Any,
        expected_output: Any,
        metadata: Any,
    ) -> float:
        del input, metadata
        retrieved = (output or {}).get("retrieved_policy_ids", [])
        gold = expected_output or []
        return recall_at_k(retrieved, gold, k)

    return _recall


def build_target(retriever: PolicyRetriever, *, rank_depth: int, local_results: list[dict[str, Any]]):
    def target(case_input: Any) -> dict[str, Any]:
        case_input = case_input or {}
        documents = retriever.retrieve(
            RetrievalRequest(
                query=case_input["user_input"],
                user_profile=case_input.get("user_profile", {}),
                exclude_expired=case_input.get("exclude_expired", False),
            )
        )
        retrieved_ids = [
            str(document.metadata.get("plcyNo") or "")
            for document in documents[:rank_depth]
        ]
        local_results.append({
            "case_id": case_input.get("case_id"),
            "retrieved_policy_ids": retrieved_ids,
        })
        return {"retrieved_policy_ids": retrieved_ids}

    return target


def summarize_locally(
    cases: list[EvaluationCase],
    local_results: list[dict[str, Any]],
) -> dict[str, Any]:
    results_by_case_id = {row["case_id"]: row["retrieved_policy_ids"] for row in local_results}
    failed_case_ids = [case.case_id for case in cases if case.case_id not in results_by_case_id]

    recalls = {k: [] for k in DEFAULT_K_VALUES}
    details = []
    for case in cases:
        if case.case_id not in results_by_case_id:
            continue
        retrieved = results_by_case_id[case.case_id]
        per_case = {}
        for k in DEFAULT_K_VALUES:
            score = recall_at_k(retrieved, case.gold_policy_ids, k)
            recalls[k].append(score)
            per_case[f"recall_at_{k}"] = score
        details.append({
            "case_id": case.case_id,
            "user_input": case.user_input,
            "gold_policy_ids": case.gold_policy_ids,
            "retrieved_policy_ids": retrieved,
            **per_case,
        })
    metrics = (
        {f"recall_at_{k}": statistics.mean(scores) for k, scores in recalls.items()}
        if details
        else {f"recall_at_{k}": 0.0 for k in DEFAULT_K_VALUES}
    )
    return {"metrics": metrics, "details": details, "failed_case_ids": failed_case_ids}


def write_local_summary(
    *,
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
    existing = [path for path in (summary_path, details_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"결과 파일이 이미 존재합니다: {existing}. --overwrite를 사용하세요.")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as details_file:
        for detail in details:
            details_file.write(json.dumps(detail, ensure_ascii=False) + "\n")
    return summary_path, details_path


def main() -> None:
    load_dotenv()
    args = parse_args()

    dataset_path = project_path(args.dataset)
    chroma_dir = project_path(args.chroma_dir)
    cases = load_evaluation_cases(dataset_path)
    if args.limit is not None:
        cases = cases[: args.limit]

    collection = open_collection(chroma_dir, args.collection)
    if not args.skip_corpus_validation:
        validate_collection_corpus(collection, load_corpus_policy_ids(project_path(args.raw_policy_path)))
    validate_filter_metadata(collection)
    validate_gold_coverage(collection, cases)

    from src.evaluation.retrieval import create_query_embedding_model

    embedding_model = (
        create_query_embedding_model(
            provider=args.provider,
            model_name=args.model,
            ollama_base_url=args.ollama_base_url,
        )
        if args.retrieval_mode != "bm25"
        else None
    )
    evaluation_today = datetime.strptime(args.today, "%Y%m%d").date() if args.today else date.today()
    retriever = build_policy_retriever(
        collection=collection,
        chroma_dir=chroma_dir,
        collection_name=args.collection,
        embedding_model=embedding_model,
        retrieval_mode=args.retrieval_mode,
        rank_depth=args.rank_depth,
        evaluation_today=evaluation_today,
        bm25_candidate_k=args.bm25_candidate_k,
        dense_weight=args.dense_weight,
        rrf_k=args.hybrid_rrf_k,
        bm25_tokenizer=args.bm25_tokenizer,
    )

    langfeather.configure(args.langfeather_endpoint)

    dataset_name = args.langfeather_dataset_name or f"youth-policy-retrieval-{dataset_path.stem}"
    lf_dataset = build_or_reuse_dataset(
        name=dataset_name,
        cases=cases,
        endpoint=args.langfeather_endpoint,
    )

    experiment_name = args.experiment_name or (
        "bm25" if args.retrieval_mode == "bm25" else f"{args.retrieval_mode}-{args.provider}-{args.model}"
    )

    local_results: list[dict[str, Any]] = []
    target = build_target(retriever, rank_depth=args.rank_depth, local_results=local_results)
    evaluators = [build_recall_evaluator(k) for k in DEFAULT_K_VALUES]

    started_at = datetime.now(timezone.utc)
    run = langfeather.evaluate(
        dataset=lf_dataset.dataset_id,
        target=target,
        evaluators=evaluators,
        name=experiment_name,
        endpoint=args.langfeather_endpoint,
        target_metadata={
            "provider": args.provider,
            "query_model": args.model,
            "retrieval_mode": args.retrieval_mode,
            "rank_depth": args.rank_depth,
            "chroma_dir": str(chroma_dir),
            "collection": args.collection,
            "dataset_path": str(dataset_path),
        },
    )
    langfeather.flush()

    if run.case_count != len(cases):
        raise RuntimeError(
            f"LangFeather experiment case_count({run.case_count})가 "
            f"local case 수({len(cases)})와 다릅니다. dataset이 이전 실행과 "
            "이름이 겹쳤을 수 있습니다 (--langfeather-dataset-name으로 분리하세요)."
        )
    if len(local_results) != run.completed_case_count:
        raise RuntimeError(
            f"target() 실행 횟수({len(local_results)})가 LangFeather "
            f"completed_case_count({run.completed_case_count})와 다릅니다."
        )

    local_summary = summarize_locally(cases, local_results)
    summary = {
        "experiment": {
            "name": experiment_name,
            "provider": args.provider,
            "query_model": args.model,
            "retrieval_mode": args.retrieval_mode,
            "chroma_dir": str(chroma_dir),
            "collection": args.collection,
            "dataset": str(dataset_path),
            "started_at": started_at.isoformat(),
        },
        "langfeather": {
            "endpoint": args.langfeather_endpoint,
            "dataset_id": lf_dataset.dataset_id,
            "dataset_name": lf_dataset.name,
            "dataset_revision": lf_dataset.revision,
            "experiment_id": run.experiment_id,
            "status": run.status,
            "case_count": run.case_count,
            "completed_case_count": run.completed_case_count,
            "failed_case_count": run.failed_case_count,
        },
        "metrics": local_summary["metrics"],
        "failed_case_ids": local_summary["failed_case_ids"],
    }
    if local_summary["failed_case_ids"]:
        print(
            f"WARNING: {len(local_summary['failed_case_ids'])}/{len(cases)} case(s) failed "
            f"during retrieve() and were excluded from recall metrics: "
            f"{local_summary['failed_case_ids'][:10]}"
        )
    summary_path, details_path = write_local_summary(
        summary=summary,
        details=local_summary["details"],
        output_dir=project_path(args.output_dir),
        experiment_name=experiment_name,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")


if __name__ == "__main__":
    main()
