"""Run retrieval benchmarks locally or in Langfuse, and sweep hybrid weights."""

import argparse
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config
from src.evaluation.datasets import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATASET_PATH,
    DEFAULT_RAW_POLICY_PATH,
    load_corpus_policy_ids,
    load_evaluation_cases,
    load_planner_query_records,
    open_collection,
    project_path,
    validate_collection_corpus,
    validate_filter_metadata,
    validate_gold_coverage,
)
from src.evaluation.hybrid_sweep import (
    evaluate_cached_hybrid_sweep,
    load_dense_details,
)
from src.evaluation.langfuse import (
    build_mean_run_evaluator,
    build_recall_evaluator,
    ensure_retrieval_dataset,
    item_value,
    planner_raw_fallback_evaluator,
    planner_retriever_route_evaluator,
    reciprocal_rank_evaluator,
    reranker_latency_evaluator,
)
from src.evaluation.retrieval import (
    DEFAULT_HYBRID_BM25_CANDIDATE_K,
    DEFAULT_HYBRID_DENSE_WEIGHT,
    DEFAULT_HYBRID_RRF_K,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RANK_DEPTH,
    DEFAULT_RRF_K,
    build_policy_retriever,
    build_retrieval_task,
    create_query_embedding_model,
    evaluate_retrieval,
    safe_experiment_name,
)
from src.observability import initialize_langfuse, shutdown_langfuse
from src.rag.reranker import LlamaCppReranker


DEFAULT_OUTPUT_DIR = project_path("data/eval/retrieval_results")
DEFAULT_LANGFUSE_DATASET_NAME = "PolicyRetrievalEval_v1_200"
DEFAULT_RERANKER_BASE_URL = "http://127.0.0.1:11435"


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracking", choices=("local", "langfuse"), default="local")
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
    parser.add_argument("--exclude-expired", action="store_true", help="local 실험의 만료 정책 제외")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--experiment-name")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-details", type=Path, help="Langfuse item 결과 JSONL")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--planner-query-cache", type=Path)
    parser.add_argument("--planner-query-mode", choices=("first", "rrf"), default="first")
    parser.add_argument("--planner-rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--langfuse-dataset-name", default=DEFAULT_LANGFUSE_DATASET_NAME)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-base-url", default=DEFAULT_RERANKER_BASE_URL)
    parser.add_argument("--reranker-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--bm25-candidate-k", type=int, default=DEFAULT_HYBRID_BM25_CANDIDATE_K)
    parser.add_argument("--bm25-tokenizer", choices=("kiwi", "legacy"), default="kiwi")
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_HYBRID_DENSE_WEIGHT)
    parser.add_argument("--hybrid-rrf-k", type=int, default=DEFAULT_HYBRID_RRF_K)


def parse_weights(value: str) -> list[float]:
    try:
        weights = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("가중치는 숫자여야 합니다.") from error
    if not weights or any(not 0 <= weight <= 1 for weight in weights):
        raise argparse.ArgumentTypeError("가중치는 0과 1 사이여야 합니다.")
    return weights


def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--planner-query-cache", type=Path, required=True)
    parser.add_argument("--dense-details", type=Path, required=True)
    parser.add_argument("--chroma-dir", type=Path, required=True)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--today", required=True, help="YYYYMMDD")
    parser.add_argument("--rank-depth", type=int, default=DEFAULT_RANK_DEPTH)
    parser.add_argument("--bm25-candidate-k", type=int, default=DEFAULT_HYBRID_BM25_CANDIDATE_K)
    parser.add_argument("--bm25-tokenizer", choices=("kiwi", "legacy"), default="kiwi")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_HYBRID_RRF_K)
    parser.add_argument(
        "--dense-weights",
        type=parse_weights,
        default=parse_weights("0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8"),
    )
    parser.add_argument("--selected-dense-weight", type=float, default=DEFAULT_HYBRID_DENSE_WEIGHT)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-details", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="청년정책 retrieval 평가 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_arguments(subparsers.add_parser("run", help="Retriever를 직접 실행해 평가"))
    add_sweep_arguments(subparsers.add_parser("sweep", help="캐시된 dense 결과로 hybrid 가중치 탐색"))
    args = parser.parse_args()

    if args.command == "run":
        if args.rank_depth < 1:
            parser.error("--rank-depth는 1 이상이어야 합니다.")
        if args.retrieval_mode != "bm25" and not (args.provider and args.model):
            parser.error("dense/hybrid 모드에는 --provider와 --model이 필요합니다.")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit은 1 이상이어야 합니다.")
        if args.max_concurrency < 1:
            parser.error("--max-concurrency는 1 이상이어야 합니다.")
        if args.planner_rrf_k < 1 or args.hybrid_rrf_k < 1:
            parser.error("RRF k는 1 이상이어야 합니다.")
        if args.bm25_candidate_k < args.rank_depth:
            parser.error("--bm25-candidate-k는 --rank-depth 이상이어야 합니다.")
        if not 0 <= args.dense_weight <= 1:
            parser.error("--dense-weight는 0과 1 사이여야 합니다.")
        if args.tracking == "local" and (args.planner_query_cache or args.reranker_model):
            parser.error("Planner/reranker 실험은 --tracking langfuse에서 실행하세요.")
        if args.tracking == "langfuse" and args.limit is not None:
            parser.error("--limit은 local smoke test에서만 사용할 수 있습니다.")
        if args.tracking == "langfuse" and not args.experiment_name:
            parser.error("--tracking langfuse에는 --experiment-name이 필요합니다.")
        if args.planner_query_mode == "rrf" and not args.planner_query_cache:
            parser.error("--planner-query-mode rrf에는 --planner-query-cache가 필요합니다.")
        if args.planner_query_mode == "rrf" and args.retrieval_mode != "dense":
            parser.error("Planner multi-query RRF는 dense 모드에서만 지원합니다.")
        if args.planner_query_mode == "rrf" and args.reranker_model:
            parser.error("현재 Planner RRF와 reranker를 동시에 사용할 수 없습니다.")
        if args.reranker_model and args.retrieval_mode != "dense":
            parser.error("reranker 실험은 dense 모드에서만 지원합니다.")
    else:
        if args.rank_depth < 1 or args.bm25_candidate_k < args.rank_depth:
            parser.error("sweep 후보 깊이는 rank depth 이상이어야 합니다.")
        if args.rrf_k < 1:
            parser.error("--rrf-k는 1 이상이어야 합니다.")
        if not 0 <= args.selected_dense_weight <= 1:
            parser.error("--selected-dense-weight는 0과 1 사이어야 합니다.")
    try:
        if args.today:
            datetime.strptime(args.today, "%Y%m%d")
    except ValueError:
        parser.error("--today는 유효한 YYYYMMDD 형식이어야 합니다.")
    return args


def write_local_results(
    *,
    summary: dict,
    details: list[dict],
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


def prepare_run(args: argparse.Namespace):
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
    return dataset_path, chroma_dir, cases, collection, retriever, evaluation_today


def run_local(args: argparse.Namespace) -> None:
    dataset_path, chroma_dir, cases, collection, retriever, _ = prepare_run(args)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result, details = evaluate_retrieval(
        collection=collection,
        retriever=retriever,
        cases=cases,
        rank_depth=args.rank_depth,
        exclude_expired=True if args.exclude_expired else None,
        today_yyyymmdd=int(args.today) if args.today else None,
    )
    experiment_name = args.experiment_name or (
        "bm25"
        if args.retrieval_mode == "bm25"
        else f"{args.retrieval_mode}-{args.provider}-{args.model}"
    )
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
            "duration_seconds": time.perf_counter() - started,
        },
        **result,
    }
    summary_path, details_path = write_local_results(
        summary=summary,
        details=details,
        output_dir=project_path(args.output_dir),
        experiment_name=experiment_name,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")


def run_langfuse(args: argparse.Namespace) -> None:
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        raise RuntimeError("LANGFUSE_PUBLIC_KEY와 LANGFUSE_SECRET_KEY가 필요합니다.")
    os.environ["LANGFUSE_TRACING"] = "true"
    dataset_path, chroma_dir, cases, _, retriever, evaluation_today = prepare_run(args)
    planner_records = (
        load_planner_query_records(project_path(args.planner_query_cache), cases)
        if args.planner_query_cache
        else None
    )
    reranker = (
        LlamaCppReranker(
            base_url=args.reranker_base_url,
            model=args.reranker_model,
            timeout_seconds=args.reranker_timeout_seconds,
        )
        if args.reranker_model
        else None
    )
    config = load_config()
    langfuse = initialize_langfuse(config)
    if langfuse is None:
        from langfuse import get_client

        langfuse = get_client()
    try:
        dataset = ensure_retrieval_dataset(
            langfuse,
            dataset_name=args.langfuse_dataset_name,
            dataset_path=dataset_path,
            cases=cases,
        )
        item_evaluators = [
            build_recall_evaluator(3),
            build_recall_evaluator(5),
            build_recall_evaluator(10),
            reciprocal_rank_evaluator,
        ]
        run_evaluators = [
            build_mean_run_evaluator(item_metric_name="recall_at_3", run_metric_name="recall_at_3"),
            build_mean_run_evaluator(item_metric_name="recall_at_5", run_metric_name="recall_at_5"),
            build_mean_run_evaluator(item_metric_name="recall_at_10", run_metric_name="recall_at_10"),
            build_mean_run_evaluator(item_metric_name="reciprocal_rank", run_metric_name="mrr"),
        ]
        if planner_records is not None:
            item_evaluators.extend([planner_retriever_route_evaluator, planner_raw_fallback_evaluator])
            run_evaluators.extend([
                build_mean_run_evaluator(item_metric_name="planner_retriever_route", run_metric_name="planner_retriever_route_rate"),
                build_mean_run_evaluator(item_metric_name="planner_raw_fallback", run_metric_name="planner_raw_fallback_rate"),
            ])
        if reranker is not None:
            item_evaluators.append(reranker_latency_evaluator)
            run_evaluators.append(build_mean_run_evaluator(
                item_metric_name="reranker_latency_ms",
                run_metric_name="mean_reranker_latency_ms",
            ))
        results = dataset.run_experiment(
            name=args.experiment_name,
            description="PolicyRetriever Recall@3/5/10 및 MRR deterministic 평가",
            task=build_retrieval_task(
                retriever,
                planner_records,
                planner_query_mode=args.planner_query_mode,
                rrf_k=args.planner_rrf_k,
                reranker=reranker,
            ),
            evaluators=item_evaluators,
            run_evaluators=run_evaluators,
            max_concurrency=args.max_concurrency,
            metadata={
                "dataset_name": args.langfuse_dataset_name,
                "dataset_path": str(dataset_path),
                "retriever_class": type(retriever).__name__,
                "retrieval_mode": args.retrieval_mode,
                "retriever_provider": args.provider,
                "retriever_query_model": args.model,
                "chroma_dir": str(chroma_dir),
                "collection": args.collection,
                "rank_depth": args.rank_depth,
                "bm25_candidate_k": args.bm25_candidate_k,
                "bm25_tokenizer": args.bm25_tokenizer,
                "dense_weight": args.dense_weight,
                "hybrid_rrf_k": args.hybrid_rrf_k,
                "reranker_model": args.reranker_model,
                "today": evaluation_today.isoformat(),
                "query_mode": f"planner_{args.planner_query_mode}" if planner_records else "raw",
                "planner_query_cache": str(project_path(args.planner_query_cache)) if args.planner_query_cache else None,
            },
        )
        print(results.format())
        if results.dataset_run_url:
            print(f"Langfuse dataset run: {results.dataset_run_url}")
        if args.output_details:
            output_details = project_path(args.output_details)
            output_details.parent.mkdir(parents=True, exist_ok=True)
            with output_details.open("w", encoding="utf-8") as details_file:
                for item_result in results.item_results:
                    metadata = item_value(item_result.item, "metadata", {}) or {}
                    expected = item_value(item_result.item, "expected_output", {}) or {}
                    details_file.write(json.dumps({
                        "case_id": metadata.get("case_id"),
                        "expected_policy_ids": expected.get("expected_policy_ids", []),
                        "output": item_result.output,
                        "evaluations": {
                            evaluation.name: evaluation.value
                            for evaluation in item_result.evaluations
                        },
                        "trace_id": item_result.trace_id,
                    }, ensure_ascii=False) + "\n")
            print(f"Local details: {output_details}")
    finally:
        shutdown_langfuse()


def run_sweep(args: argparse.Namespace) -> None:
    cases = load_evaluation_cases(project_path(args.dataset))
    planner_records = load_planner_query_records(project_path(args.planner_query_cache), cases)
    dense_details_path = project_path(args.dense_details)
    summary, details = evaluate_cached_hybrid_sweep(
        collection=open_collection(project_path(args.chroma_dir), args.collection),
        cases=cases,
        planner_records=planner_records,
        dense_details=load_dense_details(dense_details_path),
        evaluation_today=datetime.strptime(args.today, "%Y%m%d").date(),
        bm25_candidate_k=args.bm25_candidate_k,
        rrf_k=args.rrf_k,
        dense_weights=args.dense_weights,
        selected_dense_weight=args.selected_dense_weight,
        rank_depth=args.rank_depth,
        bm25_tokenizer=args.bm25_tokenizer,
    )
    summary["dense_candidate_source"] = str(dense_details_path)
    output_summary = project_path(args.output_summary)
    output_details = project_path(args.output_details)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_details.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_details.open("w", encoding="utf-8") as details_file:
        for row in details:
            details_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.command == "sweep":
        run_sweep(args)
    elif args.tracking == "langfuse":
        run_langfuse(args)
    else:
        os.environ["LANGFUSE_TRACING"] = "false"
        run_local(args)


if __name__ == "__main__":
    main()
