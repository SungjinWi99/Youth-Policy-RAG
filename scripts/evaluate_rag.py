"""Run end-to-end RAG quality evaluation in Langfuse."""

import os
import sys
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client
from tqdm import tqdm

from src.config import load_config
from src.observability import initialize_langfuse, shutdown_langfuse
from src.evaluation.langfuse import (
    is_not_found_error,
    item_value,
    stable_dataset_item_id,
)
from src.evaluation.rag import build_langfuse_evaluators, load_evaluation_items
from src.factory import build_rag_graph, create_chat_model


def get_langfuse_ui_url() -> str:
    return os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")


def build_evaluator_llm(config):
    return create_chat_model(
        provider=config.evaluation.provider,
        model_name=config.evaluation.model,
        temperature=0,
    )


class EvaluationProgress:
    def __init__(
        self,
        *,
        total_cases: int,
        evaluator_count: int,
        case_id_by_question: dict[str, str],
    ):
        self.total_cases = total_cases
        self.evaluator_count = evaluator_count
        self.case_id_by_question = case_id_by_question
        self.rag_started = 0
        self.rag_completed = 0
        self.evaluator_completed = 0
        self.lock = Lock()

    def log(self, message: str) -> None:
        with self.lock:
            print(message, file=sys.stderr, flush=True)

    def start_case(self, case_id: str, question: str) -> None:
        with self.lock:
            self.rag_started += 1
            index = self.rag_started
            short_question = question.replace("\n", " ")[:80]
            print(
                f"[RAG {index}/{self.total_cases}] start {case_id}: {short_question}",
                file=sys.stderr,
                flush=True,
            )

    def finish_case(self, case_id: str, retrieved_count: int) -> None:
        with self.lock:
            self.rag_completed += 1
            print(
                f"[RAG {self.rag_completed}/{self.total_cases}] done  {case_id} "
                f"(retrieved={retrieved_count})",
                file=sys.stderr,
                flush=True,
            )

    def fail_case(self, case_id: str, error: Exception) -> None:
        with self.lock:
            self.rag_completed += 1
            print(
                f"[RAG {self.rag_completed}/{self.total_cases}] fail  {case_id}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

    def finish_evaluator(self, case_id: str, metric_name: str, score: Any) -> None:
        with self.lock:
            self.evaluator_completed += 1
            total = self.total_cases * self.evaluator_count
            print(
                f"[EVAL {self.evaluator_completed}/{total}] {case_id} "
                f"{metric_name}={score}",
                file=sys.stderr,
                flush=True,
            )


def ensure_dataset(langfuse, dataset_name: str, example_path: str):
    try:
        langfuse.get_dataset(dataset_name)
    except Exception as error:
        if not is_not_found_error(error):
            raise
        langfuse.create_dataset(
            name=dataset_name,
            description="청년정책 RAG Langfuse 평가 데이터셋",
            metadata={"source_path": example_path},
        )

    examples = load_evaluation_items(example_path)
    print(
        f"Syncing {len(examples)} dataset items to Langfuse dataset "
        f"{dataset_name!r}",
        file=sys.stderr,
        flush=True,
    )
    for example in tqdm(
        examples,
        desc="Langfuse dataset sync",
        unit="item",
        file=sys.stderr,
    ):
        case_id = example["case_id"]
        metadata = {
            **example.get("metadata", {}),
            "case_id": case_id,
            "source_path": example_path,
        }
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=stable_dataset_item_id(
                dataset_name,
                case_id,
                namespace="",
            ),
            input=example["input"],
            expected_output=example["expected_output"],
            metadata=metadata,
        )

    return langfuse.get_dataset(dataset_name)


def run_rag_target(*, item, rag, dataset_name: str, **kwargs) -> dict:
    inputs = item_value(item, "input", {}) or {}
    metadata = item_value(item, "metadata", {}) or {}
    case_id = metadata.get("case_id") or item_value(item, "id", "unknown")
    thread_id = f"eval:{dataset_name}:{case_id}"
    result = rag.generate_answer(
        user_input=inputs["question"],
        user_profile=inputs.get("user_profile", {}),
        exclude_expired=inputs.get("exclude_expired", True),
        thread_id=thread_id,
        trace_user_id="evaluation",
        trace_tags=["evaluation", "youth-policy-rag"],
        trace_metadata={
            "case_id": case_id,
            "dataset_name": dataset_name,
        },
    )
    return result.model_dump()


def _case_id_from_item(item: Any) -> str:
    metadata = item_value(item, "metadata", {}) or {}
    return metadata.get("case_id") or str(item_value(item, "id", "unknown"))


def _case_id_from_evaluator_kwargs(
    kwargs: dict,
    progress: EvaluationProgress,
) -> str:
    item = kwargs.get("item")
    if item is not None:
        return _case_id_from_item(item)

    metadata = kwargs.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("case_id"):
        return metadata["case_id"]

    inputs = kwargs.get("input") or {}
    if isinstance(inputs, dict):
        case_id = progress.case_id_by_question.get(inputs.get("question"))
        if case_id:
            return case_id

    return "unknown"


def _wrap_task_with_progress(*, rag, dataset_name: str, progress: EvaluationProgress):
    def task(*, item, **kwargs):
        inputs = item_value(item, "input", {}) or {}
        case_id = _case_id_from_item(item)
        progress.start_case(case_id, inputs.get("question", ""))
        try:
            output = run_rag_target(
                item=item,
                rag=rag,
                dataset_name=dataset_name,
                **kwargs,
            )
        except Exception as error:
            progress.fail_case(case_id, error)
            raise

        progress.finish_case(
            case_id,
            len(output.get("retrieved_policy_ids", [])),
        )
        return output

    return task


def _wrap_evaluator_with_progress(evaluator, progress: EvaluationProgress):
    def wrapped(**kwargs):
        result = evaluator(**kwargs)
        case_id = _case_id_from_evaluator_kwargs(kwargs, progress)
        progress.finish_evaluator(
            case_id,
            getattr(result, "name", evaluator.__name__),
            getattr(result, "value", None),
        )
        return result

    wrapped.__name__ = evaluator.__name__
    return wrapped


def main():
    load_dotenv()
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        os.environ.setdefault("LANGFUSE_TRACING", "true")

    config = load_config()
    langfuse = initialize_langfuse(config) or get_client()
    dataset = ensure_dataset(
        langfuse,
        config.evaluation.dataset_name,
        config.path(config.evaluation.example_path),
    )
    rag = build_rag_graph(config)
    evaluator_llm = build_evaluator_llm(config)
    evaluators = build_langfuse_evaluators(evaluator_llm)
    evaluation_items = load_evaluation_items(
        config.path(config.evaluation.example_path)
    )
    progress = EvaluationProgress(
        total_cases=len(evaluation_items),
        evaluator_count=len(evaluators),
        case_id_by_question={
            item["input"]["question"]: item["case_id"]
            for item in evaluation_items
        },
    )
    experiment_name = (
        f"{config.evaluation.experiment_prefix}-"
        f"{config.evaluation.dataset_name}"
    )

    try:
        progress.log(
            f"Starting Langfuse experiment {experiment_name!r} "
            f"({len(evaluation_items)} cases, "
            f"max_concurrency={config.evaluation.max_concurrency})"
        )
        results = dataset.run_experiment(
            name=experiment_name,
            description=(
                "청년정책 RAG의 Context Recall, Context Average Helpfulness, "
                "Faithfulness, Answer Relevance 평가"
            ),
            task=_wrap_task_with_progress(
                rag=rag,
                dataset_name=config.evaluation.dataset_name,
                progress=progress,
            ),
            evaluators=[
                _wrap_evaluator_with_progress(evaluator, progress)
                for evaluator in evaluators
            ],
            max_concurrency=config.evaluation.max_concurrency,
            metadata={
                "dataset_name": config.evaluation.dataset_name,
                "example_path": config.evaluation.example_path,
                "llm_provider": config.llm.provider,
                "llm_model": config.llm.model,
                "retriever_provider": config.retriever.provider,
                "retriever_query_model": config.retriever.query_model,
                "retriever_passage_model": config.retriever.passage_model,
                "evaluator_provider": config.evaluation.provider,
                "evaluator_model": config.evaluation.model,
            },
        )
        print(results.format())
        print(f"Langfuse UI: {get_langfuse_ui_url()}")
    finally:
        rag.close()
        shutdown_langfuse()


if __name__ == "__main__":
    main()
