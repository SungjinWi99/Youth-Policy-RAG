"""Run end-to-end RAG quality evaluation locally."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config import load_config
from src.evaluation.rag import build_evaluators, load_evaluation_items
from src.factory import build_rag_graph, create_chat_model


def build_evaluator_llm(config):
    return create_chat_model(
        provider=config.evaluation.provider,
        model_name=config.evaluation.model,
        temperature=0,
    )


def evaluate_case(
    *,
    item: dict[str, Any],
    output: dict[str, Any],
    evaluators: dict[str, Any],
) -> list[dict[str, Any]]:
    inputs = item["input"]
    expected_output = item["expected_output"]
    return [
        evaluators["context_recall"](
            outputs=output,
            reference_outputs=expected_output,
        ),
        evaluators["context_average_helpfulness"](
            inputs=inputs,
            outputs=output,
        ),
        evaluators["faithfulness"](
            inputs=inputs,
            outputs=output,
        ),
        evaluators["answer_relevance"](
            inputs=inputs,
            outputs=output,
        ),
    ]


def main() -> None:
    load_dotenv()
    config = load_config()
    items = load_evaluation_items(config.path(config.evaluation.example_path))
    rag = build_rag_graph(config)
    evaluator_llm = build_evaluator_llm(config)
    evaluators = {evaluator.__name__: evaluator for evaluator in build_evaluators(evaluator_llm)}
    results: list[dict[str, Any]] = []
    scores: dict[str, list[float]] = defaultdict(list)
    try:
        for index, item in enumerate(items, start=1):
            case_id = item["case_id"]
            print(f"[RAG {index}/{len(items)}] {case_id}", file=sys.stderr, flush=True)
            output = rag.generate_answer(
                user_input=item["input"]["question"],
                user_profile=item["input"].get("user_profile", {}),
                exclude_expired=item["input"].get("exclude_expired", True),
                thread_id=f"eval:{config.evaluation.dataset_name}:{case_id}",
            ).model_dump()
            evaluations = evaluate_case(item=item, output=output, evaluators=evaluators)
            for evaluation in evaluations:
                scores[evaluation["key"]].append(float(evaluation["score"]))
            results.append({
                "case_id": case_id,
                "output": output,
                "evaluations": evaluations,
            })
    finally:
        rag.close()

    summary = {
        "dataset_name": config.evaluation.dataset_name,
        "case_count": len(results),
        "metrics": {
            name: sum(values) / len(values)
            for name, values in scores.items()
        },
    }
    output_path = Path(config.path("data/eval/rag_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
