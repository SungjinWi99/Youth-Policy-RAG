import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.datasets import (
    DEFAULT_DATASET_PATH,
    load_evaluation_cases,
    project_path,
)
from src.evaluation.models import PlannerQueryRecord
from src.factory import create_chat_model
from src.rag.nodes.retrieval_planner import (
    PLANNER_HUMAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    make_retrieval_planner_node,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/eval/planner_query_results"
    / "eval_v1_200-deepseek-v4-flash.jsonl"
)


def planner_prompt_sha256() -> str:
    prompt = f"{PLANNER_SYSTEM_PROMPT}\n{PLANNER_HUMAN_PROMPT}"
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as cache_file:
        for line_number, line in enumerate(cache_file, start=1):
            if not line.strip():
                continue
            try:
                record = PlannerQueryRecord.model_validate_json(line)
                completed.add(record.case_id)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} Planner cache가 유효하지 않습니다."
                ) from error
    return completed


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(
        description=(
            "평가 데이터의 Retrieval Planner 출력을 JSONL로 고정합니다."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--provider", default=config.llm.provider)
    parser.add_argument("--model", default=config.llm.model)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts는 1 이상이어야 합니다.")
    return args


def main() -> None:
    args = parse_args()
    config = load_config()
    load_dotenv()
    dataset_path = project_path(args.dataset)
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_evaluation_cases(dataset_path)
    completed_case_ids = load_completed_case_ids(output_path)
    unknown_case_ids = completed_case_ids - {case.case_id for case in cases}
    if unknown_case_ids:
        raise ValueError(
            "Planner cache에 현재 데이터셋에 없는 case_id가 있습니다: "
            f"{sorted(unknown_case_ids)[:5]}"
        )

    llm = create_chat_model(
        provider=args.provider,
        model_name=args.model,
        temperature=0,
    )
    planner = make_retrieval_planner_node(
        llm,
        config.rag.planner.history_window,
    )
    prompt_hash = planner_prompt_sha256()
    remaining_cases = [
        case for case in cases if case.case_id not in completed_case_ids
    ]

    with output_path.open("a", encoding="utf-8") as output_file:
        for case in tqdm(
            remaining_cases,
            desc="Freeze Retrieval Planner queries",
        ):
            last_error = None
            for attempt in range(1, args.max_attempts + 1):
                try:
                    plan = planner.invoke({
                        "user_input": case.user_input,
                        "user_profile": case.user_profile,
                        "exclude_expired": case.exclude_expired,
                        "messages": [],
                        "documents": [],
                        "retrieval_count": 0,
                        "retrieved_policies": [],
                        "checked_policies": [],
                    })
                    break
                except Exception as error:
                    last_error = error
                    if attempt == args.max_attempts:
                        raise RuntimeError(
                            f"Planner 생성 실패: {case.case_id}"
                        ) from error
                    time.sleep(attempt)
            else:
                raise RuntimeError(
                    f"Planner 생성 실패: {case.case_id}"
                ) from last_error

            row = {
                "schema_version": 2,
                "case_id": case.case_id,
                "raw_query": case.user_input,
                "user_profile": case.user_profile,
                "user_requirement": plan["user_requirement"],
                "needs_retrieval": plan["needs_retrieval"],
                "retrieval_reason": plan["retrieval_reason"],
                "retrieval_query": plan["retrieval_query"],
                "planner_provider": args.provider,
                "planner_model": args.model,
                "planner_prompt_sha256": prompt_hash,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()

    print(f"Planner query cache: {output_path}")
    print(f"Cases: {len(cases)}")
    print(f"Generated: {len(remaining_cases)}")
    print(f"Already cached: {len(completed_case_ids)}")


if __name__ == "__main__":
    main()
