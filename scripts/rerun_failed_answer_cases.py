"""Run reusable answer-failure regression cases against the live RAG graph."""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from src.config import load_config
from src.factory import build_rag_graph
from src.observability import create_observability_runtime
from src.rag.state import (
    CHECKER_REASONING_METADATA_KEY,
    CHECKER_VERDICT_METADATA_KEY,
    RAGUserProfile,
)


DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "docs/failure_regression_cases.yaml"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/eval/failure_regression_runs"
TRACE_TAGS = [
    "youth-policy-rag",
    "qualitative-eval",
    "answer-failure-regression",
]


class AutomatedChecks(BaseModel):
    expected_retrieval_count: int | None = Field(default=None, ge=0)
    min_retrieval_count: int | None = Field(default=None, ge=0)
    max_retrieval_count: int | None = Field(default=None, ge=0)
    preserve_previous_policy_ids: bool = False
    require_new_policy_ids_if_selected: bool = False
    no_duplicate_policy_ids: bool = False
    no_duplicate_policy_titles: bool = False
    forbidden_policy_title_terms: list[str] = Field(default_factory=list)


class CaseTurn(BaseModel):
    turn: int = Field(ge=1)
    question: str = Field(min_length=1)
    automated_checks: AutomatedChecks = Field(
        default_factory=AutomatedChecks
    )
    review_criteria: list[str] = Field(default_factory=list)


class HistoricalFailure(BaseModel):
    layer: str
    summary: str


class FailureCase(BaseModel):
    case_id: str
    title: str
    historical_failure: HistoricalFailure
    profile: RAGUserProfile
    exclude_expired: bool = True
    turns: list[CaseTurn] = Field(min_length=1)


class FailureSuite(BaseModel):
    schema_version: int
    suite_id: str
    description: str
    cases: list[FailureCase] = Field(min_length=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="재실행할 YAML 회귀 케이스 파일",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="재실행할 case_id. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="trace metadata, thread ID, 결과 파일에 사용할 실행 식별자",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="결과 JSON 경로. 생략하면 failure_regression_runs 아래에 저장",
    )
    parser.add_argument(
        "--fail-on-automated-check",
        action="store_true",
        help="자동 구조 검사가 하나라도 실패하면 종료 코드 1을 반환",
    )
    parser.add_argument(
        "--include-expired",
        action="store_true",
        help="진단용으로 케이스의 exclude_expired 값을 false로 덮어씁니다.",
    )
    return parser.parse_args()


def load_suite(path: Path) -> FailureSuite:
    with path.open(encoding="utf-8") as file:
        return FailureSuite.model_validate(yaml.safe_load(file))


def select_cases(
    suite: FailureSuite,
    case_ids: set[str],
) -> list[FailureCase]:
    available_ids = {case.case_id for case in suite.cases}
    unknown_case_ids = case_ids - available_ids
    if unknown_case_ids:
        raise ValueError(f"알 수 없는 case_id: {sorted(unknown_case_ids)}")
    return [
        case
        for case in suite.cases
        if not case_ids or case.case_id in case_ids
    ]


def policy_title(document: Document) -> str:
    metadata_title = str(document.metadata.get("plcyNm") or "").strip()
    if metadata_title:
        return metadata_title
    first_line = document.page_content.splitlines()[0].strip()
    return re.sub(r"^정책명:\s*", "", first_line).strip()


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", "", title).casefold()


def evaluate_automated_checks(
    checks: AutomatedChecks,
    *,
    retrieval_count: int,
    policy_ids: list[str],
    policy_titles: list[str],
    previous_policy_ids: list[str] | None,
) -> list[dict]:
    results = []

    def add(name: str, passed: bool, expected, actual) -> None:
        results.append({
            "name": name,
            "passed": passed,
            "expected": expected,
            "actual": actual,
        })

    if checks.expected_retrieval_count is not None:
        add(
            "expected_retrieval_count",
            retrieval_count == checks.expected_retrieval_count,
            checks.expected_retrieval_count,
            retrieval_count,
        )
    if checks.min_retrieval_count is not None:
        add(
            "min_retrieval_count",
            retrieval_count >= checks.min_retrieval_count,
            f">= {checks.min_retrieval_count}",
            retrieval_count,
        )
    if checks.max_retrieval_count is not None:
        add(
            "max_retrieval_count",
            retrieval_count <= checks.max_retrieval_count,
            f"<= {checks.max_retrieval_count}",
            retrieval_count,
        )
    if checks.preserve_previous_policy_ids:
        add(
            "preserve_previous_policy_ids",
            bool(previous_policy_ids)
            and policy_ids == previous_policy_ids,
            previous_policy_ids,
            policy_ids,
        )
    if checks.require_new_policy_ids_if_selected:
        new_policy_ids = [
            policy_id
            for policy_id in policy_ids
            if policy_id not in (previous_policy_ids or [])
        ]
        add(
            "require_new_policy_ids_if_selected",
            not policy_ids
            or (bool(previous_policy_ids) and bool(new_policy_ids)),
            (
                "no policies selected or at least one policy ID not selected "
                "in the previous turn"
            ),
            new_policy_ids,
        )
    if checks.no_duplicate_policy_ids:
        add(
            "no_duplicate_policy_ids",
            len(policy_ids) == len(set(policy_ids)),
            "all unique",
            policy_ids,
        )
    if checks.no_duplicate_policy_titles:
        normalized_titles = [
            normalize_title(title)
            for title in policy_titles
            if title
        ]
        add(
            "no_duplicate_policy_titles",
            len(normalized_titles) == len(set(normalized_titles)),
            "all unique",
            policy_titles,
        )
    if checks.forbidden_policy_title_terms:
        normalized_terms = [
            term.casefold().strip()
            for term in checks.forbidden_policy_title_terms
            if term.strip()
        ]
        matched_titles = [
            title
            for title in policy_titles
            if any(term in title.casefold() for term in normalized_terms)
        ]
        add(
            "forbidden_policy_title_terms",
            not matched_titles,
            f"no titles containing {checks.forbidden_policy_title_terms}",
            matched_titles,
        )
    return results


def checked_policy_summary(state: dict) -> list[dict]:
    return [
        {
            "policy_id": str(
                item["document"].metadata.get("plcyNo") or ""
            ),
            "verdict": item["verdict"],
            "reasoning": item["reasoning"],
            "retrieval_rank": item.get("retrieval_rank", 1),
            "retrieval_round": item.get("retrieval_round", 1),
        }
        for item in state.get("checked_policies", [])
    ]


def selected_policy_summary(documents: list[Document]) -> list[dict]:
    return [
        {
            "policy_id": str(document.metadata.get("plcyNo") or ""),
            "title": policy_title(document),
            "checker_verdict": document.metadata.get(
                CHECKER_VERDICT_METADATA_KEY
            ),
            "checker_reasoning": document.metadata.get(
                CHECKER_REASONING_METADATA_KEY
            ),
        }
        for document in documents
    ]


async def run_suite(
    *,
    cases: list[FailureCase],
    run_id: str,
    results: list[dict] | None = None,
    exclude_expired_override: bool | None = None,
) -> list[dict]:
    if results is None:
        results = []
    config = load_config()
    observability = create_observability_runtime(config)
    graph = None
    try:
        graph = build_rag_graph(
            config,
            trace_config_factory=observability.build_trace_config,
        )
        for case in cases:
            exclude_expired = (
                case.exclude_expired
                if exclude_expired_override is None
                else exclude_expired_override
            )
            thread_id = (
                f"failure-regression-{run_id}-{case.case_id.lower()}"
            )
            previous_policy_ids = None
            for turn in case.turns:
                result = await graph.agenerate_answer(
                    user_input=turn.question,
                    user_profile=case.profile,
                    thread_id=thread_id,
                    exclude_expired=exclude_expired,
                    trace_user_id=(
                        f"failure-regression-{case.case_id.lower()}"
                    ),
                    trace_tags=TRACE_TAGS,
                    trace_metadata={
                        "evaluation_case_id": case.case_id,
                        "evaluation_turn": turn.turn,
                        "evaluation_run": run_id,
                    },
                )
                snapshot = await graph.graph.aget_state({
                    "configurable": {"thread_id": thread_id}
                })
                state = snapshot.values
                documents = list(state.get("documents", []))
                selected_policies = selected_policy_summary(documents)
                policy_ids = [
                    policy["policy_id"]
                    for policy in selected_policies
                    if policy["policy_id"]
                ]
                policy_titles = [
                    policy["title"]
                    for policy in selected_policies
                    if policy["title"]
                ]
                checks = evaluate_automated_checks(
                    turn.automated_checks,
                    retrieval_count=state.get("retrieval_count", 0),
                    policy_ids=policy_ids,
                    policy_titles=policy_titles,
                    previous_policy_ids=previous_policy_ids,
                )
                turn_result = {
                    "case_id": case.case_id,
                    "case_title": case.title,
                    "historical_failure": (
                        case.historical_failure.model_dump()
                    ),
                    "turn": turn.turn,
                    "thread_id": thread_id,
                    "question": turn.question,
                    "profile": dict(case.profile),
                    "exclude_expired": exclude_expired,
                    "retrieval_count": state.get("retrieval_count", 0),
                    "retrieval_query": state.get("retrieval_query", ""),
                    "selected_policies": selected_policies,
                    "checked_policies": checked_policy_summary(state),
                    "answer": result.answer,
                    "automated_checks": checks,
                    "automated_status": (
                        "PASS"
                        if all(check["passed"] for check in checks)
                        else "FAIL"
                    ),
                    "review_criteria": turn.review_criteria,
                }
                results.append(turn_result)
                print(json.dumps(turn_result, ensure_ascii=False))
                previous_policy_ids = policy_ids
    finally:
        if graph is not None:
            graph.close()
        observability.shutdown()
    return results


async def main() -> None:
    args = parse_args()
    suite = load_suite(args.cases_path)
    cases = select_cases(suite, set(args.case_ids or ()))
    output = args.output or (
        DEFAULT_OUTPUT_DIR / f"{args.run_id}.json"
    )
    results = []
    execution_error = None
    try:
        await run_suite(
            cases=cases,
            run_id=args.run_id,
            results=results,
            exclude_expired_override=(
                False if args.include_expired else None
            ),
        )
    except Exception as error:
        execution_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "suite_id": suite.suite_id,
                    "schema_version": suite.schema_version,
                    "run_id": args.run_id,
                    "cases_path": str(args.cases_path),
                    "execution_error": execution_error,
                    "automated_summary": {
                        "passed_turns": sum(
                            result["automated_status"] == "PASS"
                            for result in results
                        ),
                        "failed_turns": sum(
                            result["automated_status"] == "FAIL"
                            for result in results
                        ),
                        "total_turns": len(results),
                    },
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    print(f"result_path={output}")

    if (
        args.fail_on_automated_check
        and any(result["automated_status"] == "FAIL" for result in results)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
