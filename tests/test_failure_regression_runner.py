from pathlib import Path

import pytest

from scripts.rerun_failed_answer_cases import (
    AutomatedChecks,
    evaluate_automated_checks,
    load_suite,
    select_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_failure_regression_suite_loads_all_historical_failures():
    suite = load_suite(
        PROJECT_ROOT / "docs/failure_regression_cases.yaml"
    )

    assert suite.schema_version == 1
    assert [case.case_id for case in suite.cases] == [
        "AG-04",
        "AG-05",
        "AG-09",
        "AG-10",
        "AG-11",
        "AG-12",
        "AG-13",
    ]
    assert [turn.turn for turn in suite.cases[-1].turns] == [1, 2]


def test_select_cases_rejects_unknown_case_id():
    suite = load_suite(
        PROJECT_ROOT / "docs/failure_regression_cases.yaml"
    )

    with pytest.raises(ValueError, match="알 수 없는 case_id"):
        select_cases(suite, {"UNKNOWN"})


def test_automated_checks_detect_duplicates_and_lost_follow_up_policy():
    results = evaluate_automated_checks(
        AutomatedChecks(
            expected_retrieval_count=0,
            preserve_previous_policy_ids=True,
            no_duplicate_policy_ids=True,
            no_duplicate_policy_titles=True,
        ),
        retrieval_count=1,
        policy_ids=["A", "A"],
        policy_titles=["같은 정책", "같은 정책"],
        previous_policy_ids=["A", "B"],
    )

    assert {result["name"] for result in results} == {
        "expected_retrieval_count",
        "preserve_previous_policy_ids",
        "no_duplicate_policy_ids",
        "no_duplicate_policy_titles",
    }
    assert all(result["passed"] is False for result in results)


def test_automated_checks_detect_reused_policy_and_forbidden_title():
    results = evaluate_automated_checks(
        AutomatedChecks(
            min_retrieval_count=1,
            require_new_policy_ids_if_selected=True,
            forbidden_policy_title_terms=["매입임대", "임차보증금"],
        ),
        retrieval_count=1,
        policy_ids=["A", "B"],
        policy_titles=["청년 매입임대주택", "청년 월세 지원"],
        previous_policy_ids=["A", "B"],
    )

    assert results == [
        {
            "name": "min_retrieval_count",
            "passed": True,
            "expected": ">= 1",
            "actual": 1,
        },
        {
            "name": "require_new_policy_ids_if_selected",
            "passed": False,
            "expected": (
                "no policies selected or at least one policy ID not selected "
                "in the previous turn"
            ),
            "actual": [],
        },
        {
            "name": "forbidden_policy_title_terms",
            "passed": False,
            "expected": (
                "no titles containing ['매입임대', '임차보증금']"
            ),
            "actual": ["청년 매입임대주택"],
        },
    ]


def test_automated_checks_allow_safe_empty_new_search_result():
    results = evaluate_automated_checks(
        AutomatedChecks(require_new_policy_ids_if_selected=True),
        retrieval_count=1,
        policy_ids=[],
        policy_titles=[],
        previous_policy_ids=["A"],
    )

    assert results == [
        {
            "name": "require_new_policy_ids_if_selected",
            "passed": True,
            "expected": (
                "no policies selected or at least one policy ID not selected "
                "in the previous turn"
            ),
            "actual": [],
        }
    ]
