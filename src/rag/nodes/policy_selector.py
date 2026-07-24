from src.rag.state import (
    CHECKER_REASONING_METADATA_KEY,
    CHECKER_VERDICT_METADATA_KEY,
    CheckedPolicy,
    RAGGraphState,
)


ACCEPTED_VERDICTS = frozenset({
    "direct_fit",
    "fit_needs_clarification",
})
VERDICT_PRIORITY = {
    "direct_fit": 0,
    "fit_needs_clarification": 1,
    "indirect": 2,
    "mismatch": 3,
}


def _policy_key(item: CheckedPolicy) -> str:
    document = item["document"]
    policy_id = str(document.metadata.get("plcyNo") or "").strip()
    return policy_id or f"document:{id(document)}"


def _latest_checks(
    checked_policies: list[CheckedPolicy],
) -> list[CheckedPolicy]:
    latest_by_policy = {}
    for item in sorted(
        checked_policies,
        key=lambda checked: (
            checked.get("retrieval_round", 1),
            checked.get("retrieval_rank", 1),
        ),
    ):
        latest_by_policy[_policy_key(item)] = item
    return list(latest_by_policy.values())


def _annotate_document(item: CheckedPolicy):
    document = item["document"]
    return document.model_copy(update={
        "metadata": {
            **document.metadata,
            CHECKER_VERDICT_METADATA_KEY: item["verdict"],
            CHECKER_REASONING_METADATA_KEY: item["reasoning"],
        }
    })


def make_policy_selector_node():
    def policy_selector_node(state: RAGGraphState) -> dict:
        accepted = []
        seen_policy_ids = set()
        for item in sorted(
            _latest_checks(state.get("checked_policies", [])),
            key=lambda checked: (
                VERDICT_PRIORITY[checked["verdict"]],
                checked.get("retrieval_round", 1),
                checked.get("retrieval_rank", 1),
            ),
        ):
            if item["verdict"] not in ACCEPTED_VERDICTS:
                continue
            document = item["document"]
            policy_id = str(document.metadata.get("plcyNo") or "").strip()
            dedupe_key = policy_id or id(document)
            if dedupe_key in seen_policy_ids:
                continue
            seen_policy_ids.add(dedupe_key)
            accepted.append(_annotate_document(item))

        if accepted:
            reason = (
                "Checker verdict가 direct_fit 또는 "
                f"fit_needs_clarification인 정책 {len(accepted)}개를 선택했습니다."
            )
        else:
            reason = "Checker verdict를 통과한 정책이 없습니다."
        return {
            "documents": accepted,
            **({"active_policies": accepted} if accepted else {}),
            "selection_reason": reason,
        }

    return policy_selector_node
