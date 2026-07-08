import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config
from src.factory import CHAT_MODEL_CLASSES, create_chat_model
from src.policy.utils import (
    REGION_CODES,
    REGION_NAME_TO_CODE,
    build_age_metadata,
    extract_sido_codes,
)


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/eval/eval_v1_500.jsonl"
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SEED = 42
DEFAULT_MIN_AGE = 18
DEFAULT_MAX_AGE = 39
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_POLICY_CHARS = 6000
DEFAULT_SAMPLING_STRATEGY = "stratified"
DEFAULT_DETAIL_DISTRIBUTION = (0.4, 0.4, 0.2)
DEFAULT_STYLE_DISTRIBUTION = (0.3, 0.3, 0.2, 0.2)
SUPPORTED_PROVIDERS = tuple(sorted(CHAT_MODEL_CLASSES))
DETAIL_LEVELS = (1, 2, 3)
QUESTION_STYLES = ("colloquial", "polite", "keyword", "situational")
UNCLASSIFIED_TAXONOMY = "미분류"

DETAIL_LEVEL_GUIDES = {
    1: {
        "instruction": (
            "사용자의 핵심 문제나 목적 하나만 짧게 표현하세요. 정책을 특정하는 "
            "세부 조건, 지원 규모, 신청 절차는 덧붙이지 마세요."
        ),
        "example": "면접에 입고 갈 옷이 없네",
    },
    2: {
        "instruction": (
            "사용자의 목적과 정책을 구분하는 핵심 특징 한 가지를 포함하세요. "
            "필요하면 궁금한 사항을 한 가지 덧붙이세요."
        ),
        "example": "취업 면접용 정장을 무료로 빌릴 수 있는 정책이 있나요?",
    },
    3: {
        "instruction": (
            "사용자의 목적과 정책을 구분하는 특징 두세 가지를 포함하세요. "
            "지원 방식, 이용 조건, 신청 방법 중 관련 있는 세부사항을 자연스럽게 "
            "물어보되 정책 설명을 그대로 복사하지 마세요."
        ),
        "example": (
            "취업 면접용 정장을 무료로 빌리고 싶은데, 대여 기간과 신청 방법도 알고 싶어요"
        ),
    },
}

STYLE_GUIDES = {
    "colloquial": {
        "instruction": (
            "친구에게 묻듯 자연스러운 구어체를 사용하세요. 짧은 문장이나 "
            "일상적인 표현은 허용하지만 과장된 유행어는 피하세요."
        ),
        "example": "면접 보러 가야 하는데 정장 빌릴 데 없을까?",
    },
    "polite": {
        "instruction": "검색 상담에서 사용할 법한 완결된 존댓말 질문으로 작성하세요.",
        "example": "면접용 정장을 대여할 수 있는 지원이 있나요?",
    },
    "keyword": {
        "instruction": (
            "검색창에 입력하는 핵심어 중심의 짧은 검색어 형태로 작성하세요. "
            "완전한 문장이나 물음표가 없어도 됩니다."
        ),
        "example": "취업 면접 정장 무료 대여 신청",
    },
    "situational": {
        "instruction": (
            "사용자가 처한 상황을 먼저 짧게 말한 뒤 필요한 도움을 묻는 "
            "상황 서술형 질문으로 작성하세요."
        ),
        "example": (
            "곧 취업 면접이 있는데 입을 정장이 없어요. 빌릴 수 있는 지원이 "
            "있을까요?"
        ),
    },
}

POLICY_FIELDS = (
    ("정책명", "plcyNm"),
    ("키워드", "plcyKywdNm"),
    ("대분류", "lclsfNm"),
    ("중분류", "mclsfNm"),
    ("정책 설명", "plcyExplnCn"),
    ("지원 내용", "plcySprtCn"),
    ("참여 대상", "ptcpPrpTrgtCn"),
    ("추가 신청 자격", "addAplyQlfcCndCn"),
    ("신청 기간", "aplyYmd"),
    ("신청 방법", "plcyAplyMthdCn"),
    ("제출 서류", "sbmsnDcmntCn"),
)

SYSTEM_PROMPT = """
당신은 청년정책 RAG 평가 질문을 만드는 데이터셋 작성자입니다.
정책 정보를 보고, 그 정책을 실제로 찾고 싶은 사용자가 검색창에 입력할 법한
single-turn 한국어 질문 하나를 작성하세요.

규칙:
1. 질문에 정책번호, 정책명, 기관명, URL을 그대로 쓰지 마세요.
2. 별도 user_profile로 제공되는 나이와 지역을 질문 문장에 쓰지 마세요.
3. 정책의 정답을 설명하지 말고 사용자의 필요나 궁금증으로 표현하세요.
4. 하나의 특정 정책을 gold로 의도하되, 여러 정책을 모두 찾아달라고 하지 마세요.
5. 지정된 상세도와 문체를 반드시 따르세요.
""".strip()

HUMAN_PROMPT = """
<policy>
{policy_context}
</policy>

<user_profile>
age: {age}
region: {region}
</user_profile>

금지 정책명: {policy_name}

<generation_controls>
detail_level: {detail_level}
detail_instruction: {detail_instruction}
detail_example: {detail_example}
question_style: {question_style}
style_instruction: {style_instruction}
style_example: {style_example}
</generation_controls>

{feedback}

평가 질문 하나를 생성하세요.
""".strip()


class GeneratedQuestion(BaseModel):
    user_input: str = Field(min_length=1)


class UserProfile(BaseModel):
    age: int = Field(ge=0)
    region: str = Field(min_length=1)


@dataclass(frozen=True)
class GenerationModel:
    provider: str
    model: str
    weight: float = 1.0

    @property
    def key(self) -> tuple[str, str]:
        return self.provider, self.model

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


def parse_generation_model(value: str) -> GenerationModel:
    provider_model, raw_weight = (
        value.rsplit("=", 1)
        if "=" in value
        else (value, "1")
    )
    if "/" not in provider_model:
        raise ValueError("--generation-model은 PROVIDER/MODEL=WEIGHT 형식입니다.")
    provider, model = provider_model.split("/", 1)
    provider = provider.strip()
    model = model.strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"지원하지 않는 provider입니다: {provider}. "
            f"지원 provider: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    weight = float(raw_weight)
    if not model or weight <= 0:
        raise ValueError("model은 비어 있을 수 없고 weight는 0보다 커야 합니다.")
    return GenerationModel(provider=provider, model=model, weight=weight)


def resolve_generation_models(
    config: AppConfig,
    values: list[str] | None,
) -> list[GenerationModel]:
    if not values:
        return [
            GenerationModel(
                provider=config.evaluation.provider,
                model=config.evaluation.model,
            )
        ]
    models = [parse_generation_model(value) for value in values]
    keys = [model.key for model in models]
    if len(keys) != len(set(keys)):
        raise ValueError("--generation-model에 같은 provider/model이 중복됐습니다.")
    return models


def load_policies(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as policy_file:
        policies = json.load(policy_file)
    if not isinstance(policies, list) or not policies:
        raise ValueError(f"{path}에는 비어 있지 않은 JSON 배열이 필요합니다.")
    seen: set[str] = set()
    for index, policy in enumerate(policies):
        policy_id = str(policy.get("plcyNo") or "").strip()
        if not policy_id:
            raise ValueError(f"{path}[{index}]에 plcyNo가 없습니다.")
        if policy_id in seen:
            raise ValueError(f"{path}에 중복 plcyNo가 있습니다: {policy_id}")
        seen.add(policy_id)
    return policies


def sample_policies(
    policies: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    strategy: str = DEFAULT_SAMPLING_STRATEGY,
) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("--sample-size는 1 이상이어야 합니다.")
    if sample_size > len(policies):
        raise ValueError("sample-size가 전체 정책 수보다 큽니다.")
    if strategy == "stratified":
        return stratified_sample_policies(policies, sample_size, seed)
    if strategy != "random":
        raise ValueError(f"지원하지 않는 sampling strategy입니다: {strategy}")
    return random.Random(seed).sample(policies, sample_size)


def primary_taxonomy_value(policy: dict[str, Any], field: str) -> str:
    value = str(policy.get(field) or "").strip()
    if not value or value == "-":
        return UNCLASSIFIED_TAXONOMY
    values = [part.strip() for part in value.split(",") if part.strip()]
    return values[0] if values else UNCLASSIFIED_TAXONOMY


def allocate_balanced_counts(capacities: dict[str, int], total: int) -> dict[str, int]:
    allocations = {key: 0 for key in capacities}
    remaining = total
    while remaining:
        allocated = False
        for key in sorted(capacities):
            if allocations[key] >= capacities[key]:
                continue
            allocations[key] += 1
            remaining -= 1
            allocated = True
            if remaining == 0:
                break
        if not allocated:
            raise RuntimeError("정책 표본 quota를 할당하지 못했습니다.")
    return allocations


def stratified_sample_policies(
    policies: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    parent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for policy in policies:
        parent_groups[primary_taxonomy_value(policy, "lclsfNm")].append(policy)

    parent_allocations = allocate_balanced_counts(
        {parent: len(group) for parent, group in parent_groups.items()},
        sample_size,
    )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for parent in sorted(parent_groups):
        child_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for policy in parent_groups[parent]:
            child_groups[primary_taxonomy_value(policy, "mclsfNm")].append(policy)
        child_allocations = allocate_balanced_counts(
            {child: len(group) for child, group in child_groups.items()},
            parent_allocations[parent],
        )
        for child in sorted(child_groups):
            quota = child_allocations[child]
            if quota:
                selected.extend(rng.sample(child_groups[child], quota))

    rng.shuffle(selected)
    return selected


def parse_distribution(value: str, labels: tuple[Any, ...], name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != len(labels):
        raise ValueError(f"{name}은 {len(labels)}개 값이 필요합니다: {labels}")
    weights = tuple(float(part) for part in parts)
    if (
        any(not math.isfinite(weight) for weight in weights)
        or any(weight < 0 for weight in weights)
        or sum(weights) <= 0
    ):
        raise ValueError(f"{name}은 음수가 아니며 합이 0보다 커야 합니다.")
    return weights


def weighted_assignments(
    labels: list[Any] | tuple[Any, ...],
    weights: list[float] | tuple[float, ...],
    count: int,
    seed: int,
    namespace: str,
) -> list[Any]:
    total_weight = sum(weights)
    exact_counts = [count * weight / total_weight for weight in weights]
    assigned_counts = [int(exact_count) for exact_count in exact_counts]
    remainder = count - sum(assigned_counts)
    order = sorted(
        range(len(labels)),
        key=lambda index: (-(exact_counts[index] - assigned_counts[index]), index),
    )
    for index in order[:remainder]:
        assigned_counts[index] += 1

    assignments = [
        label
        for label, assigned_count in zip(labels, assigned_counts, strict=True)
        for _ in range(assigned_count)
    ]
    random.Random(f"{seed}:{namespace}").shuffle(assignments)
    return assignments


def choose_profile(policy: dict[str, Any], seed: int) -> UserProfile:
    rng = random.Random(f"{seed}:{policy['plcyNo']}")
    age_metadata = build_age_metadata(
        policy.get("sprtTrgtMinAge"),
        policy.get("sprtTrgtMaxAge"),
    )
    if age_metadata["agePolicy"] == "specific":
        min_age = max(int(age_metadata["sprtTrgtMinAge"]), DEFAULT_MIN_AGE)
        max_age = min(int(age_metadata["sprtTrgtMaxAge"]), DEFAULT_MAX_AGE)
        if min_age > max_age:
            min_age = int(age_metadata["sprtTrgtMinAge"])
            max_age = int(age_metadata["sprtTrgtMaxAge"])
        age = rng.randint(min_age, max_age)
    else:
        age = rng.randint(DEFAULT_MIN_AGE, DEFAULT_MAX_AGE)

    region_codes = sorted(extract_sido_codes(policy.get("zipCd"))) or list(
        REGION_CODES
    )
    code_to_region = {code: name for name, code in REGION_NAME_TO_CODE.items()}
    return UserProfile(age=age, region=code_to_region[rng.choice(region_codes)])


def format_policy(policy: dict[str, Any]) -> str:
    lines = []
    for label, key in POLICY_FIELDS:
        value = str(policy.get(key) or "").strip()
        if value and value != "-":
            lines.append(f"{label}: {value}")
    text = "\n".join(lines)
    if len(text) <= DEFAULT_MAX_POLICY_CHARS:
        return text
    return text[:DEFAULT_MAX_POLICY_CHARS].rstrip() + "\n[이하 생략]"


def question_violations(
    question: str,
    policy: dict[str, Any],
    profile: UserProfile,
) -> list[str]:
    violations = []
    compact_question = re.sub(r"\s+", "", question)
    policy_id = str(policy["plcyNo"]).strip()
    policy_name = re.sub(r"\s+", "", str(policy.get("plcyNm") or "").strip())
    if policy_id in question:
        violations.append("정책번호 노출")
    if policy_name and policy_name in compact_question:
        violations.append("정책명 노출")
    if profile.region in question:
        violations.append("프로필 지역 노출")
    if re.search(rf"(?<!\d){profile.age}\s*세|만\s*{profile.age}(?!\d)", question):
        violations.append("프로필 나이 노출")
    return violations


def create_llm(config: AppConfig, model: GenerationModel):
    kwargs = {} if model.provider == "anthropic" else {"temperature": 0}
    return create_chat_model(
        provider=model.provider,
        model_name=model.model,
        **kwargs,
    )


def build_generators(config: AppConfig, models: list[GenerationModel]):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    return {
        model.key: prompt | create_llm(config, model).with_structured_output(
            GeneratedQuestion
        )
        for model in models
    }


def generate_question(
    chain,
    policy: dict[str, Any],
    profile: UserProfile,
    detail_level: int,
    question_style: str,
) -> tuple[str, list[str], int]:
    feedback = "이전 생성 시도 없음."
    violations: list[str] = []
    question = ""
    detail_guide = DETAIL_LEVEL_GUIDES[detail_level]
    style_guide = STYLE_GUIDES[question_style]
    for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
        result = chain.invoke(
            {
                "policy_context": format_policy(policy),
                "age": profile.age,
                "region": profile.region,
                "policy_name": str(policy.get("plcyNm") or "").strip(),
                "detail_level": detail_level,
                "detail_instruction": detail_guide["instruction"],
                "detail_example": detail_guide["example"],
                "question_style": question_style,
                "style_instruction": style_guide["instruction"],
                "style_example": style_guide["example"],
                "feedback": feedback,
            }
        )
        question = result.user_input.strip()
        violations = question_violations(question, policy, profile)
        if not violations:
            return question, [], attempt
        feedback = (
            "이전 질문은 다음 검증에 실패했습니다: "
            + ", ".join(violations)
            + ". 금지 값을 제거하고 다시 작성하세요."
        )
    return question, violations, DEFAULT_MAX_ATTEMPTS


def build_row(
    policy: dict[str, Any],
    profile: UserProfile,
    question: str,
    model: GenerationModel,
    detail_level: int,
    question_style: str,
    violations: list[str],
    attempts: int,
) -> dict[str, Any]:
    policy_id = str(policy["plcyNo"]).strip()
    return {
        "case_id": f"policy-{policy_id}",
        "user_input": question,
        "user_profile": profile.model_dump(),
        "expected_policy_ids": [policy_id],
        "exclude_expired": True,
        "metadata": {
            "policy_name": str(policy.get("plcyNm") or "").strip(),
            "generation_provider": model.provider,
            "generation_model": model.model,
            "detail_level": detail_level,
            "question_style": question_style,
            "generation_attempts": attempts,
            "validation_violations": violations,
        },
    }


def write_dataset(
    *,
    policies: list[dict[str, Any]],
    output_path: Path,
    config: AppConfig,
    models: list[GenerationModel],
    seed: int,
    detail_distribution: tuple[float, ...],
    style_distribution: tuple[float, ...],
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path}가 이미 있습니다. 덮어쓰려면 --overwrite를 사용하세요."
        )
    generators = build_generators(config, models)
    model_assignments = weighted_assignments(
        models,
        [model.weight for model in models],
        len(policies),
        seed,
        "generation-model",
    )
    detail_assignments = weighted_assignments(
        DETAIL_LEVELS,
        detail_distribution,
        len(policies),
        seed,
        "detail-level",
    )
    style_assignments = weighted_assignments(
        QUESTION_STYLES,
        style_distribution,
        len(policies),
        seed,
        "question-style",
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, (policy, model, detail_level, question_style) in enumerate(
            zip(
                policies,
                model_assignments,
                detail_assignments,
                style_assignments,
                strict=True,
            ),
            start=1,
        ):
            profile = choose_profile(policy, seed)
            question, violations, attempts = generate_question(
                generators[model.key],
                policy,
                profile,
                detail_level,
                question_style,
            )
            row = build_row(
                policy,
                profile,
                question,
                model,
                detail_level,
                question_style,
                violations,
                attempts,
            )
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{index}/{len(policies)}] {policy['plcyNo']} "
                f"model={model.label} detail={detail_level} style={question_style}"
            )


def parse_args() -> argparse.Namespace:
    detail_distribution_default = ",".join(
        f"{weight:g}" for weight in DEFAULT_DETAIL_DISTRIBUTION
    )
    style_distribution_default = ",".join(
        f"{weight:g}" for weight in DEFAULT_STYLE_DISTRIBUTION
    )
    parser = argparse.ArgumentParser(
        description="청년정책 RAG 평가 JSONL을 생성합니다."
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sampling-strategy",
        choices=("stratified", "random"),
        default=DEFAULT_SAMPLING_STRATEGY,
    )
    parser.add_argument(
        "--detail-distribution",
        default=detail_distribution_default,
        help=(
            "질문 상세도 1/2/3단계 가중치. "
            f"기본값: {detail_distribution_default}"
        ),
    )
    parser.add_argument(
        "--style-distribution",
        default=style_distribution_default,
        help=(
            "문체 가중치. 순서: colloquial,polite,keyword,situational. "
            f"기본값: {style_distribution_default}"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--generation-model",
        action="append",
        help="PROVIDER/MODEL=WEIGHT 형식. 여러 번 지정 가능",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="질문 생성 LLM 호출을 LangSmith에 trace합니다.",
    )
    args = parser.parse_args()
    try:
        args.detail_distribution = parse_distribution(
            args.detail_distribution,
            DETAIL_LEVELS,
            "--detail-distribution",
        )
        args.style_distribution = parse_distribution(
            args.style_distribution,
            QUESTION_STYLES,
            "--style-distribution",
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not args.enable_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    config = load_config()
    models = resolve_generation_models(config, args.generation_model)
    policies = sample_policies(
        load_policies(Path(config.path(config.data.raw))),
        sample_size=args.sample_size,
        seed=args.seed,
        strategy=args.sampling_strategy,
    )
    write_dataset(
        policies=policies,
        output_path=args.output.resolve(),
        config=config,
        models=models,
        seed=args.seed,
        detail_distribution=args.detail_distribution,
        style_distribution=args.style_distribution,
        overwrite=args.overwrite,
    )
    print(f"Dataset ready: {args.output.resolve()} ({args.sample_size} examples)")


if __name__ == "__main__":
    main()
