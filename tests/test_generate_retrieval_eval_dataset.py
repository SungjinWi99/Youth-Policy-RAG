import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from scripts.generate_retrieval_eval_dataset import (
    DEFAULT_DETAIL_DISTRIBUTION,
    DETAIL_LEVELS,
    GenerationModelSpec,
    GeneratedQuestion,
    QUESTION_SYSTEM_PROMPT,
    QUESTION_STYLES,
    build_weighted_assignments,
    build_user_profile,
    build_question_generator,
    configure_langsmith_tracing,
    create_generation_llm,
    find_question_violations,
    format_policy_context,
    generate_example,
    load_policies,
    parse_args,
    parse_distribution,
    parse_generation_model_spec,
    resolve_generation_model_specs,
    sample_policies,
    write_dataset,
)


class RetrievalEvalDatasetGeneratorTest(unittest.TestCase):
    def test_sample_policies_is_deterministic_and_unique(self):
        policies = [
            {"plcyNo": f"policy-{index}"}
            for index in range(10)
        ]

        first = sample_policies(policies, sample_size=5, seed=7)
        second = sample_policies(policies, sample_size=5, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(
            len({policy["plcyNo"] for policy in first}),
            5,
        )

    def test_stratified_sampling_balances_parent_and_child_taxonomy(self):
        policies = []
        for parent in ("일자리", "주거"):
            for child in ("유형A", "유형B"):
                for index in range(5):
                    policies.append(
                        {
                            "plcyNo": f"{parent}-{child}-{index}",
                            "lclsfNm": parent,
                            "mclsfNm": child,
                        }
                    )

        sampled = sample_policies(
            policies,
            sample_size=8,
            seed=7,
            strategy="stratified",
        )

        strata = [
            (policy["lclsfNm"], policy["mclsfNm"])
            for policy in sampled
        ]
        self.assertEqual(
            {
                stratum: strata.count(stratum)
                for stratum in set(strata)
            },
            {
                ("일자리", "유형A"): 2,
                ("일자리", "유형B"): 2,
                ("주거", "유형A"): 2,
                ("주거", "유형B"): 2,
            },
        )

    def test_stratified_sampling_caps_small_strata_and_remains_unique(self):
        policies = [
            {
                "plcyNo": "rare-1",
                "lclsfNm": "희소",
                "mclsfNm": "희소",
            },
            *[
                {
                    "plcyNo": f"common-{index}",
                    "lclsfNm": "다수",
                    "mclsfNm": "다수",
                }
                for index in range(10)
            ],
        ]

        sampled = sample_policies(
            policies,
            sample_size=6,
            seed=3,
            strategy="stratified",
        )

        self.assertEqual(len(sampled), 6)
        self.assertEqual(
            len({policy["plcyNo"] for policy in sampled}),
            6,
        )
        self.assertIn("rare-1", {policy["plcyNo"] for policy in sampled})

    def test_weighted_assignments_use_exact_reproducible_counts(self):
        first = build_weighted_assignments(
            labels=DETAIL_LEVELS,
            weights=(0.2, 0.6, 0.2),
            count=200,
            seed=42,
            namespace="detail-level",
        )
        second = build_weighted_assignments(
            labels=DETAIL_LEVELS,
            weights=(0.2, 0.6, 0.2),
            count=200,
            seed=42,
            namespace="detail-level",
        )

        self.assertEqual(first, second)
        self.assertEqual(
            {level: first.count(level) for level in DETAIL_LEVELS},
            {1: 40, 2: 120, 3: 40},
        )

    def test_parse_distribution_validates_length_and_values(self):
        self.assertEqual(
            parse_distribution(
                "0.2,0.6,0.2",
                labels=("1", "2", "3"),
                argument_name="detail",
            ),
            (0.2, 0.6, 0.2),
        )
        with self.assertRaisesRegex(ValueError, "3개 값"):
            parse_distribution(
                "0.5,0.5",
                labels=("1", "2", "3"),
                argument_name="detail",
            )
        with self.assertRaisesRegex(ValueError, "음수"):
            parse_distribution(
                "0.5,-0.2,0.7",
                labels=("1", "2", "3"),
                argument_name="detail",
            )

    def test_parse_generation_model_supports_provider_model_and_weight(self):
        self.assertEqual(
            parse_generation_model_spec(
                "openai/gpt-5.4-mini=0.4"
            ),
            GenerationModelSpec(
                provider="openai",
                model="gpt-5.4-mini",
                weight=0.4,
            ),
        )
        self.assertEqual(
            parse_generation_model_spec("ollama/qwen3:8b"),
            GenerationModelSpec(
                provider="ollama",
                model="qwen3:8b",
                weight=1.0,
            ),
        )

    def test_resolve_generation_models_rejects_duplicates(self):
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                provider="openai",
                model="default-model",
            )
        )
        duplicate = GenerationModelSpec(
            provider="openai",
            model="same-model",
        )

        with self.assertRaisesRegex(ValueError, "중복"):
            resolve_generation_model_specs(
                config,
                model_specs=(duplicate, duplicate),
            )

    def test_cli_defaults_follow_detail_distribution_constant(self):
        args = parse_args([])

        self.assertEqual(
            args.detail_distribution,
            DEFAULT_DETAIL_DISTRIBUTION,
        )

    def test_cli_accepts_multiple_generation_models(self):
        args = parse_args(
            [
                "--generation-model",
                "openai/model-a=0.4",
                "--generation-model",
                "upstage/model-b=0.6",
            ]
        )

        self.assertEqual(
            args.generation_models,
            (
                GenerationModelSpec("openai", "model-a", 0.4),
                GenerationModelSpec("upstage", "model-b", 0.6),
            ),
        )

    def test_profile_is_inside_policy_age_and_region_conditions(self):
        policy = {
            "plcyNo": "policy-1",
            "sprtTrgtMinAge": "19",
            "sprtTrgtMaxAge": "34",
            "zipCd": "52140",
        }

        profile = build_user_profile(policy, seed=42)

        self.assertGreaterEqual(profile.age, 19)
        self.assertLessEqual(profile.age, 34)
        self.assertEqual(profile.region, "전북")

    def test_unrestricted_profile_uses_default_youth_age_and_region(self):
        policy = {
            "plcyNo": "policy-2",
            "sprtTrgtMinAge": "0",
            "sprtTrgtMaxAge": "0",
            "zipCd": "",
        }

        profile = build_user_profile(policy, seed=42)

        self.assertGreaterEqual(profile.age, 18)
        self.assertLessEqual(profile.age, 39)
        self.assertTrue(profile.region)

    def test_generated_example_has_initial_review_schema(self):
        policy = {
            "plcyNo": "policy-3",
            "sprtTrgtMinAge": "18",
            "sprtTrgtMaxAge": "39",
            "zipCd": "11110",
        }

        example = generate_example(
            policy,
            question_generator=lambda _, __, ___, ____: (
                "지원 내용과 신청 방법이 궁금해요."
            ),
            seed=42,
            detail_level=3,
            question_style="situational",
            generation_provider="openai",
            generation_model="test-model",
        )

        self.assertEqual(example.gold_policy_ids, ["policy-3"])
        self.assertEqual(
            example.user_input,
            "지원 내용과 신청 방법이 궁금해요.",
        )
        self.assertEqual(example.user_profile.region, "서울")
        self.assertEqual(example.detail_level, 3)
        self.assertEqual(example.question_style, "situational")
        self.assertEqual(example.generation_provider, "openai")
        self.assertEqual(example.generation_model, "test-model")
        self.assertEqual(example.validation_violations, [])
        self.assertEqual(example.generation_attempts, 1)
        self.assertEqual(example.hard_negative_ids, [])

    def test_question_prompt_explicitly_separates_profile(self):
        self.assertIn("프로필 정보를", QUESTION_SYSTEM_PROMPT)
        self.assertIn("별도의 user_profile 필드", QUESTION_SYSTEM_PROMPT)
        self.assertIn("나이, 거주 지역", QUESTION_SYSTEM_PROMPT)
        for level in DETAIL_LEVELS:
            self.assertIn(f"{level}단계", QUESTION_SYSTEM_PROMPT)

    def test_three_validation_failures_are_saved_with_warning(self):
        answers = iter(
            [
                "서울에 사는 27세인데 금지정책을 신청하고 싶어요.",
                "서울 거주 만 27세인데 금지 정책이 궁금해요.",
                "27세 서울 거주자용 금지정책을 알려주세요.",
            ]
        )

        class FakeLlm:
            def with_structured_output(self, _schema):
                return RunnableLambda(
                    lambda _: GeneratedQuestion(
                        user_input=next(answers)
                    )
                )

        question_generator = build_question_generator(
            FakeLlm(),
            max_attempts=3,
        )
        policy = {
            "plcyNo": "policy-warning",
            "plcyNm": "금지 정책",
            "sprtTrgtMinAge": "27",
            "sprtTrgtMaxAge": "27",
            "zipCd": "11110",
        }
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            example = generate_example(
                policy=policy,
                question_generator=question_generator,
                seed=42,
                generation_provider="anthropic",
                generation_model="claude-model",
            )

        self.assertEqual(
            example.user_input,
            "27세 서울 거주자용 금지정책을 알려주세요.",
        )
        self.assertEqual(example.generation_attempts, 3)
        self.assertEqual(
            set(example.validation_violations),
            {
                "정책명이 그대로 노출됨",
                "프로필 지역이 질문에 포함됨",
                "프로필 나이가 질문에 포함됨",
            },
        )
        self.assertIn("[warning]", stderr.getvalue())
        self.assertIn("policy-warning", stderr.getvalue())
        self.assertIn(
            "anthropic/claude-model",
            stderr.getvalue(),
        )

    def test_question_validation_rejects_policy_name_and_profile(self):
        policy = {
            "plcyNo": "policy-3",
            "plcyNm": "청년 주거 지원",
        }
        profile = SimpleNamespace(age=27, region="서울")

        violations = find_question_violations(
            "서울에 사는 27세인데 청년주거지원 신청하고 싶어요.",
            policy,
            profile,
        )

        self.assertEqual(
            set(violations),
            {
                "정책명이 그대로 노출됨",
                "프로필 지역이 질문에 포함됨",
                "프로필 나이가 질문에 포함됨",
            },
        )

    def test_policy_context_is_bounded(self):
        policy = {
            "plcyNm": "테스트 정책",
            "plcySprtCn": "가" * 100,
        }

        context = format_policy_context(policy, max_chars=30)

        self.assertLessEqual(len(context), 38)
        self.assertTrue(context.endswith("[이하 생략]"))

    @patch(
        "scripts.generate_retrieval_eval_dataset.create_chat_model"
    )
    def test_generation_model_uses_shared_factory(self, create_chat_model):
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                provider="openai",
                model="evaluation-model",
            )
        )
        create_chat_model.return_value = "llm-instance"

        result = create_generation_llm(config)

        create_chat_model.assert_called_once_with(
            provider="openai",
            model_name="evaluation-model",
            temperature=0,
        )
        self.assertEqual(result, "llm-instance")

    @patch(
        "scripts.generate_retrieval_eval_dataset.create_chat_model"
    )
    def test_anthropic_generation_model_omits_temperature(
        self,
        create_chat_model,
    ):
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                provider="openai",
                model="default-model",
            )
        )
        create_chat_model.return_value = "anthropic-llm"

        result = create_generation_llm(
            config,
            provider="anthropic",
            model="claude-model",
        )

        create_chat_model.assert_called_once_with(
            provider="anthropic",
            model_name="claude-model",
        )
        self.assertEqual(result, "anthropic-llm")

    @patch.dict(
        "os.environ",
        {
            "LANGSMITH_TRACING": "true",
            "LANGCHAIN_TRACING_V2": "true",
        },
        clear=False,
    )
    def test_langsmith_tracing_is_disabled_by_default(self):
        import os

        configure_langsmith_tracing(enabled=False)

        self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")
        self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "false")

    def test_write_dataset_outputs_one_json_object_per_line(self):
        policies = [
            {
                "plcyNo": "policy-4",
                "plcyNm": "첫 정책",
                "sprtTrgtMinAge": "18",
                "sprtTrgtMaxAge": "39",
                "zipCd": "11110",
            },
            {
                "plcyNo": "policy-5",
                "plcyNm": "둘째 정책",
                "sprtTrgtMinAge": "18",
                "sprtTrgtMaxAge": "39",
                "zipCd": "26110",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "dataset.jsonl"
            model_spec = GenerationModelSpec(
                provider="test",
                model="model",
            )

            write_dataset(
                policies=policies,
                output_path=output_path,
                question_generators={
                    model_spec.key: (
                        lambda policy, _, level, style: (
                            f"{policy['plcyNm']} {level} {style}"
                        )
                    )
                },
                generation_models=(model_spec,),
                seed=42,
            )

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            set(rows[0]),
            {
                "gold_policy_ids",
                "user_input",
                "user_profile",
                "detail_level",
                "question_style",
                "generation_provider",
                "generation_model",
                "validation_violations",
                "generation_attempts",
                "hard_negative_ids",
            },
        )
        self.assertEqual(rows[0]["hard_negative_ids"], [])
        self.assertEqual(rows[0]["generation_provider"], "test")
        self.assertEqual(rows[0]["generation_model"], "model")
        self.assertIn(rows[0]["detail_level"], DETAIL_LEVELS)
        self.assertIn(rows[0]["question_style"], QUESTION_STYLES)

    def test_write_dataset_respects_detail_style_and_model_distribution(self):
        policies = [
            {
                "plcyNo": f"policy-{index}",
                "plcyNm": f"정책 {index}",
                "sprtTrgtMinAge": "18",
                "sprtTrgtMaxAge": "39",
                "zipCd": "11110",
            }
            for index in range(10)
        ]
        model_specs = (
            GenerationModelSpec("openai", "model-a", 0.6),
            GenerationModelSpec("upstage", "model-b", 0.4),
        )
        question_generators = {
            spec.key: (
                lambda _, __, level, style, label=spec.label: (
                    f"{label}-{level}-{style}"
                )
            )
            for spec in model_specs
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "dataset.jsonl"
            write_dataset(
                policies=policies,
                output_path=output_path,
                question_generators=question_generators,
                generation_models=model_specs,
                seed=42,
                detail_distribution=(0.2, 0.6, 0.2),
                style_distribution=(0.3, 0.3, 0.2, 0.2),
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            {
                level: sum(row["detail_level"] == level for row in rows)
                for level in DETAIL_LEVELS
            },
            {1: 2, 2: 6, 3: 2},
        )
        self.assertEqual(
            {
                style: sum(row["question_style"] == style for row in rows)
                for style in QUESTION_STYLES
            },
            {
                "colloquial": 3,
                "polite": 3,
                "keyword": 2,
                "situational": 2,
            },
        )
        self.assertEqual(
            {
                spec.label: sum(
                    row["generation_provider"] == spec.provider
                    and row["generation_model"] == spec.model
                    for row in rows
                )
                for spec in model_specs
            },
            {
                "openai/model-a": 6,
                "upstage/model-b": 4,
            },
        )

    def test_load_policies_rejects_duplicate_policy_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policies.json"
            policy_path.write_text(
                json.dumps(
                    [
                        {"plcyNo": "duplicate"},
                        {"plcyNo": "duplicate"},
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "중복 plcyNo"):
                load_policies(policy_path)

    def test_age_sampling_is_stable_for_same_policy_and_seed(self):
        policy = {
            "plcyNo": "stable-policy",
            "sprtTrgtMinAge": "19",
            "sprtTrgtMaxAge": "39",
            "zipCd": "11110,26110",
        }

        first = build_user_profile(policy, seed=123)
        second = build_user_profile(policy, seed=123)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
