import json
from typing import Any
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, ValidationError


FAITHFULNESS_PROMPT = """
당신은 RAG의 Faithfulness 평가자입니다.

생성 답변의 독립적인 사실 주장들이 제공된 검색 context와 사용자 프로필에 의해
뒷받침되는지만 0~1로 평가하세요.

평가하지 마세요:
- 답변의 유용성
- 답변의 문장 품질
- 답변의 상세함
- 외부 지식 기준의 사실성

기준:
1.0 = 모든 주요 사실 주장이 context/profile에서 확인됨
0.75 = 대부분 확인되며 사소한 근거 부족만 있음
0.5 = 근거 있는 주장과 없는 주장이 섞임
0.25 = 일부만 근거가 있고 대부분은 근거 부족
0.0 = 대부분 근거가 없거나 context/profile과 충돌함


주의:
- policy 내용, 지원 조건, 신청 기간, 금액, 절차는 context에 있어야만 근거 있는 주장입니다.
- user_profile은 사용자 속성 확인 근거로만 사용하세요.
- context에 없는 내용을 단정하면 감점하세요.
- “제공된 정보만으로는 알 수 없음”처럼 한계를 밝힌 문장은 감점하지 마세요.
- 정보 누락 자체는 faithfulness 문제가 아닙니다.
""".strip()

ANSWER_RELEVANCY_PROMPT = """
당신은 RAG의 Answer Relevance 평가자입니다.

생성 답변이 사용자 질문과 사용자 프로필에 얼마나 직접적으로 답하는지만 0~1로 평가하세요.

평가하지 마세요:
- 사실 정확성
- context 근거 여부
- faithfulness
- 문장 품질

기준:
1.0 = 질문의 핵심 요구에 직접 답하고 필요한 프로필 조건을 반영하며 불필요한 내용이 거의 없음
0.75 = 대체로 답하지만 일부 세부 요구가 빠졌거나 약간 불필요한 내용이 있음
0.5 = 같은 주제이나 핵심 요구를 부분적으로만 다룸
0.25 = 약하게 관련되지만 실제 질문에는 거의 답하지 못함
0.0 = 무관하거나 다른 질문에 답함

주의:
- “신청 가능한가?”에는 가능 여부/조건 판단 중심으로 답해야 합니다.
- “지원 내용은?”에는 지원 내용 중심으로 답해야 합니다.
- “신청 방법은?”에는 신청 절차 중심으로 답해야 합니다.
- 답변이 사실적으로 틀렸거나 context에 없는 내용이어도, 이 지표에서는 그 자체로 감점하지 마세요.
- 단, 틀린 내용 때문에 질문의 핵심 의도에서 벗어나면 관련성 기준으로 감점하세요.
""".strip()

CONTEXT_HELPFULNESS_PROMPT = """
당신은 청년정책 RAG의 Context Helpfulness 평가자입니다.
검색된 단일 context가 사용자의 질문과 상황에 얼마나 도움이 되는지 평가하세요.

평가하지 말 것:
- 생성 답변의 문장 품질
- 생성 답변의 사실성/faithfulness
- 전체 검색 결과의 품질

기준:
- 1.0: 질문 의도와 정책 주제가 직접 일치하고, 사용자 프로필 조건에도 명백히 부합하며,
       지원 내용/대상/신청 정보 등 답변에 바로 쓸 핵심 정보가 있다.
- 0.75: 질문 해결에 직접 도움이 되지만, 일부 조건이나 세부 정보가 부족하거나 프로필 적합성이 완전히 확정되지는 않는다.
- 0.5: 같은 큰 분야의 정책이지만 질문의 핵심을 직접 해결하지 못하고, 배경 정보나 보조 정보로만 쓸 수 있다.
- 0.25: 주제나 대상이 일부만 겹치며, 답변에 쓰더라도 매우 제한적으로만 도움이 된다.
- 0.0: 질문과 무관하거나, 사용자 프로필과 명백히 맞지 않거나, 질문 해결에 쓰면 오히려 잘못된 답변을 만들 가능성이 높다.

주의:
조건이 context에 없거나 불명확하면 그것만으로 0점 처리하지 마세요.
지역/나이/소득/신청기간 등이 명백히 충돌할 때만 크게 감점하세요.
""".strip()

class MetricScore(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="평가 기준을 전혀 만족하지 못하면 0, 완전히 만족하면 1",
    )
    reasoning: str = Field(description="점수의 핵심 근거를 한국어로 간결하게 설명. 1~2 문장으로 간단하게 서술하세요.")


class EvaluationUserProfile(BaseModel):
    age: int | None = Field(default=None, ge=0)
    gender: str | None = None
    job: str | None = None
    income: int | None = Field(default=None, ge=0)
    region: str | None = None


class EvaluationInputs(BaseModel):
    question: str = Field(min_length=1)
    user_profile: EvaluationUserProfile = Field(
        default_factory=EvaluationUserProfile
    )
    exclude_expired: bool = True


class EvaluationOutputs(BaseModel):
    expected_policy_ids: list[str] = Field(min_length=1)


class EvaluationExample(BaseModel):
    case_id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    user_profile: EvaluationUserProfile = Field(
        default_factory=EvaluationUserProfile
    )
    expected_policy_ids: list[str] = Field(min_length=1)
    exclude_expired: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_evaluation_item(self) -> dict:
        return {
            "case_id": self.case_id,
            "input": EvaluationInputs(
                question=self.user_input,
                user_profile=self.user_profile,
                exclude_expired=self.exclude_expired,
            ).model_dump(),
            "expected_output": EvaluationOutputs(
                expected_policy_ids=self.expected_policy_ids,
            ).model_dump(),
            "metadata": self.metadata,
        }

def calculate_context_recall(
    retrieved_policy_ids: list[str],
    expected_policy_ids: list[str],
) -> float:
    expected = set(expected_policy_ids)
    if not expected:
        return 0.0
    retrieved = set(retrieved_policy_ids)
    return len(retrieved & expected) / len(expected)


def _score_chain(llm: Any, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{evaluation_payload}"),
        ]
    )
    return prompt | llm.with_structured_output(MetricScore)


def build_evaluators(llm: Any):
    faithfulness_chain = _score_chain(llm, FAITHFULNESS_PROMPT)
    answer_relevance_chain = _score_chain(llm, ANSWER_RELEVANCY_PROMPT)
    context_helpfulness_chain = _score_chain(llm, CONTEXT_HELPFULNESS_PROMPT)
    def context_recall(
        outputs: dict,
        reference_outputs: dict,
    ) -> dict:
        retrieved_ids = outputs.get("retrieved_policy_ids", [])
        expected_ids = reference_outputs.get("expected_policy_ids", [])
        matched_ids = sorted(set(retrieved_ids) & set(expected_ids))
        score = calculate_context_recall(
            retrieved_ids,
            expected_ids,
        )
        return {
            "key": "context_recall",
            "score": score,
            "comment": (
                f"정답 정책 {len(expected_ids)}건 중 {len(matched_ids)}건 검색: "
                f"{matched_ids}"
            ),
        }

    def context_average_helpfulness(
        inputs: dict,
        outputs: dict,
    ) -> dict:
        retrieved_ids = outputs.get("retrieved_policy_ids", [])
        contexts = outputs.get("contexts", [])
        if not contexts:
            return {
                "key": "context_average_helpfulness",
                "score": 0.0,
                "comment": "검색된 context가 없습니다.",
            }

        context_scores: list[MetricScore] = []
        for index, context in enumerate(contexts):
            policy_id = (
                retrieved_ids[index]
                if index < len(retrieved_ids)
                else f"context_{index + 1}"
            )
            context_scores.append(
                context_helpfulness_chain.invoke(
                    {
                        "evaluation_payload": json.dumps(
                            {
                                "question": inputs["question"],
                                "user_profile": inputs.get("user_profile", {}),
                                "policy_id": policy_id,
                                "retrieved_context": context,
                            },
                            ensure_ascii=False,
                        )
                    }
                )
            )

        score = sum(item.score for item in context_scores) / len(context_scores)
        comments = [
            (
                f"{retrieved_ids[index] if index < len(retrieved_ids) else index + 1}: "
                f"{item.score:.2f} - {item.reasoning}"
            )
            for index, item in enumerate(context_scores)
        ]
        return {
            "key": "context_average_helpfulness",
            "score": score,
            "comment": "\n".join(comments),
        }

    def faithfulness(inputs: dict, outputs: dict) -> dict:
        result = faithfulness_chain.invoke(
            {
                "evaluation_payload": json.dumps(
                    {
                        "question": inputs["question"],
                        "user_profile": inputs.get("user_profile", {}),
                        "generated_answer": outputs.get("answer", ""),
                        "retrieved_contexts": outputs.get("contexts", []),
                    },
                    ensure_ascii=False,
                )
            }
        )
        return {
            "key": "faithfulness",
            "score": result.score,
            "comment": result.reasoning,
        }

    def answer_relevance(inputs: dict, outputs: dict) -> dict:
        result = answer_relevance_chain.invoke(
            {
                "evaluation_payload": json.dumps(
                    {
                        "question": inputs["question"],
                        "user_profile": inputs.get("user_profile", {}),
                        "generated_answer": outputs.get("answer", ""),
                    },
                    ensure_ascii=False,
                )
            }
        )
        return {
            "key": "answer_relevance",
            "score": result.score,
            "comment": result.reasoning,
        }

    return [
        context_recall,
        context_average_helpfulness,
        faithfulness,
        answer_relevance,
    ]


def _to_langfuse_evaluation(result: dict):
    try:
        from langfuse import Evaluation
    except ImportError as error:
        raise RuntimeError(
            "Langfuse 평가를 실행하려면 langfuse 패키지가 필요합니다. "
            "requirements.txt를 설치한 뒤 다시 실행하세요."
        ) from error

    return Evaluation(
        name=result["key"],
        value=result["score"],
        comment=result.get("comment"),
    )


def build_langfuse_evaluators(llm: Any):
    evaluators = {
        evaluator.__name__: evaluator
        for evaluator in build_evaluators(llm)
    }

    def context_recall(
        *,
        output: dict,
        expected_output: dict,
        **kwargs,
    ):
        return _to_langfuse_evaluation(
            evaluators["context_recall"](
                outputs=output or {},
                reference_outputs=expected_output or {},
            )
        )

    def context_average_helpfulness(
        *,
        input: dict,
        output: dict,
        **kwargs,
    ):
        return _to_langfuse_evaluation(
            evaluators["context_average_helpfulness"](
                inputs=input or {},
                outputs=output or {},
            )
        )

    def faithfulness(
        *,
        input: dict,
        output: dict,
        **kwargs,
    ):
        return _to_langfuse_evaluation(
            evaluators["faithfulness"](
                inputs=input or {},
                outputs=output or {},
            )
        )

    def answer_relevance(
        *,
        input: dict,
        output: dict,
        **kwargs,
    ):
        return _to_langfuse_evaluation(
            evaluators["answer_relevance"](
                inputs=input or {},
                outputs=output or {},
            )
        )

    return [
        context_recall,
        context_average_helpfulness,
        faithfulness,
        answer_relevance,
    ]


def load_evaluation_items(path: Path) -> list[dict]:
    dataset_path = Path(path)
    examples: list[EvaluationExample] = []
    case_ids: set[str] = set()

    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue

            try:
                example = EvaluationExample.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"{dataset_path}:{line_number} 평가 데이터가 유효하지 않습니다."
                ) from error

            if example.case_id in case_ids:
                raise ValueError(
                    f"{dataset_path}:{line_number} 중복 case_id: {example.case_id}"
                )

            case_ids.add(example.case_id)
            examples.append(example)

    if not examples:
        raise ValueError(f"{dataset_path}에 평가 데이터가 없습니다.")

    return [example.to_evaluation_item() for example in examples]
