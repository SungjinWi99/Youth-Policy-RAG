import json
from typing import Any
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, ValidationError


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

    def to_langsmith_example(self) -> dict:
        return {
            "case_id": self.case_id,
            "inputs": EvaluationInputs(
                question=self.user_input,
                user_profile=self.user_profile,
                exclude_expired=self.exclude_expired,
            ).model_dump(),
            "outputs": EvaluationOutputs(
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
    faithfulness_chain = _score_chain(
        llm,
        """
당신은 RAG의 Faithfulness 평가자입니다.
생성 답변의 독립적인 사실 주장들이 검색된 context와 유저 프로필에 의해 얼마나 뒷받침되는지
평가하세요. 모든 사실 주장이 근거를 가지면 1, 근거 없는 주장이 대부분이면 0입니다.
답변이 질문에 유용한지는 평가하지 마세요.
""".strip(),
    )
    answer_relevance_chain = _score_chain(
        llm,
        """
당신은 RAG의 Answer Relevance 평가자입니다.
생성 답변이 사용자의 질문과 프로필에 직접 답하고 있는지 평가하세요.
질문의 핵심 요구를 빠짐없이 다루고 불필요한 내용이 거의 없으면 1입니다.
사실의 정확성이나 context 근거 여부는 이 지표에서 평가하지 마세요.
""".strip(),
    )
    context_helpfulness_chain = _score_chain(
        llm,
        """
당신은 청년정책 RAG의 Context Helpfulness 평가자입니다.
검색된 단일 context가 사용자의 질문과 프로필에 맞는 답변을 만드는 데
얼마나 도움이 되는지 평가하세요.
정책 주제, 지원 내용이 질문 해결에 직접적인 도움이 되면 1,
질문과 무관하거나 조건이 명백히 맞지 않으면 0에 가깝게 채점하세요.
생성 답변의 문장 품질이나 전체 faithfulness는 평가하지 마세요.
""".strip(),
    )
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

def load_examples(
    path: Path) -> list[dict]:
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

    return [example.to_langsmith_example() for example in examples]
