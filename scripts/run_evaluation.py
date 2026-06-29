from functools import partial

from dotenv import load_dotenv
from langsmith import Client

from scripts.create_eval_dataset import (
    ensure_dataset,
    get_langsmith_ui_url,
)
from src.config import load_config
from src.evaluators import build_evaluators
from src.factory import build_rag_pipeline, create_chat_model
from src.user.models import UserProfile


def build_evaluator_llm(config, fallback_llm):
    provider = config.evaluation.provider
    model = config.evaluation.model
    if provider == config.llm.provider and model == config.llm.model:
        return fallback_llm

    return create_chat_model(
        provider=provider,
        model_name=model,
        temperature=0,
    )


def run_rag_target(inputs: dict, *, rag) -> dict:
    result = rag.generate_answer(
        user_input=inputs["question"],
        user_profile=UserProfile(**inputs.get("user_profile", {})),
        exclude_expired=inputs.get("exclude_expired", True),
    )
    return result.model_dump()


def main():
    load_dotenv()
    config = load_config()
    client = Client()

    dataset = ensure_dataset(
        client,
        config.evaluation.dataset_name,
        config.evaluation.example_path,
    )
    rag = build_rag_pipeline(config)
    evaluator_llm = build_evaluator_llm(config, rag.llm)

    results = client.evaluate(
        partial(run_rag_target, rag=rag),
        data=dataset.name,
        evaluators=build_evaluators(evaluator_llm),
        experiment_prefix=config.evaluation.experiment_prefix,
        description=(
            "청년정책 RAG의 Context Recall, Context Precision, "
            "Faithfulness, Answer Relevance 평가"
        ),
        max_concurrency=config.evaluation.max_concurrency,
    )
    print(results)
    print(f"LangSmith UI: {get_langsmith_ui_url(client.api_url)}")


if __name__ == "__main__":
    main()
