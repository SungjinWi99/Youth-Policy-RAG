from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langsmith import Client

from src.config import load_config
from src.eval import build_evaluators, load_examples
from src.factory import build_rag_graph, create_chat_model


def get_langsmith_ui_url(api_url: str) -> str:
    return api_url.replace("api.", "", 1).rstrip("/")


def build_evaluator_llm(config):
    return create_chat_model(
        provider=config.evaluation.provider,
        model_name=config.evaluation.model,
        temperature=0,
    )


def stable_example_id(dataset_name: str, case_id: str):
    return uuid5(NAMESPACE_URL, f"langsmith:{dataset_name}:{case_id}")


def ensure_dataset(client: Client, dataset_name: str, example_path: str):
    examples = load_examples(example_path)
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="청년정책 RAG LangSmith 평가 데이터셋",
        )

    existing_examples = {
        example.id: example
        for example in client.list_examples(dataset_id=dataset.id)
    }

    for example in examples:
        case_id = example["case_id"]
        example_id = stable_example_id(dataset_name, case_id)
        metadata = {
            **example.get("metadata", {}),
            "case_id": case_id,
        }
        if example_id in existing_examples:
            client.update_example(
                example_id,
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=metadata,
                dataset_id=dataset.id,
            )
        else:
            client.create_example(
                dataset_id=dataset.id,
                example_id=example_id,
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=metadata,
            )

    return dataset


def run_rag_target(inputs: dict, *, rag) -> dict:
    result = rag.generate_answer(
        user_input=inputs["question"],
        user_profile=inputs.get("user_profile", {}),
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
        config.path(config.evaluation.example_path),
    )
    rag = build_rag_graph(config)
    evaluator_llm = build_evaluator_llm(config)

    results = client.evaluate(
        lambda inputs: run_rag_target(inputs, rag=rag),
        data=dataset.name,
        evaluators=build_evaluators(evaluator_llm),
        experiment_prefix=config.evaluation.experiment_prefix,
        description=(
            "청년정책 RAG의 Context Recall, Context Average Helpfulness, "
            "Faithfulness, Answer Relevance 평가"
        ),
        max_concurrency=config.evaluation.max_concurrency,
    )
    print(results)
    print(f"LangSmith UI: {get_langsmith_ui_url(str(client.api_url))}")


if __name__ == "__main__":
    main()
