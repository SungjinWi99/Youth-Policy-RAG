from pathlib import Path
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langsmith import Client
from src.config import load_config
from src.evaluators import load_examples

config = load_config()


def get_langsmith_ui_url(api_url: str) -> str:
    api_host = urlparse(api_url).netloc
    ui_hosts = {
        "api.smith.langchain.com": "smith.langchain.com",
        "eu.api.smith.langchain.com": "eu.smith.langchain.com",
        "apac.api.smith.langchain.com": "apac.smith.langchain.com",
        "aws.api.smith.langchain.com": "aws.smith.langchain.com",
    }
    ui_host = ui_hosts.get(api_host)
    return f"https://{ui_host}" if ui_host else api_url


def ensure_dataset(
    client: Client,
    dataset_name: str,
    examples_path: Path,
):
    examples = load_examples(examples_path)
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "청년정책 RAG의 검색 context와 생성 답변을 함께 평가하는 "
                "검증 데이터셋"
            ),
        )

    existing_ids = {
        example.id
        for example in client.list_examples(dataset_id=dataset.id)
    }
    new_examples = []
    for example in examples:
        case_id = example["case_id"]
        example_id = uuid5(NAMESPACE_URL, f"{dataset_name}:{case_id}")
        example_data = {
            "inputs": example["inputs"],
            "outputs": example["outputs"],
            "metadata": {
                "case_id": case_id,
                "source": "repository_jsonl",
                **example["metadata"],
            },
        }
        if example_id in existing_ids:
            client.update_example(
                example_id,
                dataset_id=dataset.id,
                **example_data,
            )
        else:
            new_examples.append({"id": example_id, **example_data})

    if new_examples:
        client.create_examples(
            dataset_id=dataset.id,
            examples=new_examples,
        )
    return dataset


def main():
    load_dotenv()
    config = load_config()
    client = Client()
    dataset = ensure_dataset(
        client,
        config.evaluation.dataset_name,
        config.evaluation.example_path,
    )
    print(
        f"Dataset ready: {dataset.name} "
        f"({len(load_examples(config.evaluation.example_path))} examples)"
    )
    print(f"LangSmith UI: {get_langsmith_ui_url(client.api_url)}")


if __name__ == "__main__":
    main()
