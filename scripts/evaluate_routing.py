import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.router import LLMContextRouter
from src.rag.routing_evaluation import (
    evaluate_routing,
    load_routing_evaluation_cases,
)


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data/eval/routing_eval.jsonl"


def create_evaluation_chat_model(
    provider: str,
    model_name: str,
):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_class = ChatAnthropic
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_class = ChatGoogleGenerativeAI
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        model_class = ChatOllama
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        model_class = ChatOpenAI
    elif provider == "upstage":
        from langchain_upstage import ChatUpstage

        model_class = ChatUpstage
    else:
        raise ValueError(f"지원하지 않는 provider입니다: {provider}")
    return model_class(model=model_name)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM context router의 reuse/search/clarify 정확도를 평가합니다."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--provider",
        choices=("anthropic", "google", "openai", "ollama", "upstage"),
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="LLM을 호출하지 않고 평가 데이터만 검증합니다.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = (
        args.dataset
        if args.dataset.is_absolute()
        else (PROJECT_ROOT / args.dataset).resolve()
    )
    cases = load_routing_evaluation_cases(dataset_path)

    if args.validate_only:
        counts = {
            route: sum(
                case.expected_route == route
                for case in cases
            )
            for route in ("reuse", "search", "clarify")
        }
        print(json.dumps(
            {
                "dataset": str(dataset_path),
                "total": len(cases),
                "route_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if not args.provider or not args.model:
        raise SystemExit(
            "--validate-only가 아니면 --provider와 --model이 필요합니다."
        )

    load_dotenv(PROJECT_ROOT / ".env")
    llm = create_evaluation_chat_model(
        args.provider,
        args.model,
    )
    summary = evaluate_routing(LLMContextRouter(llm), cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["accuracy"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
