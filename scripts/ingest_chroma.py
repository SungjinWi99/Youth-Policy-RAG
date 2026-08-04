import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factory import DEFAULT_OLLAMA_BASE_URL, create_passage_embedding_model
from src.policy.corpus import load_policy_snapshot
from src.policy.store import ingest_documents, prepare_vector_store
from src.policy.utils import build_documents


DEFAULT_RAW_PATH = PROJECT_ROOT / "data/raw/youth_policies.json"
DEFAULT_COLLECTION_NAME = "youth_policies_rag"
DEFAULT_BATCH_SIZE = 270


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "config.yaml과 무관하게 provider/model/path를 지정해 "
            "청년정책 ChromaDB를 생성합니다."
        )
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("google", "ollama", "openai", "upstage"),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="문서 적재용 passage embedding 모델명",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        required=True,
        help="생성할 ChromaDB 디렉터리",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
    )
    parser.add_argument(
        "--distance-metric",
        choices=("cosine", "l2", "ip"),
        default="cosine",
        help="세 실험에서 같은 값을 사용해야 합니다(기본값: cosine)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="API rate limit 대응을 위한 batch 사이 대기 시간",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="동일 경로의 동일 collection을 삭제하고 다시 생성",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds는 0 이상이어야 합니다.")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv()

    raw_path = project_path(args.raw_path)
    chroma_dir = project_path(args.chroma_dir)
    policies = load_policy_snapshot(raw_path)
    documents, ids = build_documents(policies)
    embedding_model = create_passage_embedding_model(
        provider=args.provider,
        model_name=args.model,
        ollama_base_url=args.ollama_base_url,
    )
    vector_store = prepare_vector_store(
        chroma_dir=chroma_dir,
        collection_name=args.collection,
        embedding_model=embedding_model,
        distance_metric=args.distance_metric,
        recreate=args.recreate,
        provider=args.provider,
        passage_model=args.model,
    )
    ingest_documents(
        vector_store=vector_store,
        documents=documents,
        ids=ids,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
    )

    stored_count = vector_store._collection.count()
    if stored_count != len(documents):
        raise RuntimeError(
            f"적재 건수가 일치하지 않습니다: expected={len(documents)}, "
            f"stored={stored_count}"
        )
    print(
        f"Chroma ready: provider={args.provider}, model={args.model}, "
        f"path={chroma_dir}, collection={args.collection}, "
        f"metric={args.distance_metric}, count={stored_count}"
    )


if __name__ == "__main__":
    main()
