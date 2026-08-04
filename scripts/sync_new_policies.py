"""온통청년 API의 신규 정책을 원본 JSON과 Chroma에 증분 반영한다."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.factory import DEFAULT_OLLAMA_BASE_URL, create_passage_embedding_model
from src.policy.corpus import (
    find_new_policies,
    load_policy_snapshot,
    policy_id,
)
from src.policy.source import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_PAGE_SIZE,
    DEFAULT_REQUEST_DELAY,
    DEFAULT_RETRY_BACKOFF,
    DEFAULT_TIMEOUT,
    fetch_policies,
)
from src.policy.store import apply_incremental_update


DEFAULT_BATCH_SIZE = 270


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "온통청년 API 전체 목록에서 plcyNo 기준 신규 정책만 찾아 "
            "원본 JSON과 기존 Chroma 컬렉션에 증분 반영합니다."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API와 원본 JSON만 비교하고 파일/Chroma는 변경하지 않음",
    )
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--chroma-dir", type=Path)
    parser.add_argument("--collection")
    parser.add_argument(
        "--provider",
        choices=("google", "ollama", "openai", "upstage"),
    )
    parser.add_argument("--model", help="passage embedding 모델명")
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
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
        help="embedding batch 사이 대기 시간",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
    )
    args = parser.parse_args(argv)

    positive_values = {
        "--page-size": args.page_size,
        "--timeout": args.timeout,
        "--max-attempts": args.max_attempts,
        "--batch-size": args.batch_size,
    }
    for option, value in positive_values.items():
        if value < 1:
            parser.error(f"{option}는 1 이상이어야 합니다.")
    if args.request_delay < 0:
        parser.error("--request-delay는 0 이상이어야 합니다.")
    if args.retry_backoff < 0:
        parser.error("--retry-backoff는 0 이상이어야 합니다.")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds는 0 이상이어야 합니다.")
    return args


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = str(os.getenv("YOUTH_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(".env에 YOUTH_API_KEY를 설정해주세요.")

    config = load_config()
    raw_path = resolve_project_path(
        args.raw_path or Path(config.data.raw)
    )
    existing_policies = load_policy_snapshot(raw_path)
    fetched_policies = fetch_policies(
        api_key=api_key,
        page_size=args.page_size,
        request_delay=args.request_delay,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        retry_backoff=args.retry_backoff,
    )
    new_policies = find_new_policies(existing_policies, fetched_policies)
    existing_ids = {policy_id(item) for item in existing_policies}
    fetched_ids = {policy_id(item) for item in fetched_policies}
    missing_from_api = existing_ids - fetched_ids
    print(
        f"비교 완료: local={len(existing_policies)}, "
        f"api={len(fetched_policies)}, new={len(new_policies)}, "
        f"local_only={len(missing_from_api)}"
    )
    if missing_from_api:
        print(
            "local_only 정책은 추가-only 동기화에서 삭제하지 않습니다: "
            + ", ".join(sorted(missing_from_api)[:5])
        )
    if new_policies:
        preview = ", ".join(
            f"{policy_id(item)}({item.get('plcyNm', '')})"
            for item in new_policies[:5]
        )
        print(f"신규 정책 미리보기: {preview}")
    if args.dry_run or not new_policies:
        print(
            "dry-run: 변경하지 않았습니다."
            if args.dry_run
            else "추가할 신규 정책이 없습니다."
        )
        return

    chroma_dir = resolve_project_path(
        args.chroma_dir or Path(config.data.chroma_dir)
    )
    collection_name = (
        args.collection or config.data.chroma_collection_name
    )
    provider = args.provider or config.retriever.provider
    model_name = args.model or config.retriever.passage_model

    client = chromadb.PersistentClient(path=str(chroma_dir))
    existing_collections = {
        collection.name for collection in client.list_collections()
    }
    if collection_name not in existing_collections:
        raise RuntimeError(
            f"기존 Chroma collection을 찾을 수 없습니다: "
            f"path={chroma_dir}, collection={collection_name}"
        )

    embedding_model = create_passage_embedding_model(
        provider=provider,
        model_name=model_name,
        ollama_base_url=args.ollama_base_url,
    )
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(chroma_dir),
    )
    apply_incremental_update(
        raw_path=raw_path,
        existing_policies=existing_policies,
        new_policies=new_policies,
        vector_store=vector_store,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        provider=provider,
        passage_model=model_name,
    )
    final_count = vector_store._collection.count()
    print(
        f"동기화 완료: added={len(new_policies)}, total={final_count}, "
        f"raw={raw_path}, collection={collection_name}"
    )
    print("실행 중인 API 서버가 있다면 재시작해 BM25 인덱스를 갱신하세요.")


if __name__ == "__main__":
    main()
