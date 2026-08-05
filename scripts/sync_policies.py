"""온통청년 API를 정답으로 삼아 원본 JSON과 Chroma를 동기화한다."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.factory import DEFAULT_OLLAMA_BASE_URL, create_passage_embedding_model
from src.policy.corpus import (
    load_policy_snapshot,
    policy_id,
    write_policy_snapshot_atomically,
)
from src.policy.source import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_PAGE_SIZE,
    DEFAULT_REQUEST_DELAY,
    DEFAULT_RETRY_BACKOFF,
    DEFAULT_TIMEOUT,
    fetch_policies,
)
from src.policy.store import apply_policy_sync, plan_sync


DEFAULT_BATCH_SIZE = 270
SAMPLE_RAW_PATH = "data/raw/youth_policies.sample.json"
# API가 그럴듯하지만 잘린 목록을 주면 삭제와 스냅샷 교체가 함께 일어나 되돌릴 수
# 없다. 평소 폐지 규모를 넘는 삭제는 사람이 확인하게 한다.
DELETION_LIMIT_RATIO = 0.05
DELETION_LIMIT_FLOOR = 20


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "온통청년 API 전체 목록을 받아 원본 JSON과 Chroma 컬렉션에 "
            "정책 추가·변경·삭제를 반영합니다."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="반영할 건수만 계산하고 파일/Chroma는 변경하지 않음",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="원본 JSON만 새로 수집해 저장하고 Chroma는 건드리지 않음",
    )
    parser.add_argument(
        "--limit-test",
        action="store_true",
        help=(
            f"첫 페이지 10건만 받아 {SAMPLE_RAW_PATH}에 저장 "
            "(운영 원본과 Chroma를 건드리지 않는 연결 확인)"
        ),
    )
    parser.add_argument(
        "--allow-deletions",
        action="store_true",
        help="삭제 대상이 안전 한도를 넘어도 그대로 반영",
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


def preview(policies: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{policy_id(item)}({item.get('plcyNm', '')})"
        for item in policies[:5]
    )


def open_vector_store(args: argparse.Namespace, config) -> Chroma:
    chroma_dir = resolve_project_path(
        args.chroma_dir or Path(config.data.chroma_dir)
    )
    collection_name = (
        args.collection or config.data.chroma_collection_name
    )
    client = chromadb.PersistentClient(path=str(chroma_dir))
    existing_collections = {
        collection.name for collection in client.list_collections()
    }
    if collection_name not in existing_collections:
        raise RuntimeError(
            f"기존 Chroma collection을 찾을 수 없습니다: "
            f"path={chroma_dir}, collection={collection_name}"
        )
    return Chroma(
        collection_name=collection_name,
        embedding_function=create_passage_embedding_model(
            provider=args.provider or config.retriever.provider,
            model_name=args.model or config.retriever.passage_model,
            ollama_base_url=args.ollama_base_url,
        ),
        persist_directory=str(chroma_dir),
    )


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = str(os.getenv("YOUTH_API_KEY") or "").strip()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise SystemExit(".env에 올바른 YOUTH_API_KEY를 설정해주세요.")

    config = load_config()
    default_raw = SAMPLE_RAW_PATH if args.limit_test else config.data.raw
    raw_path = resolve_project_path(args.raw_path or Path(default_raw))

    fetched_policies = fetch_policies(
        api_key=api_key,
        page_size=10 if args.limit_test else args.page_size,
        request_delay=args.request_delay,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        retry_backoff=args.retry_backoff,
        max_pages=1 if args.limit_test else None,
    )
    if not fetched_policies:
        raise RuntimeError("API에서 수집된 정책이 없습니다.")

    if args.limit_test or args.snapshot_only:
        mode = "연결 테스트" if args.limit_test else "스냅샷 수집"
        if args.dry_run:
            print(f"dry-run({mode}): count={len(fetched_policies)}, 저장 안 함")
            return
        write_policy_snapshot_atomically(raw_path, fetched_policies)
        print(f"{mode} 완료: count={len(fetched_policies)}, raw={raw_path}")
        return

    snapshot_policies = load_policy_snapshot(raw_path)
    vector_store = open_vector_store(args, config)
    plan = plan_sync(
        vector_store._collection,
        snapshot_policies,
        fetched_policies,
    )
    print(
        f"비교 완료: local={len(snapshot_policies)}, "
        f"api={len(fetched_policies)}, upsert={len(plan.upsert)}, "
        f"delete={len(plan.delete)}, chroma_only={len(plan.orphan)}"
    )
    if plan.upsert:
        print(f"추가·변경 미리보기: {preview(plan.upsert)}")
    if plan.delete:
        print(f"삭제 미리보기: {', '.join(plan.delete[:5])}")
    if plan.orphan:
        print(
            "chroma_only 문서는 스냅샷에 없어 건드리지 않습니다: "
            + ", ".join(plan.orphan[:5])
        )

    if args.dry_run or not (plan.upsert or plan.delete):
        print(
            "dry-run: 변경하지 않았습니다."
            if args.dry_run
            else "반영할 변경이 없습니다."
        )
        return

    deletion_limit = max(
        DELETION_LIMIT_FLOOR,
        int(len(snapshot_policies) * DELETION_LIMIT_RATIO),
    )
    if len(plan.delete) > deletion_limit and not args.allow_deletions:
        raise SystemExit(
            f"삭제 대상 {len(plan.delete)}건이 안전 한도 {deletion_limit}건을 "
            "넘습니다. API 응답을 확인한 뒤 의도한 삭제라면 "
            "--allow-deletions로 다시 실행하세요."
        )

    apply_policy_sync(
        raw_path=raw_path,
        fetched_policies=fetched_policies,
        plan=plan,
        vector_store=vector_store,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        provider=args.provider or config.retriever.provider,
        passage_model=args.model or config.retriever.passage_model,
    )
    print(
        f"동기화 완료: upsert={len(plan.upsert)}, "
        f"delete={len(plan.delete)}, "
        f"total={vector_store._collection.count()}, raw={raw_path}"
    )
    print("실행 중인 API 서버가 있다면 재시작해 BM25 인덱스를 갱신하세요.")


if __name__ == "__main__":
    main()
