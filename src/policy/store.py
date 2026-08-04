"""Chroma 벡터 저장소에 정책 문서를 적재·갱신하는 연산."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from src.factory import (
    EMBEDDING_PASSAGE_MODEL_KEY,
    EMBEDDING_PROVIDER_KEY,
    verify_embedding_consistency,
)
from src.policy.corpus import policy_id, write_policy_snapshot_atomically
from src.policy.utils import build_documents


def prepare_vector_store(
    chroma_dir: Path,
    collection_name: str,
    embedding_model: Any,
    distance_metric: str,
    recreate: bool,
    provider: str,
    passage_model: str,
) -> Chroma:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    existing_names = {
        collection.name
        for collection in client.list_collections()
    }
    if collection_name in existing_names:
        if not recreate:
            raise FileExistsError(
                f"{chroma_dir}에 collection '{collection_name}'이 이미 "
                "존재합니다. 다시 만들려면 --recreate를 사용하세요."
            )
        client.delete_collection(collection_name)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(chroma_dir),
        # 어떤 조합으로 적재했는지 컬렉션에 남긴다. 이 값이 없으면 차원이 같은
        # 다른 모델로 서빙해도 오류 없이 무의미한 검색 결과가 나온다(ISSUE-002).
        collection_metadata={
            "hnsw:space": distance_metric,
            EMBEDDING_PROVIDER_KEY: provider,
            EMBEDDING_PASSAGE_MODEL_KEY: passage_model,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def ingest_documents(
    vector_store: Chroma,
    documents: list[Document],
    ids: list[str],
    batch_size: int,
    sleep_seconds: float,
) -> None:
    for start in tqdm(
        range(0, len(documents), batch_size),
        desc="Chroma 적재",
    ):
        end = start + batch_size
        vector_store.add_documents(
            documents=documents[start:end],
            ids=ids[start:end],
        )
        if end < len(documents) and sleep_seconds > 0:
            time.sleep(sleep_seconds)


def get_collection_ids(collection: Any) -> set[str]:
    result = collection.get(include=[])
    return {str(item_id) for item_id in result.get("ids", [])}


def ensure_collection_matches_raw(
    collection: Any,
    existing_policies: list[dict[str, Any]],
) -> None:
    raw_ids = {policy_id(item) for item in existing_policies}
    collection_ids = get_collection_ids(collection)
    if raw_ids == collection_ids:
        return

    only_raw = sorted(raw_ids - collection_ids)
    only_chroma = sorted(collection_ids - raw_ids)
    raise RuntimeError(
        "증분 반영 전 원본 JSON과 Chroma의 정책 ID가 일치해야 합니다. "
        f"raw_only={len(only_raw)} {only_raw[:5]}, "
        f"chroma_only={len(only_chroma)} {only_chroma[:5]}"
    )


def apply_incremental_update(
    *,
    raw_path: Path,
    existing_policies: list[dict[str, Any]],
    new_policies: list[dict[str, Any]],
    vector_store: Chroma,
    batch_size: int,
    sleep_seconds: float,
    provider: str,
    passage_model: str,
) -> None:
    ensure_collection_matches_raw(
        vector_store._collection,
        existing_policies,
    )
    # 적재 전에 막아야 한다. 여기를 통과시키면 잘못된 모델로 만든 벡터가
    # 정상 인덱스 안에 섞여 들어가고, 되돌릴 방법이 없다(ISSUE-002).
    verify_embedding_consistency(
        vector_store,
        provider=provider,
        passage_model=passage_model,
    )
    documents, new_ids = build_documents(new_policies)
    original_count = len(existing_policies)

    try:
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            vector_store.add_documents(
                documents=documents[start:end],
                ids=new_ids[start:end],
            )
            print(
                f"Chroma 적재: {min(end, len(documents))}/"
                f"{len(documents)}"
            )
            if end < len(documents) and sleep_seconds > 0:
                time.sleep(sleep_seconds)

        expected_count = original_count + len(new_policies)
        stored_count = vector_store._collection.count()
        if stored_count != expected_count:
            raise RuntimeError(
                f"Chroma 적재 건수가 일치하지 않습니다: "
                f"expected={expected_count}, stored={stored_count}"
            )
        write_policy_snapshot_atomically(
            raw_path,
            [*existing_policies, *new_policies],
        )
    except BaseException:
        # 신규 ID는 사전 검증 시 Chroma에 없었으므로 모두 삭제해도 안전하다.
        vector_store._collection.delete(ids=new_ids)
        raise
