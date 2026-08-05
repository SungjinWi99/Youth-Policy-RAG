"""Chroma 벡터 저장소에 정책 문서를 적재·갱신하는 연산."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

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


class SyncPlan(NamedTuple):
    upsert: list[dict[str, Any]]
    delete: list[str]
    # Chroma에만 있고 스냅샷에도 API에도 없는 ID. 우리가 넣은 적 없는 문서이므로
    # 잘못된 컬렉션을 지우지 않도록 보고만 하고 건드리지 않는다.
    orphan: list[str]


def _document_map(
    policies: list[dict[str, Any]],
) -> dict[str, tuple[str, dict[str, Any]]]:
    documents, ids = build_documents(policies)
    return {
        item_id: (document.page_content, document.metadata)
        for item_id, document in zip(ids, documents)
    }


def plan_sync(
    collection: Any,
    snapshot_policies: list[dict[str, Any]],
    fetched_policies: list[dict[str, Any]],
) -> SyncPlan:
    """API 응답을 정답으로 삼아 Chroma에 반영할 작업을 계산한다.

    변경 판정은 원본 필드가 아니라 build_documents 결과로 한다. 조회수(inqCnt)나
    수정일시처럼 문서에 들어가지 않는 필드는 재임베딩을 유발하지 않는다.
    존재 판정은 스냅샷이 아니라 Chroma 실제 ID로 하므로, 중단된 실행이 남긴
    불일치는 다음 실행이 스스로 메운다.
    """
    chroma_ids = get_collection_ids(collection)
    snapshot_documents = _document_map(snapshot_policies)
    fetched_documents = _document_map(fetched_policies)
    fetched_by_id = {
        policy_id(item): item for item in fetched_policies
    }

    upsert = [
        item
        for item_id, item in fetched_by_id.items()
        if item_id not in chroma_ids
        or fetched_documents[item_id] != snapshot_documents.get(item_id)
    ]
    snapshot_ids = {policy_id(item) for item in snapshot_policies}
    delete = sorted(snapshot_ids - fetched_by_id.keys())
    orphan = sorted(chroma_ids - snapshot_ids - fetched_by_id.keys())
    return SyncPlan(upsert=upsert, delete=delete, orphan=orphan)


def apply_policy_sync(
    *,
    raw_path: Path,
    fetched_policies: list[dict[str, Any]],
    plan: SyncPlan,
    vector_store: Chroma,
    batch_size: int,
    sleep_seconds: float,
    provider: str,
    passage_model: str,
) -> None:
    """Chroma를 먼저 맞추고, 마지막에 스냅샷을 API 응답으로 교체한다.

    중간에 실패하면 스냅샷이 그대로 남아 다음 실행이 같은 계획을 다시 세운다.
    upsert도 delete도 멱등하므로 롤백 대신 재실행으로 수렴시킨다.
    """
    verify_embedding_consistency(
        vector_store,
        provider=provider,
        passage_model=passage_model,
    )
    documents, ids = build_documents(plan.upsert)
    ingest_documents(
        vector_store=vector_store,
        documents=documents,
        ids=ids,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
    )
    if plan.delete:
        vector_store._collection.delete(ids=plan.delete)

    # 전체 건수 대신 우리가 쓴 ID만 확인한다. 건수 비교는 반영하지 않는 표류
    # 문서(orphan)만큼 어긋나서, 적재가 조용히 실패해도 통과할 수 있다.
    stored_ids = get_collection_ids(vector_store._collection)
    missing = sorted(set(ids) - stored_ids)
    remaining = sorted(set(plan.delete) & stored_ids)
    if missing or remaining:
        raise RuntimeError(
            f"Chroma 반영이 끝나지 않았습니다: "
            f"upsert_missing={len(missing)} {missing[:5]}, "
            f"delete_remaining={len(remaining)} {remaining[:5]}"
        )
    write_policy_snapshot_atomically(raw_path, fetched_policies)
