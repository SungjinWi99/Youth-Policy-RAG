import json

import pytest

import src.policy.store as store
from src.policy.store import apply_policy_sync, plan_sync, prepare_vector_store


def test_prepare_vector_store_records_ingest_metadata(monkeypatch, tmp_path):
    captured = {}

    class FakeClient:
        def __init__(self, path):
            self.path = path

        def list_collections(self):
            return []

    def fake_chroma(**kwargs):
        captured.update(kwargs)
        return "vector-store"

    monkeypatch.setattr(
        store.chromadb, "PersistentClient", FakeClient
    )
    monkeypatch.setattr(store, "Chroma", fake_chroma)

    result = prepare_vector_store(
        chroma_dir=tmp_path / "chroma",
        collection_name="youth_policies_rag",
        embedding_model=object(),
        distance_metric="cosine",
        recreate=False,
        provider="upstage",
        passage_model="solar-embedding-1-large-passage",
    )

    metadata = captured["collection_metadata"]
    assert result == "vector-store"
    assert metadata["hnsw:space"] == "cosine"
    assert metadata["embedding_provider"] == "upstage"
    assert (
        metadata["embedding_passage_model"]
        == "solar-embedding-1-large-passage"
    )
    assert metadata["ingested_at"]


class FakeCollection:
    def __init__(self, ids, metadata=None):
        self.ids = list(ids)
        self.metadata = dict(metadata or {})

    def get(self, include):
        assert include == []
        return {"ids": list(self.ids)}

    def count(self):
        return len(self.ids)

    def delete(self, ids):
        deleted = set(ids)
        self.ids = [item_id for item_id in self.ids if item_id not in deleted]


class FakeVectorStore:
    def __init__(self, ids, *, fail=False, drop=False, metadata=None):
        self._collection = FakeCollection(ids, metadata)
        self.fail = fail
        self.drop = drop

    def add_documents(self, documents, ids):
        if self.fail:
            raise RuntimeError("embedding failed")
        if self.drop:
            return
        for item_id in ids:
            if item_id not in self._collection.ids:
                self._collection.ids.append(item_id)


def policy(plcy_no, **fields):
    return {"plcyNo": plcy_no, "plcyNm": f"정책{plcy_no}", **fields}


def test_plan_sync_covers_add_change_delete_and_drift():
    collection = FakeCollection(["P1", "P2", "P3", "GHOST"])
    snapshot = [
        policy("P1"),
        policy("P2"),
        policy("P3"),
        policy("P5"),  # Chroma에 빠져 있는 표류 문서
    ]
    fetched = [
        policy("P1"),  # 그대로
        policy("P2", plcyExplnCn="바뀐 설명"),  # 문서 내용 변경
        policy("P3", inqCnt="99999"),  # 문서에 안 들어가는 필드만 변경
        policy("P4"),  # 신규
        policy("P5"),  # 스냅샷에는 있지만 Chroma에는 없음
    ]

    plan = plan_sync(collection, snapshot, fetched)

    assert [item["plcyNo"] for item in plan.upsert] == ["P2", "P4", "P5"]
    assert plan.delete == []
    assert plan.orphan == ["GHOST"]


def test_plan_sync_deletes_policies_missing_from_api():
    collection = FakeCollection(["P1", "P2"])
    plan = plan_sync(collection, [policy("P1"), policy("P2")], [policy("P1")])

    assert plan.upsert == []
    assert plan.delete == ["P2"]


def test_apply_policy_sync_updates_chroma_then_raw(tmp_path):
    raw_path = tmp_path / "policies.json"
    snapshot = [policy("P1"), policy("P2")]
    raw_path.write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )
    vector_store = FakeVectorStore(["P1", "P2"])
    fetched = [policy("P1"), policy("P3")]

    apply_policy_sync(
        raw_path=raw_path,
        fetched_policies=fetched,
        plan=plan_sync(vector_store._collection, snapshot, fetched),
        vector_store=vector_store,
        batch_size=100,
        sleep_seconds=0,
        provider="upstage",
        passage_model="solar-embedding-1-large-passage",
    )

    assert sorted(vector_store._collection.ids) == ["P1", "P3"]
    assert [
        item["plcyNo"]
        for item in json.loads(raw_path.read_text(encoding="utf-8"))
    ] == ["P1", "P3"]


def test_apply_policy_sync_detects_silently_dropped_upsert(tmp_path):
    # 반영하지 않는 표류 문서(GHOST)가 있어도 적재 누락을 잡아내야 한다.
    raw_path = tmp_path / "policies.json"
    snapshot = [policy("P1")]
    raw_path.write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )
    vector_store = FakeVectorStore(["P1", "GHOST"], drop=True)
    fetched = [policy("P1"), policy("P2")]

    with pytest.raises(RuntimeError, match="upsert_missing=1"):
        apply_policy_sync(
            raw_path=raw_path,
            fetched_policies=fetched,
            plan=plan_sync(vector_store._collection, snapshot, fetched),
            vector_store=vector_store,
            batch_size=100,
            sleep_seconds=0,
            provider="upstage",
            passage_model="solar-embedding-1-large-passage",
        )

    assert json.loads(raw_path.read_text(encoding="utf-8")) == snapshot


def test_apply_policy_sync_leaves_snapshot_intact_on_failure(tmp_path):
    # 롤백 대신 재실행으로 수렴한다. 스냅샷이 그대로여야 다음 실행이
    # 같은 계획을 다시 세울 수 있다.
    raw_path = tmp_path / "policies.json"
    snapshot = [policy("P1")]
    raw_path.write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )
    vector_store = FakeVectorStore(["P1"], fail=True)
    fetched = [policy("P1"), policy("P2")]

    with pytest.raises(RuntimeError, match="embedding failed"):
        apply_policy_sync(
            raw_path=raw_path,
            fetched_policies=fetched,
            plan=plan_sync(vector_store._collection, snapshot, fetched),
            vector_store=vector_store,
            batch_size=100,
            sleep_seconds=0,
            provider="upstage",
            passage_model="solar-embedding-1-large-passage",
        )

    assert vector_store._collection.ids == ["P1"]
    assert json.loads(raw_path.read_text(encoding="utf-8")) == snapshot


def test_apply_policy_sync_refuses_mismatched_embedding_model(tmp_path):
    # 잘못된 모델로 만든 벡터가 정상 인덱스에 섞이면 되돌릴 수 없다.
    # 적재를 시작하기 전에 거부해야 한다.
    raw_path = tmp_path / "policies.json"
    snapshot = [policy("P1")]
    raw_path.write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )
    vector_store = FakeVectorStore(
        ["P1"],
        metadata={
            "embedding_provider": "upstage",
            "embedding_passage_model": "solar-embedding-1-large-passage",
        },
    )
    fetched = [policy("P1"), policy("P2")]

    with pytest.raises(RuntimeError, match="일치하지 않습니다"):
        apply_policy_sync(
            raw_path=raw_path,
            fetched_policies=fetched,
            plan=plan_sync(vector_store._collection, snapshot, fetched),
            vector_store=vector_store,
            batch_size=100,
            sleep_seconds=0,
            provider="ollama",
            passage_model="qwen3-embedding",
        )

    assert vector_store._collection.ids == ["P1"]
    assert json.loads(raw_path.read_text(encoding="utf-8")) == snapshot
