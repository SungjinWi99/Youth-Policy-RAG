import json

import pytest

import src.policy.store as store
from src.policy.store import apply_incremental_update, prepare_vector_store


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
    def __init__(self, ids, *, fail=False, metadata=None):
        self._collection = FakeCollection(ids, metadata)
        self.fail = fail

    def add_documents(self, documents, ids):
        self._collection.ids.extend(ids)
        if self.fail:
            raise RuntimeError("embedding failed")


def test_apply_incremental_update_updates_chroma_and_raw(tmp_path):
    raw_path = tmp_path / "policies.json"
    existing = [{"plcyNo": "P1", "plcyNm": "기존"}]
    raw_path.write_text(
        json.dumps(existing, ensure_ascii=False),
        encoding="utf-8",
    )
    vector_store = FakeVectorStore(["P1"])

    apply_incremental_update(
        raw_path=raw_path,
        existing_policies=existing,
        new_policies=[{"plcyNo": "P2", "plcyNm": "신규"}],
        vector_store=vector_store,
        batch_size=100,
        sleep_seconds=0,
        provider="upstage",
        passage_model="solar-embedding-1-large-passage",
    )

    assert vector_store._collection.ids == ["P1", "P2"]
    assert [
        item["plcyNo"]
        for item in json.loads(raw_path.read_text(encoding="utf-8"))
    ] == ["P1", "P2"]


def test_apply_incremental_update_rolls_back_chroma_on_failure(tmp_path):
    raw_path = tmp_path / "policies.json"
    existing = [{"plcyNo": "P1"}]
    raw_path.write_text(json.dumps(existing), encoding="utf-8")
    vector_store = FakeVectorStore(["P1"], fail=True)

    with pytest.raises(RuntimeError, match="embedding failed"):
        apply_incremental_update(
            raw_path=raw_path,
            existing_policies=existing,
            new_policies=[{"plcyNo": "P2"}],
            vector_store=vector_store,
            batch_size=100,
            sleep_seconds=0,
            provider="upstage",
            passage_model="solar-embedding-1-large-passage",
        )

    assert vector_store._collection.ids == ["P1"]
    assert json.loads(raw_path.read_text(encoding="utf-8")) == existing


def test_apply_incremental_update_refuses_mismatched_embedding_model(tmp_path):
    # 잘못된 모델로 만든 벡터가 정상 인덱스에 섞이면 되돌릴 수 없다.
    # 적재를 시작하기 전에 거부해야 한다.
    raw_path = tmp_path / "policies.json"
    existing = [{"plcyNo": "P1"}]
    raw_path.write_text(json.dumps(existing), encoding="utf-8")
    vector_store = FakeVectorStore(
        ["P1"],
        metadata={
            "embedding_provider": "upstage",
            "embedding_passage_model": "solar-embedding-1-large-passage",
        },
    )

    with pytest.raises(RuntimeError, match="일치하지 않습니다"):
        apply_incremental_update(
            raw_path=raw_path,
            existing_policies=existing,
            new_policies=[{"plcyNo": "P2"}],
            vector_store=vector_store,
            batch_size=100,
            sleep_seconds=0,
            provider="ollama",
            passage_model="qwen3-embedding",
        )

    assert vector_store._collection.ids == ["P1"]
    assert json.loads(raw_path.read_text(encoding="utf-8")) == existing
