import json

import pytest
import requests

from scripts.sync_new_policies import apply_incremental_update
from src.policy.corpus import find_new_policies
from src.policy.source import (
    YouthPolicyApiError,
    fetch_page,
    fetch_policies,
)


class FakeResponse:
    def __init__(self, payload, *, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def api_page(items, *, total, page, page_size=2):
    return {
        "resultCode": 200,
        "resultMessage": "성공",
        "result": {
            "pagging": {
                "totCount": total,
                "pageNum": page,
                "pageSize": page_size,
            },
            "youthPolicyList": items,
        },
    }


def test_fetch_policies_collects_every_page():
    session = FakeSession([
        FakeResponse(api_page(
            [{"plcyNo": "P1"}, {"plcyNo": "P2"}],
            total=3,
            page=1,
        )),
        FakeResponse(api_page(
            [{"plcyNo": "P3"}],
            total=3,
            page=2,
        )),
    ])

    policies = fetch_policies(
        api_key="secret",
        page_size=2,
        request_delay=0,
        retry_backoff=0,
        session=session,
    )

    assert [item["plcyNo"] for item in policies] == ["P1", "P2", "P3"]
    assert [call[1]["params"]["pageNum"] for call in session.calls] == [1, 2]


def test_fetch_policies_can_limit_connection_test_to_first_page():
    session = FakeSession([
        FakeResponse(api_page(
            [{"plcyNo": "P1"}, {"plcyNo": "P2"}],
            total=3,
            page=1,
        )),
    ])

    policies = fetch_policies(
        api_key="secret",
        page_size=2,
        request_delay=0,
        retry_backoff=0,
        max_pages=1,
        session=session,
    )

    assert [item["plcyNo"] for item in policies] == ["P1", "P2"]
    assert len(session.calls) == 1


def test_fetch_policies_rejects_duplicate_ids_across_pages():
    session = FakeSession([
        FakeResponse(api_page(
            [{"plcyNo": "P1"}, {"plcyNo": "P2"}],
            total=3,
            page=1,
        )),
        FakeResponse(api_page(
            [{"plcyNo": "P2"}],
            total=3,
            page=2,
        )),
    ])

    with pytest.raises(ValueError, match="중복 plcyNo"):
        fetch_policies(
            api_key="secret",
            page_size=2,
            request_delay=0,
            retry_backoff=0,
            session=session,
        )


def test_fetch_page_does_not_expose_api_key_in_error():
    session = FakeSession([
        requests.ConnectionError(
            "failed https://example.test?apiKeyNm=SECRET"
        )
    ])

    with pytest.raises(YouthPolicyApiError) as exc_info:
        fetch_page(
            session,
            api_key="SECRET",
            page_num=1,
            page_size=100,
            timeout=15,
        )

    assert "SECRET" not in str(exc_info.value)


def test_find_new_policies_preserves_api_order():
    existing = [{"plcyNo": "P1"}]
    fetched = [
        {"plcyNo": "P1"},
        {"plcyNo": "P3"},
        {"plcyNo": "P2"},
    ]

    assert [
        item["plcyNo"] for item in find_new_policies(existing, fetched)
    ] == ["P3", "P2"]


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
