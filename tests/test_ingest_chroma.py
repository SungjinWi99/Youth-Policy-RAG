from scripts.ingest_chroma import build_documents
from src.policy.utils import POLICY_METADATA_LABELS


def test_build_documents_adds_capacity_metadata_and_other_matters_to_content():
    documents, ids = build_documents([{
        "plcyNo": "POLICY-001",
        "plcyNm": "청년 지원 정책",
        "etcMttrCn": "선착순 모집이며 중복 신청할 수 없습니다.",
        "sprtSclLmtYn": "Y",
        "sprtSclCnt": "30",
        "sprtArvlSeqYn": "Y",
    }])

    assert ids == ["POLICY-001"]
    assert "기타 사항: 선착순 모집이며 중복 신청할 수 없습니다." in documents[0].page_content
    assert documents[0].metadata["sprtSclLmtYn"] == "Y"
    assert documents[0].metadata["sprtSclCnt"] == "30"
    assert documents[0].metadata["sprtArvlSeqYn"] == "Y"


def test_build_documents_names_registration_institution_explicitly():
    documents, _ = build_documents([{
        "plcyNo": "POLICY-001",
        "rgtrInstCdNm": "서울특별시",
    }])

    assert (
        documents[0].metadata["registrationInstitution"]
        == "서울특별시"
    )
    assert "region" not in documents[0].metadata


def test_policy_metadata_labels_cover_every_ingested_metadata_key():
    documents, _ = build_documents([{"plcyNo": "POLICY-001"}])

    assert set(POLICY_METADATA_LABELS) == set(documents[0].metadata)
