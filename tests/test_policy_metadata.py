from src.policy.utils import (
    POLICY_METADATA_LABELS,
    get_policy_metadata_label,
)


def test_policy_metadata_labels_include_derived_region_fields():
    assert POLICY_METADATA_LABELS["applicationEndYmd"] == "신청마감일"
    assert (
        POLICY_METADATA_LABELS["registrationInstitution"]
        == "등록기관명"
    )
    assert "region" not in POLICY_METADATA_LABELS
    assert POLICY_METADATA_LABELS["region_11"] == "서울지원가능여부"


def test_get_policy_metadata_label_keeps_unknown_keys_readable():
    assert get_policy_metadata_label("새로운필드") == "새로운필드"
