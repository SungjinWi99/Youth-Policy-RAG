from langchain_core.documents import Document

from src.rag.utils.formatting import format_doc


def test_format_doc_formats_capacity_and_first_come_metadata():
    formatted = format_doc(Document(
        page_content="정책 본문",
        metadata={
            "sprtSclLmtYn": "Y",
            "sprtSclCnt": "30",
            "sprtArvlSeqYn": "N",
        },
    ), 1)

    assert "지원 규모 제한: 예" in formatted
    assert "지원 규모: 30" in formatted
    assert "선착순 여부: 아니오" in formatted


def test_format_doc_does_not_present_zero_as_a_support_scale():
    formatted = format_doc(Document(
        page_content="정책 본문",
        metadata={"sprtSclCnt": "0"},
    ), 1)

    assert "지원 규모: 미제공" in formatted


def test_format_doc_uses_zip_codes_for_supported_regions():
    formatted = format_doc(Document(
        page_content="정책 본문",
        metadata={
            "registrationInstitution": "등록기관명",
            "zipCd": "11110,29140",
        },
    ), 1)

    assert "지원 지역: 서울, 광주" in formatted
    assert "등록기관명" not in formatted


def test_format_doc_presents_all_supported_regions_as_nationwide():
    nationwide_zip_codes = (
        "11110,26110,27110,28110,29110,30110,31110,36110,"
        "41111,43111,44131,46110,47111,48121,50110,51110,52111"
    )
    formatted = format_doc(Document(
        page_content="정책 본문",
        metadata={"zipCd": nationwide_zip_codes},
    ), 1)

    assert "지원 지역: 전국" in formatted


def test_format_doc_marks_missing_zip_codes_as_unavailable():
    formatted = format_doc(Document(
        page_content="정책 본문",
        metadata={},
    ), 1)

    assert "지원 지역: 미제공" in formatted
