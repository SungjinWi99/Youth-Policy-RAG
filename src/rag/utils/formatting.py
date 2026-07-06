from collections.abc import Sequence

from langchain_core.documents import Document

from src.rag.state import RAGUserProfile


def format_optional(value) -> str:
    return str(value) if value not in (None, "") else "미입력"


def format_context_value(value) -> str:
    return str(value) if value not in (None, "") else "미제공"


def format_context_unit_value(value, unit: str) -> str:
    formatted_value = format_context_value(value)
    if formatted_value == "미제공":
        return formatted_value
    return f"{formatted_value}{unit}"


def format_context_range(start, end, unit: str = "") -> str:
    if start in (None, "") and end in (None, ""):
        return "미제공"
    return (
        f"{format_context_unit_value(start, unit)}"
        f" ~ {format_context_unit_value(end, unit)}"
    )


def format_income_condition(metadata: dict) -> str:
    if metadata.get("incomePolicy") == "all":
        return "제한 없음"
    if metadata.get("incomePolicy") == "unknown":
        return "확인 필요"
    return format_context_range(
        metadata.get("earnMinAmt"),
        metadata.get("earnMaxAmt"),
    )


def format_age_condition(metadata: dict) -> str:
    if metadata.get("agePolicy") == "all":
        return "제한 없음"
    if metadata.get("agePolicy") == "unknown":
        return "확인 필요"
    return format_context_range(
        metadata.get("sprtTrgtMinAge"),
        metadata.get("sprtTrgtMaxAge"),
        "세",
    )


def format_user_profile(user: RAGUserProfile) -> str:
    return "\n".join([
        f"나이: {format_optional(user.get('age'))}",
        f"성별: {format_optional(user.get('gender'))}",
        f"소득수준: {format_optional(user.get('income'))}",
        f"주거지: {format_optional(user.get('region'))}",
        f"직업: {format_optional(user.get('job'))}",
    ])


def format_doc(doc: Document, index: int) -> str:
    metadata = doc.metadata
    return f"""
[검색 결과 {index}]
검색 문서:
{doc.page_content}

정책번호: {format_context_value(metadata.get("plcyNo"))}
대분류: {format_context_value(metadata.get("lclsfNm"))}
중분류: {format_context_value(metadata.get("mclsfNm"))}
주관기관: {format_context_value(metadata.get("sprvsnInstCdNm"))}
운영기관: {format_context_value(metadata.get("operInstCdNm"))}
지역: {format_context_value(metadata.get("region"))}
지원 연령: {format_age_condition(metadata)}
소득 조건: {format_income_condition(metadata)}
사업 기간: {format_context_range(metadata.get("bizPrdBgngYmd"), metadata.get("bizPrdEndYmd"))}
사업 기간 기타 설명: {format_context_value(metadata.get("bizPrdEtcCn"))}
신청 기간: {format_context_value(metadata.get("aplyYmd"))}
신청 방법: {format_context_value(metadata.get("plcyAplyMthdCn"))}
신청 URL: {format_context_value(metadata.get("aplyUrlAddr"))}
참고 URL 1: {format_context_value(metadata.get("refUrlAddr1"))}
참고 URL 2: {format_context_value(metadata.get("refUrlAddr2"))}
참여 대상: {format_context_value(metadata.get("ptcpPrpTrgtCn"))}
추가 신청 자격: {format_context_value(metadata.get("addAplyQlfcCndCn"))}
제출 서류: {format_context_value(metadata.get("sbmsnDcmntCn"))}
심사 방법: {format_context_value(metadata.get("srngMthdCn"))}
직업 코드: {format_context_value(metadata.get("jobCd"))}
혼인 상태 코드: {format_context_value(metadata.get("mrgSttsCd"))}
""".strip()


def format_docs(docs: Sequence[Document]) -> str:
    return "\n\n---\n\n".join(
        format_doc(doc, index)
        for index, doc in enumerate(docs, start=1)
    )
