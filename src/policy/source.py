from __future__ import annotations

import time
from typing import Any

import requests

from src.policy.corpus import validate_policies


API_URL = "https://www.youthcenter.go.kr/go/ythip/getPlcy"
DEFAULT_PAGE_SIZE = 100
DEFAULT_REQUEST_DELAY = 0.3
DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 1.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class YouthPolicyApiError(RuntimeError):
    """온통청년 API 응답을 신뢰할 수 없을 때 발생한다."""


def fetch_page(
    session: requests.Session,
    *,
    api_key: str,
    page_num: int,
    page_size: int,
    timeout: float,
) -> dict[str, Any]:
    params = {
        "apiKeyNm": api_key,
        "pageNum": page_num,
        "pageSize": page_size,
        "rtnType": "json",
    }
    try:
        response = session.get(
            API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        # requests 예외 문자열에는 인증키가 포함된 요청 URL이 들어갈 수 있다.
        raise YouthPolicyApiError(
            f"{page_num}페이지 HTTP 요청에 실패했습니다 "
            f"({type(exc).__name__})."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise YouthPolicyApiError(
            f"{page_num}페이지 응답이 JSON이 아닙니다."
        ) from exc
    if not isinstance(payload, dict):
        raise YouthPolicyApiError(
            f"{page_num}페이지 응답 최상위 값이 JSON 객체가 아닙니다."
        )

    result_code = payload.get("resultCode")
    if str(result_code) != "200":
        result_message = str(payload.get("resultMessage") or "메시지 없음")
        raise YouthPolicyApiError(
            f"{page_num}페이지 API 오류: code={result_code}, "
            f"message={result_message}"
        )
    return payload


def fetch_page_with_retry(
    session: requests.Session,
    *,
    api_key: str,
    page_num: int,
    page_size: int,
    timeout: float,
    max_attempts: int,
    retry_backoff: float,
) -> dict[str, Any]:
    last_error: YouthPolicyApiError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch_page(
                session,
                api_key=api_key,
                page_num=page_num,
                page_size=page_size,
                timeout=timeout,
            )
        except YouthPolicyApiError as exc:
            last_error = exc
            if attempt < max_attempts and retry_backoff > 0:
                time.sleep(retry_backoff * attempt)
    assert last_error is not None
    raise last_error


def parse_result_page(
    payload: dict[str, Any],
    *,
    page_num: int,
) -> tuple[list[dict[str, Any]], int, int]:
    result = payload.get("result")
    if not isinstance(result, dict):
        raise YouthPolicyApiError(
            f"{page_num}페이지 응답에 result 객체가 없습니다."
        )
    pagination = result.get("pagging")
    if not isinstance(pagination, dict):
        raise YouthPolicyApiError(
            f"{page_num}페이지 응답에 pagging 객체가 없습니다."
        )
    try:
        total_count = int(pagination["totCount"])
        actual_page_size = int(pagination["pageSize"])
    except (KeyError, TypeError, ValueError) as exc:
        raise YouthPolicyApiError(
            f"{page_num}페이지의 페이징 값이 올바르지 않습니다."
        ) from exc
    if total_count < 0 or actual_page_size < 1:
        raise YouthPolicyApiError(
            f"{page_num}페이지의 페이징 값이 유효하지 않습니다."
        )

    policies = validate_policies(
        result.get("youthPolicyList"),
        source=f"API {page_num}페이지",
    )
    return policies, total_count, actual_page_size


def fetch_policies(
    *,
    api_key: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    max_pages: int | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages는 1 이상이어야 합니다.")
    owns_session = session is None
    session = session or requests.Session()
    try:
        first_payload = fetch_page_with_retry(
            session,
            api_key=api_key,
            page_num=1,
            page_size=page_size,
            timeout=timeout,
            max_attempts=max_attempts,
            retry_backoff=retry_backoff,
        )
        first_policies, total_count, actual_page_size = parse_result_page(
            first_payload,
            page_num=1,
        )
        if total_count == 0:
            return []

        total_pages = (total_count + actual_page_size - 1) // actual_page_size
        pages_to_fetch = (
            total_pages
            if max_pages is None
            else min(total_pages, max_pages)
        )
        print(
            f"온통청년 API: total={total_count}, pages={pages_to_fetch}/"
            f"{total_pages}, page_size={actual_page_size}"
        )
        policies = list(first_policies)
        for page_num in range(2, pages_to_fetch + 1):
            if request_delay > 0:
                time.sleep(request_delay)
            payload = fetch_page_with_retry(
                session,
                api_key=api_key,
                page_num=page_num,
                page_size=actual_page_size,
                timeout=timeout,
                max_attempts=max_attempts,
                retry_backoff=retry_backoff,
            )
            page_policies, page_total, page_size_from_api = parse_result_page(
                payload,
                page_num=page_num,
            )
            if (
                page_total != total_count
                or page_size_from_api != actual_page_size
            ):
                raise YouthPolicyApiError(
                    "수집 중 API 페이징 정보가 변경되었습니다. "
                    "데이터 누락 방지를 위해 다시 실행해주세요."
                )
            policies.extend(page_policies)
            print(
                f"\rAPI 수집: {page_num}/{pages_to_fetch} "
                f"({len(policies)}/{total_count})",
                end="",
                flush=True,
            )
        if pages_to_fetch > 1:
            print()

        validated = validate_policies(policies, source="API 전체 응답")
        if max_pages is None and len(validated) != total_count:
            raise YouthPolicyApiError(
                f"API 전체 건수와 수집 건수가 다릅니다: "
                f"expected={total_count}, fetched={len(validated)}"
            )
        return validated
    finally:
        if owns_session:
            session.close()
