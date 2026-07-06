import re
from datetime import datetime
from typing import Any, Literal, get_args


ROLLING_APPLICATION_PERIOD_CODE = "0057002"
APPLICATION_PERIOD_PATTERN = re.compile(
    r"(\d{8})\s*~\s*(\d{8})"
)


def parse_int(value: Any) -> int | None:
    try:
        text = str(value or "").strip()
        return int(text) if text else None
    except (TypeError, ValueError):
        return None


def build_age_metadata(
    min_age_value: Any,
    max_age_value: Any,
) -> dict[str, int | str]:
    parsed_min_age = parse_int(min_age_value)
    parsed_max_age = parse_int(max_age_value)

    if parsed_min_age == 0 and parsed_max_age == 0:
        age_policy = "all"
    elif (
        parsed_min_age is not None
        and parsed_max_age is not None
        and 0 <= parsed_min_age <= parsed_max_age
    ):
        age_policy = "specific"
    else:
        age_policy = "unknown"

    return {
        "sprtTrgtMinAge": parsed_min_age or 0,
        "sprtTrgtMaxAge": parsed_max_age or 0,
        "agePolicy": age_policy,
    }


def build_income_metadata(
    min_income_value: Any,
    max_income_value: Any,
) -> dict[str, int | str]:
    parsed_min_income = parse_int(min_income_value)
    parsed_max_income = parse_int(max_income_value)

    if parsed_min_income == 0 and parsed_max_income == 0:
        income_policy = "all"
    elif (
        parsed_min_income is not None
        and parsed_max_income is not None
        and 0 <= parsed_min_income <= parsed_max_income
    ):
        income_policy = "specific"
    else:
        income_policy = "unknown"

    # TODO: 원천 데이터의 소득 금액 단위(원/만원) 혼재를 확인해
    # 하나의 연 소득 단위로 정규화한다.
    return {
        "earnMinAmt": parsed_min_income or 0,
        "earnMaxAmt": parsed_max_income or 0,
        "incomePolicy": income_policy,
    }


def _valid_yyyymmdd(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y%m%d")
        return True
    except ValueError:
        return False


def build_application_period_metadata(
    application_period_value: Any,
    application_period_code: Any,
) -> dict[str, int | str]:
    period_text = str(application_period_value or "").strip()
    periods = []

    for start, end in APPLICATION_PERIOD_PATTERN.findall(period_text):
        if (
            _valid_yyyymmdd(start)
            and _valid_yyyymmdd(end)
            and start <= end
        ):
            periods.append((int(start), int(end)))

    if periods:
        application_policy = (
            "fixed"
            if len(periods) == 1
            else "multi"
        )
        start_yyyymmdd = min(start for start, _ in periods)
        end_yyyymmdd = max(end for _, end in periods)
    elif (
        str(application_period_code or "").strip()
        == ROLLING_APPLICATION_PERIOD_CODE
    ):
        application_policy = "rolling"
        start_yyyymmdd = 0
        end_yyyymmdd = 0
    else:
        application_policy = "unknown"
        start_yyyymmdd = 0
        end_yyyymmdd = 0

    return {
        "applicationPolicy": application_policy,
        "applicationStartYmd": start_yyyymmdd,
        "applicationEndYmd": end_yyyymmdd,
    }


RegionName = Literal[
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "충북",
    "충남",
    "전남",
    "경북",
    "경남",
    "제주",
    "강원",
    "전북",
]

REGION_NAMES: tuple[RegionName, ...] = get_args(RegionName)

REGION_NAME_TO_CODE = {
    "서울": "11",
    "부산": "26",
    "대구": "27",
    "인천": "28",
    "광주": "29",
    "대전": "30",
    "울산": "31",
    "세종": "36",
    "경기": "41",
    "충북": "43",
    "충남": "44",
    "전남": "46",
    "경북": "47",
    "경남": "48",
    "제주": "50",
    "강원": "51",
    "전북": "52",
}

REGION_CODES: tuple[str, ...] = tuple(REGION_NAME_TO_CODE.values())

REGION_ALIASES = {
    "서울시": "서울",
    "서울특별시": "서울",
    "부산시": "부산",
    "부산광역시": "부산",
    "대구시": "대구",
    "대구광역시": "대구",
    "인천시": "인천",
    "인천광역시": "인천",
    "광주시": "광주",
    "광주광역시": "광주",
    "대전시": "대전",
    "대전광역시": "대전",
    "울산시": "울산",
    "울산광역시": "울산",
    "세종시": "세종",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주도": "제주",
    "제주특별자치도": "제주",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "전라북도": "전북",
    "전북특별자치도": "전북",
}

LEGACY_SIDO_CODE_ALIASES = {
    "42": "51",
    "45": "52",
}


def normalize_region_name(region: str | None) -> str | None:
    if not region:
        return None

    normalized = region.strip().replace(" ", "")
    if normalized in REGION_NAME_TO_CODE:
        return normalized
    return REGION_ALIASES.get(normalized)


def region_name_to_code(region: str | None) -> str | None:
    normalized = normalize_region_name(region)
    if normalized is None:
        return None
    return REGION_NAME_TO_CODE[normalized]


def extract_sido_codes(zip_cd: str | None) -> set[str]:
    region_codes = set()
    for value in str(zip_cd or "").split(","):
        code = value.strip()
        if len(code) != 5 or not code.isdigit():
            continue

        sido_code = LEGACY_SIDO_CODE_ALIASES.get(code[:2], code[:2])
        if sido_code in REGION_CODES:
            region_codes.add(sido_code)

    return region_codes


def region_metadata_key(region_code: str) -> str:
    if region_code not in REGION_CODES:
        raise ValueError(f"지원하지 않는 시·도 코드입니다: {region_code}")
    return f"region_{region_code}"


def build_region_metadata(zip_cd: str | None) -> dict[str, bool]:
    policy_region_codes = extract_sido_codes(zip_cd)
    return {
        region_metadata_key(code): code in policy_region_codes
        for code in REGION_CODES
    }
