# 청년정책 데이터 및 메타데이터 필터 현황

- 작성 기준일: 2026-07-03
- 원천 데이터: `data/raw/youth_policies.json`
- 원천 정책 수: 2,630개
- Chroma 컬렉션: `youth_policies_rag`
- 집계 기준 코드:
  - `src/policy_metadata.py`
  - `src/regions.py`
  - `src/rag/retriever.py`

이 문서는 사용자 프로필 기반 메타데이터 필터에 사용되는 속성의
정규화 규칙과 현재 데이터 분포를 기록한다.

## 1. 상태값 정의

속성마다 사용할 수 있는 상태값이 다르다.

| 상태 | 적용 속성 | 의미 | 필터 동작 |
|---|---|---|---|
| `all` | 연령, 소득 | 모든 값이 대상 | 해당 속성 조건을 통과 |
| `specific` | 연령, 소득 | 최소·최대 범위가 명시됨 | 사용자 값이 범위 안에 있을 때 통과 |
| `fixed` | 신청 기간 | 단일 신청 기간이 명시됨 | 신청 종료일이 오늘 이상일 때 통과 |
| `multi` | 신청 기간 | 신청 기간이 두 개 이상 명시됨 | 마지막 신청 종료일이 오늘 이상일 때 통과 |
| `rolling` | 신청 기간 | 상시 신청 정책 | 항상 통과 |
| `unknown` | 연령, 소득, 신청 기간 | 값이 없거나 유효하지 않아 판정 불가 | 해당 속성 조건을 통과 |

`unknown`은 해당 속성에서만 `all`과 동일하게 통과한다. 다른 속성이
불일치하면 최종 `$and` 조건에서 제외될 수 있다.

## 2. 속성별 상태 요약

| 속성 | `all` | `specific` | `fixed` | `multi` | `rolling` | `unknown` | 합계 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 연령 | 677 | 1,943 | - | - | - | 10 | 2,630 |
| 소득 | 2,595 | 28 | - | - | - | 7 | 2,630 |
| 신청 기간 | - | - | 1,305 | 3 | 432 | 890 | 2,630 |

### 2.1 연령

사용 필드:

- `sprtTrgtMinAge`
- `sprtTrgtMaxAge`
- 정규화 필드 `agePolicy`

판정 규칙:

```text
최소=0, 최대=0                 -> all
0 <= 최소 <= 최대             -> specific
빈 값, 숫자 아님, 최소 > 최대 -> unknown
```

현재 분포:

- `all`: 677개
- `specific`: 1,943개
- `unknown`: 10개
  - 최소·최대 누락 또는 숫자 아님: 1개
  - 최소값이 최대값보다 큼: 9개

필터에서 `all`과 `unknown`은 모든 사용자 연령을 통과한다.

### 2.2 소득

사용 필드:

- `earnMinAmt`
- `earnMaxAmt`
- 정규화 필드 `incomePolicy`

판정 규칙:

```text
최소=0, 최대=0                 -> all
0 <= 최소 <= 최대             -> specific
빈 값, 숫자 아님, 최소 > 최대 -> unknown
```

현재 분포:

- `all`: 2,595개
- `specific`: 28개
- `unknown`: 7개
  - 최소·최대 누락 또는 숫자 아님: 7개
- 금액 단위 이상치 후보: 1개

소득 금액의 원/만원 단위 혼재 가능성은 아직 해결하지 않았다.
`src/policy_metadata.py`에 TODO로 기록되어 있다.

필터에서 `all`과 `unknown`은 모든 사용자 소득을 통과한다.

### 2.3 신청 기간

사용 필드:

- `aplyYmd`
- `aplyPrdSeCd`
- 정규화 필드 `applicationPolicy`
- 정규화 필드 `applicationStartYmd`
- 정규화 필드 `applicationEndYmd`

원천 신청 기간 코드 분포:

| `aplyPrdSeCd` | 개수 | 현재 처리 |
|---|---:|---|
| `0057001` | 1,310 | 날짜 범위를 파싱해 `fixed`, `multi`, `unknown` 결정 |
| `0057002` | 432 | `rolling` |
| `0057003` | 888 | `unknown` |

정규화 결과:

- `fixed`: 1,305개
- `multi`: 3개
- `rolling`: 432개
- `unknown`: 890개
  - 코드 `0057003`으로 신청 날짜 미제공: 888개
  - 시작일이 종료일보다 늦은 잘못된 기간: 2개

원천 `aplyYmd`가 빈 정책은 1,320개다. 이 중 432개는
`aplyPrdSeCd=0057002`여서 `rolling`, 나머지 888개는 `unknown`이다.

`exclude_expired=True`일 때:

```text
rolling, unknown -> 포함
fixed, multi     -> applicationEndYmd >= 오늘인 경우 포함
```

`exclude_expired=False`일 때는 신청 기간으로 정책을 제외하지 않는다.

### 2.4 지역

사용 필드:

- 원천 `zipCd`
- 정규화 필드 `region_11`, `region_26`, ..., `region_52`

지역은 상태 문자열 대신 17개 시·도 boolean으로 저장한다.
정책의 `zipCd` 앞 두 자리를 시·도 코드로 사용한다.

정책 하나가 포함하는 시·도 개수:

| 포함 시·도 수 | 정책 수 |
|---:|---:|
| 1 | 2,210 |
| 2 | 4 |
| 4 | 3 |
| 8 | 1 |
| 15 | 1 |
| 16 | 2 |
| 17 | 409 |

지역 코드가 하나도 없는 정책은 0개다.

시·도별 포함 정책 수:

| 지역 | 코드 | 정책 수 |
|---|---:|---:|
| 서울 | 11 | 472 |
| 부산 | 26 | 529 |
| 대구 | 27 | 449 |
| 인천 | 28 | 631 |
| 광주 | 29 | 610 |
| 대전 | 30 | 422 |
| 울산 | 31 | 611 |
| 세종 | 36 | 459 |
| 경기 | 41 | 520 |
| 충북 | 43 | 503 |
| 충남 | 44 | 712 |
| 전남 | 46 | 540 |
| 경북 | 47 | 544 |
| 경남 | 48 | 589 |
| 제주 | 50 | 613 |
| 강원 | 51 | 494 |
| 전북 | 52 | 540 |

사용자 지역에 해당하는 boolean이 `True`인 정책만 통과한다.
전국 정책은 17개 boolean이 모두 `True`이므로 모든 지역에서 포함된다.

### 2.5 성별

원천 데이터에는 신뢰할 수 있는 구조화 성별 필드가 없다.

- 자유 텍스트 키워드 추정은 사용하지 않는다.
- `genderPolicy`는 Chroma에서 제거했다.
- 사용자 성별은 프롬프트에는 전달되지만 메타데이터 필터에는 사용하지 않는다.

### 2.6 직업

사용 가능한 원천 필드는 `jobCd`다.

- 빈 값: 0개
- 서로 다른 값: 51개
- 현재 사용자 직업과의 코드 매핑이 없으므로 필터에는 사용하지 않는다.

### 2.7 혼인 상태

사용 가능한 원천 필드는 `mrgSttsCd`다.

| 값 | 개수 |
|---|---:|
| `0055001` | 48 |
| `0055002` | 23 |
| `0055003` | 2,557 |
| 빈 값 | 2 |

현재 사용자 프로필 필드와 매핑하지 않으며 필터에도 사용하지 않는다.

### 2.8 사업 기간

사용 가능한 원천 필드:

- `bizPrdBgngYmd`
- `bizPrdEndYmd`

현황:

- 사업 시작일 누락: 957개
- 사업 종료일 누락: 957개

사업 종료일은 더 이상 신청 마감 필터에 사용하지 않는다. 신청 마감은
`aplyYmd`에서 생성한 `applicationEndYmd`를 기준으로 판단한다.

## 3. 집계에 사용한 Python 코드

프로젝트 루트에서 다음 코드를 실행하면 이 문서의 주요 수치를 다시
계산할 수 있다.

```python
import json
from collections import Counter

from src.policy_metadata import (
    build_age_metadata,
    build_application_period_metadata,
    build_income_metadata,
    parse_int,
)
from src.regions import (
    REGION_NAME_TO_CODE,
    build_region_metadata,
)


RAW_PATH = "data/raw/youth_policies.json"

with open(RAW_PATH, encoding="utf-8") as raw_file:
    policies = json.load(raw_file)


age_counts = Counter()
income_counts = Counter()
application_counts = Counter()
application_code_counts = Counter()
age_unknown_reasons = Counter()
income_unknown_reasons = Counter()
application_unknown_codes = Counter()
region_coverage_counts = Counter()
region_policy_counts = Counter()


for policy in policies:
    age_metadata = build_age_metadata(
        policy.get("sprtTrgtMinAge"),
        policy.get("sprtTrgtMaxAge"),
    )
    income_metadata = build_income_metadata(
        policy.get("earnMinAmt"),
        policy.get("earnMaxAmt"),
    )
    application_metadata = build_application_period_metadata(
        policy.get("aplyYmd"),
        policy.get("aplyPrdSeCd"),
    )
    region_metadata = build_region_metadata(
        policy.get("zipCd")
    )

    age_counts[age_metadata["agePolicy"]] += 1
    income_counts[income_metadata["incomePolicy"]] += 1
    application_counts[
        application_metadata["applicationPolicy"]
    ] += 1
    application_code_counts[
        str(policy.get("aplyPrdSeCd") or "")
    ] += 1

    min_age = parse_int(policy.get("sprtTrgtMinAge"))
    max_age = parse_int(policy.get("sprtTrgtMaxAge"))
    if age_metadata["agePolicy"] == "unknown":
        if min_age is None or max_age is None:
            age_unknown_reasons["missing_or_non_numeric"] += 1
        elif min_age > max_age:
            age_unknown_reasons["inverted_range"] += 1

    min_income = parse_int(policy.get("earnMinAmt"))
    max_income = parse_int(policy.get("earnMaxAmt"))
    if income_metadata["incomePolicy"] == "unknown":
        if min_income is None or max_income is None:
            income_unknown_reasons["missing_or_non_numeric"] += 1
        elif min_income > max_income:
            income_unknown_reasons["inverted_range"] += 1

    if application_metadata["applicationPolicy"] == "unknown":
        application_unknown_codes[
            str(policy.get("aplyPrdSeCd") or "")
        ] += 1

    active_region_codes = [
        code
        for code in REGION_NAME_TO_CODE.values()
        if region_metadata[f"region_{code}"]
    ]
    region_coverage_counts[len(active_region_codes)] += 1
    for code in active_region_codes:
        region_policy_counts[code] += 1


def count_blank(field_name: str) -> int:
    return sum(
        not str(policy.get(field_name) or "").strip()
        for policy in policies
    )


income_unit_outliers = sum(
    max(
        build_income_metadata(
            policy.get("earnMinAmt"),
            policy.get("earnMaxAmt"),
        )["earnMinAmt"],
        build_income_metadata(
            policy.get("earnMinAmt"),
            policy.get("earnMaxAmt"),
        )["earnMaxAmt"],
    )
    >= 1_000_000
    for policy in policies
)


print("total:", len(policies))
print("age:", age_counts)
print("age_unknown_reasons:", age_unknown_reasons)
print("income:", income_counts)
print("income_unknown_reasons:", income_unknown_reasons)
print("income_unit_outliers:", income_unit_outliers)
print("application:", application_counts)
print("application_codes:", application_code_counts)
print("application_unknown_codes:", application_unknown_codes)
print("application_period_blank:", count_blank("aplyYmd"))
print("region_coverage:", sorted(region_coverage_counts.items()))
print("region_policy_counts:", dict(region_policy_counts))
print("job_blank:", count_blank("jobCd"))
print(
    "job_unique:",
    len({str(policy.get("jobCd") or "") for policy in policies}),
)
print("marriage_blank:", count_blank("mrgSttsCd"))
print(
    "marriage_values:",
    Counter(str(policy.get("mrgSttsCd") or "") for policy in policies),
)
print("business_start_blank:", count_blank("bizPrdBgngYmd"))
print("business_end_blank:", count_blank("bizPrdEndYmd"))
```

## 4. 검증 상태

`tests/test_chroma_filter_integration.py`가 다음을 실제 Chroma 컬렉션에서
검증한다.

- 원천 정책 ID와 Chroma 정책 ID 일치
- Python 기준 판정과 Chroma `$where` 결과 일치
- 연령·소득·신청 기간 `unknown` 포함
- 마감 필터 on/off 포함 관계
- 지역 boolean 최소 1개 이상
- `genderPolicy`가 남아 있지 않음

실행 명령:

```bash
venv/bin/python -m unittest tests.test_chroma_filter_integration -v
```
