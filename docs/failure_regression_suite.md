# 실패 케이스 회귀 검증

이 문서는 청년정책 RAG에서 실제로 관찰된 실패 사례를 수정 후에도 반복 실행할 수
있도록 관리하는 방법을 정의한다. 과거 실행 결과를 그대로 정답으로 고정하지 않고,
사용자 입력과 실패 의도, 구조적 계약, 정성 검토 기준을 보존한다.

## 기준 파일

- 케이스: `docs/failure_regression_cases.yaml`
- 실행기: `scripts/rerun_failed_answer_cases.py`
- 실행 결과: `data/eval/failure_regression_runs/<run-id>.json`
- 최초 근거 문서:
  - `docs/answer_generation_review_10_cases_20260724.md`
  - `docs/answer_generation_related_failure_rerun_20260724.md`

YAML이 재실행의 기준 데이터다. 최초 근거 문서는 당시 출력과 분석을 보존하는
과거 기록이며, 현재 Checker의 verdict 계약이나 그래프 구조를 설명하는 문서로
사용하지 않는다.

## 보존된 실패 사례

| ID | 과거 실패 | 핵심 회귀 조건 |
|---|---|---|
| AG-04 | 경기 학자금대출 이자 지원 검색 누락 | 최대 검색 횟수 준수, 부적합 정책과 혼동 금지, 미발견 시 환각 금지 |
| AG-05 | 소득 단위·가구 기준 없이 자격 확정 | 소득 자격을 확정하지 않고 추가 확인 필요성을 유지 |
| AG-09 | 정책번호가 다른 동일 정책 중복, 지역 과단정 | 정책 ID와 정책명 중복 금지, 지역 정책을 예시로 구분 |
| AG-10 | 소득 과단정, 상세정보 후속 질문에서 재검색 | 2턴 검색 횟수 0, 직전 정책 ID 유지, 미제공 정보 명시 |

## 케이스 스키마

각 케이스는 다음 정보를 가진다.

- `case_id`, `title`: 고정 식별자와 설명
- `historical_failure`: 처음 관찰된 실패 계층과 요약
- `profile`: 실제 그래프에 전달할 사용자 프로필
- `exclude_expired`: 만료 정책 하드 필터 적용 여부. 기본값은 `true`
- `turns`: 같은 `thread_id`에서 순서대로 실행할 질문
- `automated_checks`: 상태만으로 결정할 수 있는 검사
- `review_criteria`: 답변 의미를 사람이 확인할 기준

현재 지원하는 자동 검사는 다음과 같다.

- `expected_retrieval_count`: 검색 횟수가 정확히 일치하는지 확인
- `max_retrieval_count`: 설정된 최대 검색 횟수를 넘지 않는지 확인
- `preserve_previous_policy_ids`: 직전 턴 정책 ID와 동일한지 확인
- `no_duplicate_policy_ids`: 같은 정책 ID가 중복되지 않는지 확인
- `no_duplicate_policy_titles`: 정책번호가 달라도 정책명이 같은 중복이 없는지 확인

자동 검사는 문장 표현을 평가하지 않는다. 소득 자격 과단정, 정책 미발견을 정책
부재로 단정하는 표현, 지역 조건 설명처럼 의미 해석이 필요한 항목은
`review_criteria`에 따라 답변을 직접 읽고 판정한다.

## 실행 방법

전체 실패 사례를 실행한다.

```bash
uv run python scripts/rerun_failed_answer_cases.py \
  --run-id <run-id>
```

일부 사례만 실행할 수 있다.

```bash
uv run python scripts/rerun_failed_answer_cases.py \
  --run-id <run-id> \
  --case-id AG-05 \
  --case-id AG-10
```

CI나 로컬 회귀 확인에서 자동 검사 실패를 종료 코드로 받고 싶다면 다음 옵션을
추가한다.

```bash
uv run python scripts/rerun_failed_answer_cases.py \
  --run-id <run-id> \
  --fail-on-automated-check
```

만료 필터가 검색 실패 원인인지 분리할 때만 진단 옵션을 사용한다.

```bash
uv run python scripts/rerun_failed_answer_cases.py \
  --run-id <run-id> \
  --case-id AG-04 \
  --include-expired
```

이 옵션은 선택한 케이스의 `exclude_expired`를 `false`로 덮어쓴다. 기본 회귀
실행 결과와 섞지 말고 별도 실행 ID를 사용한다.

실행기는 각 턴의 다음 정보를 JSON에 저장한다.

- 실제 검색 횟수와 마지막 Query
- 선택 정책 ID·정책명
- 선택 정책에 보존된 Checker verdict와 reasoning
- 현재 턴에 수행된 전체 Checker 결과
- 최종 답변
- 자동 검사별 기대값·실제값·통과 여부
- 사람이 확인할 정성 기준

모델 API나 검색 API 호출이 중간에 실패해도 완료된 턴까지의 부분 결과와
`execution_error`를 결과 파일에 저장한다. 오류가 있으면 실행 자체는 실패 종료되므로
회귀 PASS로 집계하지 않는다.

## 결과 판정 절차

1. `automated_summary.failed_turns`가 0인지 확인한다.
2. 각 턴의 `automated_checks`에서 실패한 상태 계약을 확인한다.
3. `answer`를 해당 턴의 `review_criteria`와 대조한다.
4. 자동 검사와 정성 검사를 분리해 최종 PASS·PARTIAL·FAIL을 기록한다.
5. 새 실패가 기존 유형과 다르면 YAML에 새 케이스를 추가하고, 같은 유형의 변형이면
   기존 케이스에 턴이나 정성 기준을 보강한다.

## 해석 시 주의사항

- LLM 출력과 검색 순위는 달라질 수 있으므로 특정 정책 ID나 답변 전문을 정답으로
  고정하지 않는다.
- `checked_policies`의 탈락 정책 제외는 현재 사용자 턴의 재검색에만 누적된다.
- 후속 턴에서는 탈락 이력이 초기화되지만 `active_policies`는 유지된다.
- `fit_needs_clarification`은 통과 verdict다. 최종 답변도 Checker의 불확실성을
  유지해야 한다.
- Langfuse trace 저장 여부는 이 회귀 결과의 통과 조건에 포함하지 않는다.
