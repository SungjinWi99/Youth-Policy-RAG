# RAG 후속 작업 TODO

- 작성일: 2026-07-02
- 기준 코드: `src/rag/`, `src/chat/`, `src/user/`
- 관련 문서: `docs/private/rag_architecture_decisions.md`
- 우선순위: P0는 안정성 문제, P1은 품질·운영 문제, P2는 구조 개선

## P0. Context overflow와 대화 압축 보강

### 현재 문제

현재 Planner와 Generator 호출 전 크기를 각각 검사하지만, prompt가 임계값을
넘고 오래된 대화가 있을 때만 `summarize`로 보낸다.

다음 경우에는 안전하게 처리되지 않는다.

- 현재 사용자 메시지 하나가 지나치게 길지만 요약할 과거 메시지는 없는 경우
- 검색 문서가 prompt 대부분을 차지하는 경우
- `conversation_summary` 자체가 길어진 경우
- 요약 대상 메시지가 summary model의 context limit을 초과하는 경우
- 요약 후에도 Planner 또는 Generator prompt가 context limit을 초과하는 경우
- 현재 요약 노드 이후 크기를 다시 검사하지 않는 문제

### 해야 할 일

- [ ] `trigger_tokens`와 실제 호출을 금지할 hard limit을 구분한다.
  - `summary_trigger_ratio`: 미리 압축을 시작하는 기준
  - `max_prompt_tokens`: LLM 호출을 허용하는 최대 입력
  - `reserved_output_tokens`: 답변 생성을 위해 남겨 둘 토큰
- [ ] 현재 사용자 입력만으로 허용 크기를 넘는지 그래프 진입 전에 검사한다.
  - FastAPI schema의 문자 수 제한은 빠른 1차 방어선으로 사용
  - 실제 graph 경계에서는 token 기준으로 다시 검사
  - 초과 시 `UserInputTooLongError`로 종료
- [ ] Retrieval query에도 별도 길이 제한을 둔다.
  - 매우 긴 사용자 입력이 embedding model에 그대로 전달되는 문제 방지
- [ ] Planner와 Generator의 context guard를 3방향 결과로 확장한다.

```python
Literal["summarize", "continue", "overflow"]
```

- [ ] `summarize` 이후 다시 context 크기를 검사한다.

```text
summarize
  -> context guard
       -> 다음 LLM 노드
       -> summarize
       -> overflow
```

- [ ] 무한 요약 루프를 막는다.
  - `compaction_attempted` 또는 `compaction_count`를 request-scoped State로 관리
  - 직전 압축에서 제거한 메시지가 없으면 다시 요약하지 않음
- [ ] 요약할 과거 메시지가 없는데 한도를 초과하면 `overflow`로 보낸다.
- [ ] 검색 문서에 token budget을 적용한다.
  - 우선순위가 낮은 문서 제거
  - 문서별 최대 길이 제한
  - 필요하면 top-k와 prompt에 넣는 문서 수를 분리
- [ ] 누적 `conversation_summary`의 최대 크기와 재압축 정책을 정한다.
- [ ] summary LLM 입력도 한 번에 한도를 넘을 수 있으므로 batch 또는 계층형 요약 필요 여부를 결정한다.

### 수정 대상

- `src/rag/summarizer.py`
  - trigger와 hard limit 구분
  - 요약 가능 여부와 단순 크기 초과 상태 분리
- `src/rag/graph.py`
  - 압축 후 context 재검사
  - `overflow` 경로와 반복 방지
- `src/rag/state.py`
  - 필요하면 `compaction_count` 추가
- `src/chat/schemas.py`
  - 명백하게 비정상적인 입력을 막는 문자 수 제한
- `src/rag/utils/formatting.py` 또는 별도 context builder
  - 검색 문서 token budget 적용
- `config.yaml`, `src/config.py`
  - hard limit과 output reserve를 설정으로 관리할지 결정

### 필수 테스트

- [ ] 단일 HumanMessage만으로 hard limit을 넘으면 요약하지 않고 명시적 오류
- [ ] 오래된 대화 압축 후 LLM prompt가 허용 범위로 감소
- [ ] 압축 후에도 초과하면 LLM을 호출하지 않고 `overflow`
- [ ] 검색 문서만으로 한도를 넘을 때 문서 budget 적용
- [ ] summary 입력 자체가 지나치게 큰 경우 처리
- [ ] 압축 노드가 무한 반복되지 않음

### 완료 조건

- LLM provider의 context overflow 예외에 의존하기 전에 애플리케이션이 크기 초과를 판별한다.
- 모든 Planner와 Generator 호출 직전 prompt가 hard limit 이하임을 보장한다.
- 압축으로 해결할 수 없는 입력은 사용자에게 명시적인 오류로 반환한다.

## P0. 예외 처리와 오류 응답 계약

### 현재 문제

- `src/chat/router.py`의 `try/except`는 `StreamingResponse`가 반환된 뒤 async generator에서 발생한 예외를 잡지 못한다.
- `print(e)`만 사용하므로 구조화된 로그와 traceback 관리가 부족하다.
- Retriever, Planner, Generator, Summarizer, Checkpointer의 외부 라이브러리
  예외가 정규화되지 않았다.
- Retrieval 실패를 빈 문서와 안전한 오류 문구로 바꾸는 현재 복구 정책이
  운영 로그를 남기지 않는다.
- 사용자 DB 쓰기 실패 시 명시적인 rollback 처리가 없다.

### 해야 할 일

- [ ] `src/rag/exceptions.py`를 추가하고 서비스 예외 계약을 정의한다.

```text
RAGServiceError
├── PolicyRetrievalError
├── ModelInvocationError
├── ConversationSummaryError
├── ContextOverflowError
└── ConversationPersistenceError
```

- [ ] 예외마다 다음 공개 정보를 정의한다.
  - `code`
  - `public_message`
  - `retryable`
  - 필요하면 HTTP status
- [ ] `PolicyRetriever`에서 vector store와 embedding 예외를 `PolicyRetrievalError`로 변환한다.
  - 검색 실패 시 `[]`를 반환하지 않음
  - 검색 결과 없음과 시스템 장애를 구분
- [ ] `RetrievalPlanner`와 `AnswerGenerator`의 sync/async LLM 호출 오류를
  `ModelInvocationError`로 변환한다.
- [ ] `ConversationSummarizer`의 sync/async LLM 호출 오류를 `ConversationSummaryError`로 변환한다.
- [ ] Checkpointer의 SQLite 오류를 `ConversationPersistenceError`로 변환할 경계를 결정한다.
- [ ] 스트리밍 오류 계약을 추가한다.

```text
정상: metadata -> chunk... -> done
실패: metadata? -> chunk? -> error
```

- [ ] `error` SSE data 형식을 고정한다.

```json
{
  "type": "error",
  "data": {
    "code": "POLICY_RETRIEVAL_FAILED",
    "message": "정책 검색에 실패했습니다.",
    "retryable": true
  }
}
```

- [ ] `error`와 `done`을 동시에 terminal event로 보내지 않는다.
- [ ] 클라이언트 연결 종료에 따른 `asyncio.CancelledError`는 변환하지 않고 재전파한다.
- [ ] 강제 Retrieval 실패 정책을 결정한다.
  - 권장: 답변 중단 후 retryable error
- [ ] `print()` 대신 모듈별 `logging.getLogger(__name__)`를 사용한다.
- [ ] 예외는 최종 처리 경계에서 한 번만 traceback과 함께 기록한다.
- [ ] 질문 원문, 전체 프로필, 검색 문서, API key는 기본 로그에 남기지 않는다.
- [ ] `UserProfile.create/update/delete`의 DB commit 실패 시 rollback한다.
- [ ] 장기적으로 `UserProfile` model의 `HTTPException` 의존성을 domain exception으로 분리한다.

### 수정 대상

- 신규 `src/rag/exceptions.py`
- `src/rag/retriever.py`
- `src/rag/planner.py`
- `src/rag/generator.py`
- `src/rag/summarizer.py`
- `src/rag/graph.py`
- `src/factory.py`
- `src/chat/router.py`
- `src/user/models.py`
- 필요하면 `main.py`의 공통 exception handler와 logging 설정

### 필수 테스트

- [ ] 강제 Retrieval 실패 시 SSE `error`로 종료
- [ ] 선택 Retrieval 실패 시 기존 문서 보존 여부와 답변 정책 검증
- [ ] Planner·Generator LLM 실패 시 내부 provider 예외 문자열 미노출
- [ ] 일부 token 전송 후 실패해도 terminal `error` 발생
- [ ] 클라이언트 취소 시 `CancelledError` 재전파
- [ ] DB commit 실패 시 rollback 호출
- [ ] 알 수 없는 예외는 일반화된 `INTERNAL_ERROR`로 변환

### 완료 조건

- 사용자에게는 안정적인 오류 code와 안전한 message만 전달한다.
- 운영 로그에는 원인 traceback과 실행 위치가 남는다.
- 스트리밍 도중 실패해도 클라이언트가 정상 종료와 실패 종료를 구분한다.

## P1. Planner 검색어 품질 검증

### 현재 문제

강제 검색과 선택 검색 모두 `RetrievalPlanner`가 대화와 마지막 검색어를 참고해
query를 작성한다. 실제 운영 모델이 모호한 후속 질문을 얼마나 독립적인
검색어로 바꾸는지는 아직 검증되지 않았다.

프로필 또는 `exclude_expired`가 변경된 뒤 사용자가 다음처럼 짧게 물으면 검색 품질이 낮을 수 있다.

```text
그럼 나는?
지금은 가능해?
```

### 해야 할 일

- [x] Planner에 State 사용자 프로필을 전달하지 않도록 분리
- [x] 키워드형 압축 대신 독립적인 자연어 query를 작성하도록 prompt 수정
- [ ] 고정 평가셋으로 Solar Pro3의 query 품질을 반복 검증
- [ ] 생성된 query가 대화 없이 이해 가능한지 평가
- [ ] `last_retrieval_query`와 현재 질문을 적절히 결합하는지 평가
- [ ] embedding model에 전달할 query 길이 제한
- [ ] 검색어 생성 LLM 호출이 추가된 비용과 retrieval 품질 개선 폭을 비교

### 평가 항목

- 짧은 대명사형 후속 질문의 retrieval Recall@K
- 프로필 변경 직후 기존 정책 재검색 정확도
- query rewrite 추가에 따른 latency와 LLM 호출 비용

## P1. Retrieval Planner 품질 평가

- [ ] 실제 운영 모델인 Solar Pro3에서 구조화된 검색 판단을 실호출로 검증한다.
- [ ] 고정 평가셋을 만든다.
  - 기존 문서로 답해야 하는 질문
  - 새 검색이 필요한 질문
  - 다른 정책 또는 추가 추천 요청
  - 최신·현재 신청 가능 여부 재확인
  - 모호한 후속 질문
- [ ] 다음 지표를 측정한다.
  - 검색 필요 질문의 `needs_retrieval` recall
  - 검색 불필요 질문의 `needs_retrieval=false` precision
  - 요청당 평균 LLM 호출 수
  - 요청당 평균 vector search 횟수
  - 전체 latency
- [ ] 검색 누락과 과다 검색 사례를 LangSmith trace로 분석한다.
- [ ] 원문 query와 Planner query의 retrieval 결과를 A/B 평가한다.

## P1. 실행 상태와 관측성

- [ ] 스트리밍 응답으로 현재 실행 중인 graph node를 전달할지 결정한다.
- [ ] 적용한다면 LangGraph custom stream과 `get_stream_writer()`를 사용한다.
- [ ] `updates`는 노드 완료 이벤트라는 점과 실행 시작 이벤트를 구분한다.
- [ ] 공개 SSE 이벤트 이름을 고정한다.

```text
node_started
metadata
chunk
done
error
```

- [ ] Retrieval 판단, 검색 횟수, 요약 실행 여부를 trace metadata로 남긴다.
- [ ] 로그와 LangSmith trace에 공통 request 또는 thread 식별자를 연결한다.
- [ ] 사용자 질문과 프로필 같은 민감 정보의 로그·trace 저장 범위를 결정한다.

## P2. Token 계산 정확도

- [ ] `count_tokens_approximately()` 오차를 provider별 실제 tokenizer와 비교한다.
- [ ] Upstage, OpenAI, Google provider별 token counter 지원 여부를 조사한다.
- [ ] provider tokenizer가 없을 때만 근사 계산을 fallback으로 사용할지 결정한다.
- [ ] 모델 변경 시 `max_input_tokens` 설정값을 함께 검증하는 방법을 추가한다.
- [ ] prompt input과 예상 output token을 분리해 budget을 계산한다.

## P2. 사용자 직업 조건 필터

- [ ] 현재 Chroma metadata의 `jobCd`와 사용자 `job` 값의 대응 규칙을 확인한다.
- [ ] 자유 문자열 직업 값을 코드로 정규화할지 별도 선택값으로 받을지 결정한다.
- [ ] 정책의 직업 조건이 명시되지 않은 경우 포함한다는 기존 원칙을 유지한다.
- [ ] `build_user_filter()`에 넣기 전에 filter 정답 데이터와 테스트 케이스를 만든다.

## P2. Graph 코드 정리 후보

- [ ] `ConversationSummarizer`와 개념적 이름 `ConversationCompactor` 중 하나로 용어를 통일할지 결정한다.
- [ ] `RAGGraph.stream_answer()`가 SSE 직렬화까지 담당하는 현재 경계를 유지할지 검토한다.
  - 장기적으로 graph는 typed event를 반환하고 Router가 SSE로 변환하는 구조가 더 명확할 수 있음
- [ ] Graph의 State adapter 메서드와 public API 부분을 파일로 더 분리할 필요가 있는지 코드 크기를 보고 판단한다.

## P2. 문서와 데모 동기화

- [ ] SSE `error`와 node 상태 이벤트가 추가되면 README 응답 형식을 갱신한다.
- [ ] `demo_streamlit.py`가 `error` 이벤트를 사용자에게 표시하도록 수정한다.
- [ ] 새 config 항목이 생기면 `config.yaml`, README, 테스트 fixture를 동시에 갱신한다.
- [x] private architecture 문서의 Mermaid 흐름도를 분리형 graph와 동기화

## 이미 완료됐거나 유지하기로 한 결정

다음 항목은 현재 TODO가 아니다.

- [x] LangChain 중심 흐름을 LangGraph로 이전
- [x] `UserProfile.user_id`를 checkpointer `thread_id`의 단일 출처로 사용
- [x] 사용자별 SQLite checkpointer로 멀티턴 대화 유지
- [x] Retrieval을 무조건 매 요청 실행하지 않고 기존 문서 재사용
- [x] 최초 요청·문서 없음·프로필 변경·만료 옵션 변경 시 결정적 강제 검색
- [x] 검색 판단·query 작성과 답변 생성을 별도 LLM 노드로 분리
- [x] 의미적으로 새 검색이 필요한지는 `RetrievalPlanner`가 구조화 출력으로 판단
- [x] 최초 요청·검색 조건 변경은 코드가 강제 검색
- [x] 검색은 일반 `retrieve` 노드에서 `PolicyRetriever`를 직접 호출
- [x] `retrieval_mode` 제거
- [x] Planner에 State 사용자 프로필을 전달하지 않음
- [x] 검색 이후 graph edge를 답변 방향으로만 연결
- [x] Answer Generator prompt에서 검색 판단과 Tool 규칙 제거
- [x] `generate_answer()`, `agenerate_answer()`, `stream_answer()` 애플리케이션 연결
- [x] context 압축 임계값을 `config.yaml`에서 관리
- [x] 개인 설계 문서를 Git에서 제외

## 추천 구현 순서

1. Context hard limit과 `overflow` 계약 정의
2. 압축 후 재검사 및 단일 대형 입력 방어
3. 서비스 예외 타입과 SSE `error` 계약 추가
4. Retriever, Planner, Generator, Summarizer 예외 정규화
5. Planner 검색어 품질 실검증
6. 멀티턴 Retrieval 판단 평가셋과 운영 모델 실검증
7. node 실행 상태와 trace metadata 추가
8. token counter와 직업 필터 개선
