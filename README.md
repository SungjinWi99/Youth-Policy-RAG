# 청년정책 RAG

온통청년 OpenAPI의 청년정책 데이터를 수집하고, 사용자 프로필을 반영해 관련
정책을 검색·안내하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- 온통청년 OpenAPI 청년정책 데이터 수집
- Chroma 기반 semantic search
- 사용자 프로필 기반 metadata filtering
- 정책 신청 방법, 기간, 자격 조건 등 상세 metadata를 포함한 답변 생성
- LangGraph `StateGraph` 기반 TurnPlanner·검색·생성 워크플로
- 현재 질문, 대화 기록, 활성 문서를 바탕으로 한 검색/재사용 분기
- 답변 전략별 생성(`policy_recommendation`, `focused_followup`, `summary` 등)
- 사용자 ID별 SQLite 대화 기록 및 연속 대화
- FastAPI SSE 스트리밍 응답
- SQLite 기반 사용자 프로필 CRUD
- 정책 ID 기반 원본 정책 상세 조회 API
- Streamlit 기반 API 테스트 화면
- Langfuse Dataset과 evaluator를 이용한 RAG 품질 평가

## 시스템 구조

```mermaid
flowchart LR
    API["온통청년 OpenAPI"] --> RAW["정책 원본 JSON"]
    RAW --> INGEST["ingest_chroma.py"]
    INGEST --> CHROMA["Chroma Vector Store"]

    CLIENT["API Client / Streamlit"] --> FASTAPI["FastAPI"]
    FASTAPI --> USERDB["SQLite User Profile"]
    FASTAPI --> GRAPH["LangGraph RAG"]
    GRAPH --> PLANNER["TurnPlanner<br/>route + strategy + queries"]
    PLANNER -->|"route=retriever"| RETRIEVE["retriever node"]
    PLANNER -->|"route=agent"| AGENT["agent node"]
    RETRIEVE --> CHROMA
    RETRIEVE --> AGENT
    AGENT --> LLM["Chat Model"]
    GRAPH --> SSE["SSE metadata / chunks"]

    EVALDATA["Evaluation JSONL"] --> LANGFUSE["Langfuse Dataset"]
    LANGFUSE --> EVALRUN["RAG Evaluation"]
    EVALRUN --> GRAPH
```

## 프로젝트 구조

```text
.
├── config.yaml                    # 모델, 저장소, 평가 설정
├── main.py                        # FastAPI 애플리케이션
├── demo_streamlit.py              # 로컬 테스트 UI
├── data/
│   ├── raw/                       # OpenAPI 원본 데이터
│   ├── chroma/                    # Chroma 영속 데이터
│   ├── sqlite/                    # 사용자 프로필 DB, 대화 checkpoint DB
│   └── eval/                      # 평가 데이터셋 JSONL
├── scripts/
│   ├── collect_data.py            # 정책 데이터 수집
│   ├── ingest_chroma.py           # 문서 임베딩 및 Chroma 적재
│   ├── generate_eval_dataset.py   # 평가 데이터 생성
│   ├── generate_planner_query_cache.py # Planner query 고정
│   ├── evaluate_retrieval.py      # local/Langfuse retrieval 평가
│   └── evaluate_rag.py            # Langfuse RAG 평가 실행
├── src/
│   ├── evaluation/                # 평가 스키마, 지표, 실험 로직
│   ├── chat/
│   │   ├── models.py              # 대화 thread ID 저장 모델
│   │   ├── router.py              # chat API
│   │   └── schemas.py             # chat request schema
│   ├── policy/                    # 정책 상세 조회 모델과 API
│   ├── rag/
│   │   ├── graph.py               # LangGraph workflow와 public API
│   │   ├── nodes/
│   │   │   ├── turn_planner.py     # 실행 경로·답변 전략·검색 질의 계획
│   │   │   ├── router.py          # TurnPlanner 호환 alias
│   │   │   ├── retriever.py       # 사용자 조건 기반 정책 검색
│   │   │   └── agent.py           # prompt 구성과 답변 생성
│   │   ├── state.py               # graph state schema
│   │   ├── prompts.py             # 라우터·생성 prompt
│   │   └── utils/formatting.py    # context와 사용자 프로필 포맷
│   ├── user/                      # 사용자 프로필 모델과 API
│   ├── config.py                  # config.yaml 로더
│   ├── database.py                # SQLite engine과 session
│   ├── dependencies.py            # FastAPI dependencies
│   ├── eval.py                    # 평가 데이터 검증 및 evaluator
│   ├── observability.py           # Langfuse tracing 설정
│   ├── checkpointer.py            # SQLite LangGraph checkpointer
│   └── factory.py                 # 모델·RAG factory
└── tests/
```

## 설치

`uv`로 Python 환경과 의존성을 관리합니다.

```bash
uv sync
```

명령은 가상환경을 활성화하지 않고 실행합니다.

```bash
uv run python -m scripts.collect_data --limit-test
uv run uvicorn main:app --reload
uv run streamlit run demo_streamlit.py
```

## 설정

모델과 검색 설정은 `config.yaml`에서 관리합니다.

```yaml
retriever:
  provider: "upstage"
  query_model: "solar-embedding-1-large-query"
  passage_model: "solar-embedding-1-large-passage"
  search_k: 3

llm:
  provider: "deepseek"
  model: "deepseek-v4-flash"

rag:
  router:
    history_window: 6
  agent:
    history_window: 10

evaluation:
  example_path: "data/eval/eval_v1_50.jsonl"
  provider: "anthropic"
  model: "claude-haiku-4-5"
  dataset_name: "PolicyRAGEval_v2_50"
  experiment_prefix: "260709"
  max_concurrency: 3
```

정책 수집과 현재 기본 모델 실행에는 provider별 API 키가 필요합니다. 예를 들어
현재 `config.yaml` 설정을 그대로 사용할 때는 다음 값을 `.env`에 둡니다.

```bash
YOUTH_API_KEY=...
UPSTAGE_API_KEY=...
DEEPSEEK_API_KEY=...
```

Langfuse tracing은 선택 사항이며, 아래 값이 모두 설정되고
`LANGFUSE_TRACING`이 활성화된 경우에만 동작합니다.

```bash
LANGFUSE_TRACING=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## 데이터 준비

모든 명령은 프로젝트 루트에서 실행합니다.

### 1. 정책 데이터 수집

API 연결과 파일 생성을 10건으로 먼저 확인할 수 있습니다.

```bash
uv run python -m scripts.collect_data --limit-test
```

전체 정책을 수집합니다.

```bash
uv run python -m scripts.collect_data
```

수집 결과는 `config.yaml`의 `data.raw` 경로에 저장됩니다.

### 2. Chroma 적재

```bash
uv run python -m scripts.ingest_chroma
```

정책명, 키워드, 카테고리, 정책 설명, 지원 내용을 임베딩하며 다음 정보는
metadata로 함께 저장합니다.

- 정책 ID와 분류
- 주관·운영 기관
- 지원 연령과 소득 조건
- 사업·신청 기간
- 신청 방법과 URL
- 추가 자격 조건과 제출 서류
- 지역, 직업, 성별, 혼인 상태

## 애플리케이션 실행

### FastAPI

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API 문서: `http://127.0.0.1:8000/docs`
- OpenAPI 스키마: `http://127.0.0.1:8000/openapi.json`

서버 시작 시 SQLite 테이블과 컴파일된 LangGraph RAG를 초기화합니다.

### LangGraph 워크플로

`src/rag/graph.py`의 그래프는 매 턴 `TurnPlanner`가 현재 질문, 최근 대화,
사용자 프로필, 현재 checkpoint에 저장된 활성 문서를 함께 보고 실행 계획을
만듭니다.

```text
START -> planner -> retriever? -> agent -> END
```

- `planner`: 구조화된 `TurnPlan`을 만들고 `route`, `answer_strategy`,
  `retrieval_queries`, `route_reason`을 결정
- `route="retriever"`: 새 정책 검색이 필요한 경우입니다. planner가 만든 검색 질의를
  순서대로 시도하고, 문서를 찾으면 다음 질의는 생략합니다.
- `route="agent"`: 활성 문서 재사용, 인사, 프로필 업데이트, 요약 등 새 검색 없이
  답할 수 있는 경우입니다.
- `retriever`: 사용자 프로필로 Chroma metadata filter를 만들고 정책 문서를 검색
- `agent`: 검색 문서, 사용자 프로필, `answer_strategy`를 prompt에 넣어 답변 생성
- 답변 전략은 `brief_reply`, `policy_recommendation`, `profile_update_response`,
  `focused_followup`, `clarifying_question`, `summary`를 지원합니다.
- planner에는 잘못된 계획을 보정하는 guard가 있어 검색 질의 누락, 문서 없는 후속 질문,
  대화 없는 요약 요청을 안전한 기본 전략으로 바꿉니다.
- graph state: `user_input`, `user_profile`, `exclude_expired`, `messages`, `documents`,
  `route`, `route_reason`, `answer_strategy`, `retrieval_queries`, `answer`
- conversation state: Human/AI 메시지를 사용자별 `thread_id`에 누적

### Streamlit 데모

FastAPI 서버를 먼저 실행한 뒤 별도 터미널에서 실행합니다.

```bash
streamlit run demo_streamlit.py --server.port 8501
```

브라우저에서 `http://127.0.0.1:8501`에 접속합니다. 다른 API 주소를 사용할
경우 환경변수로 지정할 수 있습니다.

```bash
YOUTH_RAG_API_URL=http://127.0.0.1:8001 \
streamlit run demo_streamlit.py --server.port 8501
```

## API

| Method | Endpoint | 설명 |
| --- | --- | --- |
| `POST` | `/user/registration` | 사용자 프로필 등록 |
| `GET` | `/user/{user_id}` | 사용자 프로필 조회 |
| `POST` | `/user/{user_id}` | 사용자 프로필 수정 |
| `DELETE` | `/user/{user_id}` | 사용자 프로필 삭제 |
| `GET` | `/policies/{policy_id}` | 정책 상세 정보 조회 |
| `POST` | `/chat` | 사용자 프로필 기반 정책 검색 및 SSE 답변 |
| `DELETE` | `/chat/{user_id}` | 사용자 대화 기록 삭제 |

사용자 등록 예시:

```bash
curl -X POST http://127.0.0.1:8000/user/registration \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "sample-user",
    "age": 27,
    "gender": "여성",
    "job": "구직자",
    "income": 3000,
    "region": "서울특별시"
  }'
```

스트리밍 채팅 예시:

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "sample-user",
    "user_input": "서울에서 지원받을 수 있는 주거 정책을 알려줘",
    "exclude_expired": true
  }'
```

SSE 응답은 검색 context와 정책 ID를 담은 `metadata`, 답변 텍스트 조각을 담은
`chunk`, 완료를 알리는 `done` 이벤트로 전달됩니다.

```text
data: {"type":"metadata","data":{"contexts":[...],"retrieved_policy_ids":[...]}}

data: {"type":"chunk","data":"답변 일부"}

data: {"type":"done"}
```

## RAG 평가

평가 데이터는 `config.yaml`의 `evaluation.example_path`에서 관리합니다. 각 사례는
질문, 사용자 프로필, 정답 정책 ID, 만료 정책 제외 여부, metadata를 포함합니다.

평가 데이터를 생성합니다.

```bash
uv run python -m scripts.generate_eval_dataset --sample-size 100 --overwrite
```

기본 생성 크기는 500건이며, 재현 가능한 생성을 위해 seed는 기본값 `42`를
사용합니다. 여러 모델을 섞어 질문을 생성하려면
`--generation-model PROVIDER/MODEL=WEIGHT` 옵션을 사용할 수 있습니다.

Langfuse Dataset을 생성하거나 갱신하고, LangGraph RAG와 evaluator를 실행합니다.

```bash
uv run python -m scripts.evaluate_rag
```

같은 retrieval 진입점에서 dense, BM25, hybrid를 local 또는
Langfuse 실험으로 평가합니다.

```bash
uv run python -m scripts.evaluate_retrieval run \
  --tracking local \
  --provider upstage \
  --model solar-embedding-1-large-query \
  --chroma-dir data/chroma \
  --retrieval-mode dense
```

Planner query cache나 hybrid 가중치 sweep의 전체 옵션은 각각
`uv run python -m scripts.generate_planner_query_cache --help`,
`uv run python -m scripts.evaluate_retrieval sweep --help`로 확인할 수 있습니다.

평가 지표:

| 지표 | 계산 방식 |
| --- | --- |
| Context Recall | 정답 정책 ID 중 검색된 정책 ID의 비율 |
| Context Average Helpfulness | 검색된 각 context가 질문과 프로필에 얼마나 도움이 되는지 LLM judge로 평가 |
| Faithfulness | 답변의 사실 주장이 검색 context에 근거하는지 LLM judge로 평가 |
| Answer Relevance | 답변이 질문과 사용자 프로필 요구에 직접 답하는지 LLM judge로 평가 |

Context Recall은 정책 ID를 직접 비교하고, Context Average Helpfulness,
Faithfulness, Answer Relevance는 평가 모델을 호출합니다.

## 테스트

```bash
uv run pytest -q
```
