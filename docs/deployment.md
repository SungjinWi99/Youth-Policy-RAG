# 배포 가이드

Docker Compose로 로컬에서 전체 스택을 띄우는 방법과, 같은 스택을 GCP Compute
Engine VM 한 대에 올리는 방법입니다. 애플리케이션 자체의 설정·실행은
[README](../README.md)를 참고하세요.

이 서비스는 SQLite 두 개(세션·대화 checkpoint)에 매 요청 쓰고 BM25 역색인을
프로세스 메모리에 들고 있어 **단일 인스턴스 전제**입니다. 수평 확장하려면
저장소부터 바꿔야 합니다.

## 컨테이너 구성

| 서비스 | 역할 | 노출 |
| --- | --- | --- |
| `api` | FastAPI + LangGraph RAG | 내부 `8000`만 (`expose`) |
| `web` | Next.js 상담 UI, `/api/*` 프록시 | 로컬은 `3000`, GCP는 Caddy 뒤 |
| `langfeather` | trace·피드백 collector | `127.0.0.1:4319` |
| `proxy` (GCP 전용) | Caddy 리버스 프록시 | `80` |

브라우저에 열리는 포트는 `web`(또는 GCP의 `proxy`)뿐입니다. FastAPI의 `8000`은
Compose 내부 네트워크에서만 접근되며, `web`은 `BACKEND_URL=http://api:8000`으로
연결합니다.

`langfeather`는 `network_mode: service:api`로 `api`와 network namespace를
공유합니다. collector가 Host 헤더가 `localhost`/`127.0.0.1`이 아닌 요청을
거부하기 때문에 Compose 서비스 이름(`http://langfeather:4319`)으로는 연결할 수
없고, `api`가 `LANGFEATHER_ENDPOINT=http://127.0.0.1:4319`로 loopback을 씁니다.
인증이 없고 trace에 사용자 질문 원문이 들어가므로 호스트 포트는 loopback에만
바인딩합니다.

`data/`는 이미지에 넣지 않고 호스트에서 마운트합니다. 컨테이너를 교체해도 Chroma
인덱스와 상담 기록이 남습니다.

## 로컬 Docker Compose

먼저 정책 원본과 Chroma 인덱스가 들어 있는 `data/` 디렉터리를 준비합니다
(README의 "데이터 준비" 참고).

```bash
cp .env.example .env
# .env에 config.yaml이 사용하는 provider의 API 키를 입력합니다.

docker compose up --build -d
docker compose logs -f api
```

상담 화면은 `http://localhost:3000`, LangFeather 대시보드는
`http://127.0.0.1:4319`입니다. `3000`을 다른 서비스가 쓰고 있으면 `.env`에
`WEB_PORT=3001`처럼 지정합니다.

- HTTPS 리버스 프록시 뒤에서 운영할 때는 `.env`의 `SESSION_COOKIE_SECURE=true`.
  로컬 HTTP 테스트에서는 기본값 `false`를 유지합니다.
- tracing을 끄려면 `LANGFEATHER_TRACING=false docker compose up -d --build`.

## GCP Compute Engine 배포

`main` 푸시가 GitHub Actions에서 테스트 → 이미지 빌드 → Artifact Registry 푸시 →
VM 재기동까지 처리합니다(`.github/workflows/deploy.yml`). PR에서는 테스트 job만
돕니다.

### 1. 인프라 생성 (최초 1회)

```bash
gcloud auth login
PROJECT_ID=<프로젝트ID> bash deploy/gcp-bootstrap.sh
```

스크립트가 API 활성화, Artifact Registry, 고정 IP, VM(e2-medium·Ubuntu 24.04·
Docker), 방화벽, 배포용 서비스 계정, Workload Identity Federation을 만들고
마지막에 GitHub Secrets에 넣을 값을 출력합니다.

방화벽은 **80만** 외부에 열고 SSH(22)는 IAP 터널 대역만 허용합니다. 배포도 접속도
`--tunnel-through-iap`로 들어갑니다.

### 2. `.env` 업로드

로컬 `.env`에 `IMAGE_REPOSITORY`를 더한 것입니다. `/opt`는 root 소유이므로 홈으로
올린 뒤 옮깁니다.

```bash
VM=youth-policy-rag ZONE=asia-northeast3-a
TUNNEL="--zone $ZONE --tunnel-through-iap"

gcloud compute scp .env $VM:~/app.env $TUNNEL
gcloud compute ssh $VM $TUNNEL --command '
  sudo install -m 600 ~/app.env /opt/youth-policy-rag/.env && rm ~/app.env
'
```

```bash
# .env에 추가
IMAGE_REPOSITORY=asia-northeast3-docker.pkg.dev/<프로젝트ID>/youth-policy-rag
SESSION_COOKIE_SECURE=true   # HTTPS를 붙였다면
```

### 3. 데이터 준비 — 배포보다 **먼저**

`data/`는 로컬에서 올리지 않습니다. `scripts/`가 API 이미지에 들어 있고 `data/`가
볼륨으로 마운트되므로, 수집과 적재를 서버에서 실행하면 됩니다.

```bash
gcloud compute ssh $VM $TUNNEL --command '
  cd /opt/youth-policy-rag
  sudo docker compose -f compose.gcp.yaml run --rm --no-deps api \
    uv run --no-sync python scripts/sync_policies.py --snapshot-only
  sudo docker compose -f compose.gcp.yaml run --rm --no-deps api \
    uv run --no-sync python scripts/ingest_chroma.py \
      --provider upstage --model solar-embedding-1-large-passage \
      --chroma-dir data/chroma --recreate
'
```

**순서를 지켜야 합니다.** 문서가 0건이면 `BM25DocumentIndex`가 기동 시점에 예외를
던져 컨테이너가 재시작 루프에 빠집니다(빈 인덱스로 조용히 서빙하는 것보다
낫습니다). `docker compose run`은 서버 기동 경로를 타지 않으므로 데이터가 없는
상태에서도 위 두 명령은 정상 동작합니다.

`data/sqlite`는 비워 둡니다. 앱이 기동할 때 스키마를 만듭니다.

### 4. 배포

`main`에 푸시하거나 Actions 탭에서 `Build and Deploy`를 수동 실행합니다. 배포
job은 VM에 `compose.gcp.yaml`과 `Caddyfile.http`를 올린 뒤 다음을 실행합니다.

```bash
sudo docker compose -f compose.gcp.yaml pull
sudo docker compose -f compose.gcp.yaml up -d --wait --wait-timeout 180
sudo docker image prune -f
```

`--wait`은 `api`의 healthcheck가 healthy가 될 때까지 기다리고, 안 되면 배포를
실패시킵니다. 실패하면 컨테이너가 재시작되며 증거가 사라지기 전에 `ps`와 최근
로그 100줄을 Actions 기록에 남깁니다.

이미지는 러너가 VM으로 보내지 않습니다. VM이 자기 서비스 계정으로 Artifact
Registry에서 직접 pull합니다.

## 운영

### 정책 갱신

매일 04:00 KST(`cron: 0 19 * * *`)에 `Sync Policies` 워크플로가 VM에서
`sync_policies.py`를 실행합니다. 손으로 돌릴 때는 되돌릴 수 없는 작업이므로
`--dry-run`으로 변경 규모를 먼저 확인하는 것을 권합니다.

```bash
gcloud compute ssh $VM $TUNNEL --command '
  cd /opt/youth-policy-rag
  sudo docker compose -f compose.gcp.yaml run --rm --no-deps api \
    uv run --no-sync python scripts/sync_policies.py --dry-run
'
```

원본 스냅샷 교체가 끝나면 실행 중인 API가 `data/raw`의 mtime 변화를 감지해 BM25
역색인을 백그라운드에서 재구성합니다. 재시작할 필요가 없습니다.

### LangFeather UI 보기

외부에 열려 있지 않습니다. 터널을 열고 `http://127.0.0.1:4319`로 접속합니다.

```bash
gcloud compute ssh $VM --zone $ZONE --tunnel-through-iap -- -L 4319:127.0.0.1:4319
```

### 상태 확인

```bash
gcloud compute ssh $VM $TUNNEL --command '
  cd /opt/youth-policy-rag
  sudo docker compose -f compose.gcp.yaml ps
  sudo docker compose -f compose.gcp.yaml logs --tail 100 api
'
```

`/health`는 Chroma 컬렉션 접근과 문서 수만 확인합니다. 문서가 0건이거나 컬렉션에
접근할 수 없으면 503이며, 이는 대개 `data` 볼륨 미마운트 또는 빈 인덱스입니다.
