# 청년정책 상담 챗봇 프론트엔드

기존 FastAPI를 백엔드로 사용하는 Next.js App Router 애플리케이션입니다.
브라우저 요청은 Next.js의 `/api/*` 프록시를 거치므로 FastAPI를 LAN이나
인터넷에 직접 노출할 필요가 없습니다.

## 로컬 실행

프로젝트 루트에서 FastAPI를 실행합니다.

```bash
uv run --locked uvicorn main:app --host 127.0.0.1 --port 8000
```

다른 터미널에서 프론트엔드를 실행합니다.

```bash
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

같은 와이파이의 다른 기기에서 접속을 허용하려면 다음 명령을 사용합니다.

```bash
npm run dev:lan
```

프론트 서버의 `BACKEND_URL` 기본값은 `http://127.0.0.1:8000`입니다.
EC2에서도 FastAPI와 Next.js를 같은 인스턴스에 두면 같은 값을 사용할 수
있습니다.

## 검증

```bash
npm run lint
npm run build
npm audit
```

프로덕션 빌드는 standalone 서버와 정적 자산을 한 실행 디렉터리에
준비합니다.

```bash
HOSTNAME=0.0.0.0 PORT=3000 npm start
```

Ubuntu와 EC2 배포 구조는 `docs/frontend_deployment.md`를 참고합니다.
