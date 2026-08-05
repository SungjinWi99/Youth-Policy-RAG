FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Fix the container timezone to KST: policy deadline filters are computed
# against date.today(), and UTC would shift the baseline day during 00:00-09:00 KST.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    TZ=Asia/Seoul

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py config.yaml ./
COPY src ./src
# 정책 동기화·적재를 배포 서버에서 실행하기 위해 포함한다. 이 스크립트들은
# src와 런타임 의존성만 쓰므로 이미지가 커지지 않는다.
COPY scripts ./scripts

# The application expects these paths to exist. Their contents are supplied
# at runtime through the Compose data bind mount.
RUN mkdir -p /app/data/chroma /app/data/sqlite

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
