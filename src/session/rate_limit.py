import time
from dataclasses import dataclass

from fastapi import Request


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """단일 프로세스 in-memory 토큰 버킷. 여러 워커/컨테이너로 확장하면
    워커마다 버킷이 따로 생겨 실제 상한이 (워커 수 x capacity)가 된다."""

    def __init__(
        self,
        capacity: int,
        refill_period_seconds: float,
        clock=time.monotonic,
    ) -> None:
        self.capacity = capacity
        self.refill_period_seconds = refill_period_seconds
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def try_consume(self, key: str) -> tuple[bool, float]:
        """토큰이 있으면 하나 소비하고 (True, 0.0)을, 없으면
        (False, 다음 토큰까지 남은 초)를 반환한다."""
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=float(self.capacity), last_refill=now)
            self._buckets[key] = bucket
        else:
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                self.capacity,
                bucket.tokens + elapsed / self.refill_period_seconds,
            )
            bucket.last_refill = now

        if bucket.tokens < 1:
            retry_after = (1 - bucket.tokens) * self.refill_period_seconds
            return False, retry_after

        bucket.tokens -= 1
        return True, 0.0


def client_ip(request: Request) -> str:
    # Caddy가 X-Forwarded-For에 실제 클라이언트 IP를 append한다.
    # 이 헤더는 클라이언트가 직접 보낼 수도 있는 값이라 맨 앞이 아니라
    # Caddy가 붙인 마지막 값만 신뢰한다.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"
