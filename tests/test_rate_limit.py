from src.session.rate_limit import TokenBucketLimiter


def test_allows_up_to_capacity_then_blocks():
    clock = [0.0]
    limiter = TokenBucketLimiter(capacity=3, refill_period_seconds=10, clock=lambda: clock[0])

    for _ in range(3):
        allowed, _ = limiter.try_consume("session-a")
        assert allowed

    allowed, retry_after = limiter.try_consume("session-a")
    assert not allowed
    assert retry_after == 10


def test_refills_over_time():
    clock = [0.0]
    limiter = TokenBucketLimiter(capacity=1, refill_period_seconds=10, clock=lambda: clock[0])

    assert limiter.try_consume("session-a")[0]
    assert not limiter.try_consume("session-a")[0]

    clock[0] = 10.0
    assert limiter.try_consume("session-a")[0]


def test_keys_are_independent():
    clock = [0.0]
    limiter = TokenBucketLimiter(capacity=1, refill_period_seconds=10, clock=lambda: clock[0])

    assert limiter.try_consume("session-a")[0]
    assert limiter.try_consume("session-b")[0]
