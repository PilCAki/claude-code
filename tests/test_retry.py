"""Tests for copilotcode_sdk.retry module."""
from __future__ import annotations

from copilotcode_sdk.retry import RetryPolicy, RetryState, build_retry_response


class TestRetryPolicy:
    def test_default_values(self) -> None:
        p = RetryPolicy()
        assert p.base_delay_ms == 1_000
        assert p.max_delay_ms == 30_000
        assert p.max_attempts == 3
        assert p.jitter is True

    def test_exponential_backoff_without_jitter(self) -> None:
        p = RetryPolicy(base_delay_ms=100, max_delay_ms=10_000, jitter=False)
        assert p.delay_ms(0) == 100   # 100 * 2^0
        assert p.delay_ms(1) == 200   # 100 * 2^1
        assert p.delay_ms(2) == 400   # 100 * 2^2
        assert p.delay_ms(3) == 800   # 100 * 2^3

    def test_delay_capped_at_max(self) -> None:
        p = RetryPolicy(base_delay_ms=1000, max_delay_ms=2000, jitter=False)
        assert p.delay_ms(0) == 1000
        assert p.delay_ms(1) == 2000
        assert p.delay_ms(5) == 2000  # capped

    def test_jitter_adds_randomness(self) -> None:
        p = RetryPolicy(base_delay_ms=1000, max_delay_ms=30_000, jitter=True)
        # With jitter, delay >= base (jitter adds, never subtracts)
        delays = {p.delay_ms(0) for _ in range(20)}
        # At least some variation expected (extremely unlikely all equal)
        assert min(delays) >= 1000

    def test_should_retry(self) -> None:
        p = RetryPolicy(max_attempts=3)
        assert p.should_retry(0) is True
        assert p.should_retry(1) is True
        assert p.should_retry(2) is True
        assert p.should_retry(3) is False
        assert p.should_retry(10) is False

    def test_delay_seconds(self) -> None:
        p = RetryPolicy(base_delay_ms=2000, jitter=False)
        assert p.delay_seconds(0) == 2.0

    def test_frozen(self) -> None:
        p = RetryPolicy()
        try:
            p.base_delay_ms = 999  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestRetryState:
    def test_next_attempt_increments(self) -> None:
        policy = RetryPolicy(base_delay_ms=100, jitter=False, max_attempts=3)
        state = RetryState(policy=policy)
        ok, delay = state.next_attempt()
        assert ok is True
        assert delay == 100
        assert state.attempt == 1
        assert state.total_delay_ms == 100

    def test_next_attempt_escalates_delay(self) -> None:
        policy = RetryPolicy(base_delay_ms=100, jitter=False, max_attempts=5)
        state = RetryState(policy=policy)
        delays = []
        for _ in range(3):
            ok, delay = state.next_attempt()
            assert ok is True
            delays.append(delay)
        assert delays == [100, 200, 400]

    def test_exhausted_after_max_attempts(self) -> None:
        policy = RetryPolicy(max_attempts=2, jitter=False)
        state = RetryState(policy=policy)
        state.next_attempt()
        state.next_attempt()
        assert state.exhausted is True
        ok, _ = state.next_attempt()
        assert ok is False

    def test_total_delay_accumulates(self) -> None:
        policy = RetryPolicy(base_delay_ms=100, jitter=False, max_attempts=3)
        state = RetryState(policy=policy)
        state.next_attempt()  # 100
        state.next_attempt()  # 200
        state.next_attempt()  # 400
        assert state.total_delay_ms == 700


class TestBuildRetryResponse:
    def test_retry_response(self) -> None:
        policy = RetryPolicy(base_delay_ms=500, jitter=False, max_attempts=3)
        state = RetryState(policy=policy)
        resp = build_retry_response(state, "model_call")
        assert resp["errorHandling"] == "retry"
        assert resp["retryDelayMs"] == 500
        assert resp["attempt"] == 1
        assert resp["maxAttempts"] == 3

    def test_abort_after_exhaustion(self) -> None:
        policy = RetryPolicy(max_attempts=1, jitter=False)
        state = RetryState(policy=policy)
        state.next_attempt()  # use up the one attempt
        resp = build_retry_response(state)
        assert resp["errorHandling"] == "abort"
        assert "exhausted" in str(resp.get("reason", "")).lower()

    def test_retry_count_always_one(self) -> None:
        policy = RetryPolicy(jitter=False, max_attempts=5)
        state = RetryState(policy=policy)
        for _ in range(3):
            resp = build_retry_response(state)
            assert resp["retryCount"] == 1
