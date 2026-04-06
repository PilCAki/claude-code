"""Exponential backoff retry policy for transient failures."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Configuration for exponential backoff retries."""
    base_delay_ms: int = 1_000
    max_delay_ms: int = 30_000
    max_attempts: int = 3
    jitter: bool = True

    def delay_ms(self, attempt: int) -> int:
        """Calculate delay for the given attempt number (0-indexed).

        Uses exponential backoff: base * 2^attempt, capped at max_delay.
        With jitter enabled, adds random 0-50% of the computed delay.
        """
        delay = min(self.base_delay_ms * (2 ** attempt), self.max_delay_ms)
        if self.jitter:
            delay += random.randint(0, delay // 2)
        return min(delay, self.max_delay_ms)

    def should_retry(self, attempt: int) -> bool:
        """Whether another retry is allowed at this attempt number."""
        return attempt < self.max_attempts

    def delay_seconds(self, attempt: int) -> float:
        """Delay in seconds (convenience for asyncio.sleep)."""
        return self.delay_ms(attempt) / 1000.0


@dataclass
class RetryState:
    """Mutable retry state for a single operation."""
    policy: RetryPolicy
    attempt: int = 0
    total_delay_ms: int = 0

    def next_attempt(self) -> tuple[bool, int]:
        """Advance to next attempt.

        Returns (should_retry, delay_ms). If should_retry is False,
        the operation should be aborted.
        """
        if not self.policy.should_retry(self.attempt):
            return False, 0
        delay = self.policy.delay_ms(self.attempt)
        self.attempt += 1
        self.total_delay_ms += delay
        return True, delay

    @property
    def exhausted(self) -> bool:
        return self.attempt >= self.policy.max_attempts


def build_retry_response(
    state: RetryState,
    error_context: str = "",
) -> dict[str, object]:
    """Build a hook error-handling response with retry + backoff info.

    Returns a dict suitable for returning from on_error_occurred.
    """
    should_retry, delay_ms = state.next_attempt()
    if not should_retry:
        return {
            "errorHandling": "abort",
            "reason": f"Max retries ({state.policy.max_attempts}) exhausted after {state.total_delay_ms}ms total delay.",
        }
    return {
        "errorHandling": "retry",
        "retryCount": 1,
        "retryDelayMs": delay_ms,
        "attempt": state.attempt,
        "maxAttempts": state.policy.max_attempts,
    }
