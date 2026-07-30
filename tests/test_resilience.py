"""Unit tests for the resilience primitives (retry + circuit breaker) that
back every external call in this system (LLM calls, credit bureau calls)."""

from __future__ import annotations

import pytest

from underwriting.infrastructure.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    with_retry,
)


def test_with_retry_retries_until_success() -> None:
    attempts = {"count": 0}

    @with_retry(max_attempts=3, initial_wait_seconds=0.001, max_wait_seconds=0.01)
    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("transient failure")
        return "ok"

    assert flaky() == "ok"
    assert attempts["count"] == 3


def test_with_retry_gives_up_after_max_attempts() -> None:
    attempts = {"count": 0}

    @with_retry(max_attempts=2, initial_wait_seconds=0.001, max_wait_seconds=0.01)
    def always_fails() -> None:
        attempts["count"] += 1
        raise ValueError("permanent failure")

    with pytest.raises(ValueError):
        always_fails()
    assert attempts["count"] == 2


def test_circuit_breaker_opens_after_threshold_failures() -> None:
    breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout_seconds=60.0)

    def failing() -> None:
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(failing)

    assert breaker.state == CircuitState.OPEN
    # A third call should fail fast without even attempting `failing`.
    with pytest.raises(CircuitBreakerOpenError):
        breaker.call(failing)


def test_circuit_breaker_closes_after_successful_call() -> None:
    breaker = CircuitBreaker(name="test2", failure_threshold=1, recovery_timeout_seconds=0.0)

    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert breaker.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)

    # recovery_timeout_seconds=0.0 means the breaker is immediately eligible
    # to move to HALF_OPEN and accept a trial call.
    result = breaker.call(lambda: "recovered")
    assert result == "recovered"
    assert breaker.state == CircuitState.CLOSED
