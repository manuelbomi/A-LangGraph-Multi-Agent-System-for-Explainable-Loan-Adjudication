"""
Reusable resilience primitives: retry-with-backoff and a circuit breaker.

Architectural role
-------------------
Every call this system makes to something outside its own process boundary
(an LLM provider, the mock credit bureau "external" tool) is wrapped with
these primitives. Centralizing them here means the retry/backoff/jitter
policy and the circuit-breaker behavior are defined once and reused
identically by `infrastructure/llm.py` and `infrastructure/credit_bureau.py`,
rather than each call site inventing its own ad hoc error handling.

Design notes
------------
* Retries use `tenacity` with exponential backoff *and* jitter. Jitter
  matters at production scale: without it, many clients that fail at the
  same instant (e.g. a provider blip) retry in lockstep and re-create the
  very load spike that caused the failure ("thundering herd").
* The circuit breaker is intentionally a small, dependency-free state
  machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED) rather than pulling in a
  heavier library -- it's easy to audit and enough for a single-process
  demo/service. In a real deployment with multiple replicas you'd want a
  shared-state breaker (e.g. backed by Redis) so state is consistent across
  pods; that tradeoff is called out in the README.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry(
    *,
    max_attempts: int = 3,
    initial_wait_seconds: float = 0.25,
    max_wait_seconds: float = 4.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a tenacity retry decorator with exponential backoff + jitter.

    Kept as a small factory (rather than a single pre-built decorator) so
    different call sites can tune attempt counts without duplicating the
    backoff math.
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=initial_wait_seconds, max=max_wait_seconds),
        retry=retry_if_exception_type(retry_on),
    )


class CircuitState(str, Enum):
    """The three states of a classic circuit breaker."""

    CLOSED = "closed"  # normal operation, calls pass through
    OPEN = "open"  # tripped: calls fail fast without attempting the operation
    HALF_OPEN = "half_open"  # cooldown elapsed: allow one trial call through


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a call is rejected because the breaker is OPEN."""


@dataclass
class CircuitBreaker:
    """A minimal, thread-safe circuit breaker.

    Trips to OPEN after `failure_threshold` consecutive failures, stays
    OPEN for `recovery_timeout_seconds`, then allows a single HALF_OPEN
    probe call. A successful probe closes the breaker; a failed probe
    re-opens it and restarts the cooldown.
    """

    name: str
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 15.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _maybe_transition_to_half_open(self) -> None:
        if self._state is CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout_seconds:
                logger.info("circuit_breaker_half_open", extra={"breaker": self.name})
                self._state = CircuitState.HALF_OPEN

    def call(self, fn: Callable[[], T]) -> T:
        """Execute `fn` under breaker protection, raising CircuitBreakerOpenError if OPEN."""
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state is CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open; failing fast."
                )

        try:
            result = fn()
        except Exception:
            with self._lock:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "circuit_breaker_opened",
                        extra={
                            "breaker": self.name,
                            "consecutive_failures": self._consecutive_failures,
                        },
                    )
            raise
        else:
            with self._lock:
                self._consecutive_failures = 0
                if self._state is not CircuitState.CLOSED:
                    logger.info("circuit_breaker_closed", extra={"breaker": self.name})
                self._state = CircuitState.CLOSED
            return result

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state
