"""
Mock "credit bureau" external tool.

Architectural role
-------------------
`risk_scoring_agent` (service/nodes.py) needs a credit report the way a
real underwriting pipeline would call out to an external bureau API. This
module simulates that external dependency: same call shape (a function that
can fail, that we wrap in retry/backoff/circuit-breaker), same output shape
(a `CreditBureauReport`), but entirely offline and deterministic so the
demo and test suite never depend on network access or a paid data vendor.

Determinism
-----------
The "bureau" derives a score deterministically from a SHA-256 digest of the
applicant's full name (stable across processes and Python versions, unlike
the builtin `hash()`, which is salted per-run). This keeps repeated calls
for the same synthetic applicant reproducible -- useful for tests and for a
reviewer re-running the demo and getting consistent output -- while still
looking like "real" varied bureau data across different applicants.
"""

from __future__ import annotations

import hashlib
import logging

from underwriting.domain.models import CreditBand, CreditBureauReport
from underwriting.infrastructure.resilience import CircuitBreaker, with_retry

logger = logging.getLogger(__name__)

# Dedicated breaker instance for the bureau dependency -- kept separate from
# the LLM breaker so a flaky bureau doesn't trip circuit protection for
# unrelated LLM calls, and vice versa.
_bureau_breaker = CircuitBreaker(name="credit-bureau", failure_threshold=3,
                                  recovery_timeout_seconds=10.0)


def _score_from_name(applicant_full_name: str) -> int:
    """Deterministically derive a 300-850 style credit score from a name.

    Uses SHA-256 (not the builtin `hash()`, which is randomly salted per
    process) so the same synthetic applicant always yields the same score,
    across runs, machines, and CI workers.
    """
    digest = hashlib.sha256(applicant_full_name.strip().lower().encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return 300 + (seed % 551)  # 300..850 inclusive


def _band_from_score(score: int) -> CreditBand:
    """Map a numeric score to an illustrative band. Thresholds are synthetic."""
    if score >= 740:
        return CreditBand.EXCELLENT
    if score >= 670:
        return CreditBand.GOOD
    if score >= 580:
        return CreditBand.FAIR
    return CreditBand.POOR


def _fetch_report_uncached(applicant_full_name: str) -> CreditBureauReport:
    """Simulate the actual bureau API round trip.

    In a real system this would be an HTTP call with its own timeout. Here
    it's pure computation, but it is still routed through the retry +
    circuit-breaker wrapper below so the calling code exercises the same
    resilience path it would in production (and so tests of that path have
    something real to point at, via dependency injection of a failing
    stand-in -- see tests/test_resilience.py).
    """
    score = _score_from_name(applicant_full_name)
    band = _band_from_score(score)
    # Deterministic secondary attributes, derived from the same score so
    # they move sensibly together (higher score -> fewer delinquencies).
    delinquencies = max(0, (850 - score) // 150)
    open_tradelines = 3 + (score % 7)
    return CreditBureauReport(
        credit_score=score,
        credit_band=band,
        open_tradelines=open_tradelines,
        delinquencies_last_24mo=delinquencies,
    )


@with_retry(max_attempts=3, initial_wait_seconds=0.1, max_wait_seconds=1.0)
def get_credit_report(applicant_full_name: str) -> CreditBureauReport:
    """Public entry point used by risk_scoring_agent.

    Wrapped with retry-with-backoff-and-jitter (`with_retry`) and a circuit
    breaker (`_bureau_breaker`), matching the resilience pattern applied to
    every external call in this system (see infrastructure/resilience.py).
    """
    logger.info("credit_bureau_request", extra={"applicant": applicant_full_name})
    report = _bureau_breaker.call(lambda: _fetch_report_uncached(applicant_full_name))
    logger.info(
        "credit_bureau_response",
        extra={"applicant": applicant_full_name, "credit_band": report.credit_band.value},
    )
    return report
