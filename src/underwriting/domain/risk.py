"""
Risk-scoring math: debt-to-income ratio and credit-band-driven risk tiering.

Kept as pure functions of primitive/domain-model inputs (no I/O, no graph
awareness) so `risk_scoring_agent` (service/nodes.py) stays a thin
adapter: fetch the credit report, call these functions, wrap the result in
a `RiskAssessment`. That thinness is what makes the risk math itself
unit-testable without mocking a chat model or a graph.
"""

from __future__ import annotations

from underwriting.config import Settings
from underwriting.domain.models import CreditBand, RiskTier


def compute_debt_to_income_ratio(
    *, monthly_income: float, monthly_debt_payments: float
) -> float:
    """Compute DTI, guarding against a zero/negative income denominator.

    An applicant with no measurable monthly income is treated as maximally
    risky (DTI = 1.0, i.e. "100% of income" ) rather than raising a
    ZeroDivisionError -- the graph should always be able to produce a
    (likely adverse) decision rather than crash on a degenerate input.
    """
    if monthly_income <= 0:
        return 1.0
    return round(monthly_debt_payments / monthly_income, 4)


def classify_risk_tier(*, dti: float, credit_band: CreditBand, settings: Settings) -> RiskTier:
    """Classify a coarse risk tier from DTI and credit band.

    Thresholds come from `Settings` (env-configurable, see .env.example)
    rather than being hardcoded, so a reviewer can tune the demo's risk
    appetite without touching code. This is a simple, transparent rule
    table by design: an underwriting risk tier that a regulator or auditor
    might ask to see explained should be easy to explain.

    Rule precedence (checked in order):
      1. HIGH   -- DTI at/above the hard-decline threshold, OR credit band
                   is POOR. Either alone is disqualifying.
      2. BORDERLINE -- DTI in the borderline band, OR credit band is FAIR.
                   Routed to human_escalation rather than decided
                   automatically.
      3. LOW    -- DTI comfortably below the borderline band AND credit
                   band is GOOD or EXCELLENT.
      4. MEDIUM -- everything else (the default, moderate-risk bucket).
    """
    if dti >= settings.dti_hard_decline or credit_band == CreditBand.POOR:
        return RiskTier.HIGH

    if (settings.dti_borderline_low <= dti < settings.dti_borderline_high) or credit_band == CreditBand.FAIR:
        return RiskTier.BORDERLINE

    if dti < settings.dti_borderline_low and credit_band in (CreditBand.EXCELLENT, CreditBand.GOOD):
        return RiskTier.LOW

    return RiskTier.MEDIUM


def build_rationale_factors(*, dti: float, credit_band: CreditBand, credit_score: int) -> list[str]:
    """Human-readable factor strings carried forward for explainability."""
    return [
        f"debt_to_income_ratio={dti:.1%}",
        f"credit_band={credit_band.value}",
        f"credit_score={credit_score}",
    ]
