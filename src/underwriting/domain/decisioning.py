"""
Final outcome determination: approve/deny + adverse-action reasons.

This is the last piece of deterministic business logic before the
`explainability_agent` node turns the outcome into prose via the pluggable
chat model. Keeping outcome determination deterministic (rather than
letting the LLM decide approve/deny) is a deliberate governance choice --
see the README's "Key Design Decisions" section: the LLM's job in this
system is to *explain* a decision that deterministic, auditable code already
made, never to *make* the decision itself.
"""

from __future__ import annotations

from underwriting.domain.models import (
    CreditBand,
    DecisionOutcome,
    EscalationResolution,
    LoanApplication,
    PolicyResult,
    PolicySeverity,
    RiskAssessment,
    RiskTier,
)


def _adverse_reasons(risk: RiskAssessment, policy: PolicyResult) -> list[str]:
    """Build factor-level adverse-action reasons from risk + policy findings.

    Mirrors the real-world practice of naming specific reasons for an
    adverse credit decision. Every reason string here traces back to a
    `PERMITTED_FACTORS` entry (domain/policy.py) or a concrete policy flag
    -- never to an excluded factor.
    """
    reasons: list[str] = []

    if risk.risk_tier == RiskTier.HIGH:
        reasons.append(
            f"Debt-to-income ratio of {risk.debt_to_income_ratio:.1%} exceeds the "
            "acceptable underwriting threshold."
        )
        if risk.credit_report.credit_band == CreditBand.POOR:
            reasons.append(
                f"Credit bureau band classified as 'poor' (score "
                f"{risk.credit_report.credit_score})."
            )

    for flag in policy.flags:
        if flag.severity in (PolicySeverity.BLOCK, PolicySeverity.WARN):
            reasons.append(flag.description)

    return reasons


def determine_outcome(
    application: LoanApplication,
    risk: RiskAssessment,
    policy: PolicyResult,
    escalation_resolution: EscalationResolution | None = None,
) -> tuple[DecisionOutcome, float | None, list[str]]:
    """Return (outcome, approved_amount, adverse_action_reasons).

    Two paths:

    * Escalated: a human reviewer's `approve` flag is authoritative. This
      is intentional -- once a case is escalated, the deterministic
      risk/policy tiers no longer get the final word; a qualified human
      does, and their decision is durably recorded via
      `EscalationResolution`.
    * Straight-through (not escalated): approve unless the risk tier is
      HIGH or the policy check failed (i.e. a BLOCK-severity flag fired).
    """
    if escalation_resolution is not None:
        if escalation_resolution.approve:
            return DecisionOutcome.APPROVED, application.requested_amount, []
        reasons = _adverse_reasons(risk, policy) or [
            "Human reviewer determined the application did not meet lending guidelines "
            "after escalation review."
        ]
        return DecisionOutcome.DENIED, None, reasons

    if risk.risk_tier == RiskTier.HIGH or not policy.passed:
        return DecisionOutcome.DENIED, None, _adverse_reasons(risk, policy)

    return DecisionOutcome.APPROVED, application.requested_amount, []
