"""
Synthetic lending policy ruleset.

This module is an EDUCATIONAL / ILLUSTRATIVE pattern for how a bank might
encode underwriting policy checks and a fair-lending-style consistency
review as explicit, testable, versionable code rather than opaque LLM
judgment. It does not implement, reference, or assert compliance with any
specific real-world regulation (e.g. ECOA, Regulation B, FCRA) -- treat the
rule set purely as a demonstration of the *pattern*: deterministic,
auditable policy logic sitting alongside the LLM-driven agents.

Two ideas worth calling out because they're the crux of the "explainable /
governable agentic system" story this repo tells:

1. Permitted-factor allow-listing. `PolicyResult.permitted_factors_used`
   is the ONLY set of factors the downstream `explainability_agent` is
   allowed to cite in its rationale. This turns "did the model quietly rely
   on something it shouldn't have" from a trust exercise into a code-level
   contract that a reviewer (or a unit test) can check.

2. Excluded-factor verification. `LoanApplication` (domain/models.py)
   never collects age, race, gender, zip code, marital status, or national
   origin in the first place -- the safest way to guarantee a factor never
   drives a decision is to never capture it. `excluded_factors_checked`
   documents that verification explicitly rather than leaving it implicit.
"""

from __future__ import annotations

from underwriting.domain.models import (
    EmploymentStatus,
    ExtractedFinancials,
    LoanApplication,
    PolicyFlag,
    PolicyResult,
    PolicySeverity,
    RiskAssessment,
)

# Factors this policy engine is permitted to rely on when explaining a
# decision. Kept as a module-level constant (not re-derived per call) so it
# is trivially greppable/auditable by a reviewer.
PERMITTED_FACTORS: tuple[str, ...] = (
    "monthly_income",
    "debt_to_income_ratio",
    "credit_band",
    "requested_amount_to_income_ratio",
    "employment_status",
    "delinquency_history",
)

# Factors verified as structurally impossible to use, because the intake
# schema (LoanApplication) never collects them. Listed explicitly so the
# absence is a documented, auditable claim rather than an implicit one.
EXCLUDED_FACTORS: tuple[str, ...] = (
    "age",
    "race",
    "gender",
    "zip_code",
    "marital_status",
    "national_origin",
)

# Above this multiple of stated annual income, a requested amount is
# considered over-leveraged regardless of DTI/credit band.
MAX_REQUEST_TO_INCOME_MULTIPLE = 5.0

# Below this extraction confidence, we don't trust the parsed financials
# enough to decide automatically -- routed to a BLOCK flag which forces
# escalation regardless of the computed risk tier.
MIN_EXTRACTION_CONFIDENCE = 0.50

# If the applicant-stated monthly debt and the document-extracted monthly
# debt disagree by more than this fraction, flag a consistency concern.
MAX_STATED_VS_EXTRACTED_DEBT_DELTA = 0.25


def evaluate_policy(
    application: LoanApplication,
    extracted: ExtractedFinancials,
    risk: RiskAssessment,
) -> PolicyResult:
    """Run the synthetic policy ruleset and return a PolicyResult.

    Pure function of its three inputs -- no I/O, no LLM call -- which is
    exactly why this logic lives in `domain/` rather than as a graph node
    that calls an LLM: deterministic policy should stay deterministic and
    unit-testable without mocking anything.
    """
    flags: list[PolicyFlag] = []

    # --- Over-leverage check ---
    if application.stated_annual_income > 0:
        request_to_income = application.requested_amount / application.stated_annual_income
    else:
        request_to_income = float("inf")

    if request_to_income > MAX_REQUEST_TO_INCOME_MULTIPLE:
        flags.append(
            PolicyFlag(
                code="OVER_LEVERAGED_REQUEST",
                description=(
                    f"Requested amount is {request_to_income:.1f}x stated annual income "
                    f"(policy limit {MAX_REQUEST_TO_INCOME_MULTIPLE:.1f}x)."
                ),
                severity=PolicySeverity.BLOCK,
            )
        )

    # --- Extraction confidence gate ---
    if extracted.extraction_confidence < MIN_EXTRACTION_CONFIDENCE:
        flags.append(
            PolicyFlag(
                code="LOW_EXTRACTION_CONFIDENCE",
                description=(
                    f"Document extraction confidence {extracted.extraction_confidence:.2f} is "
                    f"below the minimum {MIN_EXTRACTION_CONFIDENCE:.2f} required for a "
                    "straight-through decision."
                ),
                severity=PolicySeverity.BLOCK,
            )
        )

    # --- Stated vs. extracted debt consistency check ---
    # A basic fair-lending-adjacent "consistency" pattern: the rationale
    # must not silently prefer one source of debt data without flagging a
    # material disagreement between what the applicant declared and what
    # was extracted from their own submitted documents.
    stated_debt = application.stated_monthly_debt
    extracted_debt = extracted.existing_monthly_debt_payments
    denom = max(stated_debt, 1.0)
    delta_fraction = abs(stated_debt - extracted_debt) / denom
    if delta_fraction > MAX_STATED_VS_EXTRACTED_DEBT_DELTA:
        flags.append(
            PolicyFlag(
                code="DEBT_STATEMENT_INCONSISTENCY",
                description=(
                    f"Applicant-stated monthly debt (${stated_debt:,.2f}) differs from "
                    f"document-extracted monthly debt (${extracted_debt:,.2f}) by "
                    f"{delta_fraction:.0%}, exceeding the {MAX_STATED_VS_EXTRACTED_DEBT_DELTA:.0%} "
                    "consistency threshold."
                ),
                severity=PolicySeverity.WARN,
            )
        )

    # --- Employment/purpose sanity check (informational only) ---
    if (
        application.employment_status == EmploymentStatus.UNEMPLOYED
        and application.requested_amount > 0
    ):
        flags.append(
            PolicyFlag(
                code="UNEMPLOYED_APPLICANT",
                description=(
                    "Applicant reports unemployed status; income sufficiency is driven "
                    "entirely by non-wage income captured during document extraction."
                ),
                severity=PolicySeverity.WARN,
            )
        )

    if application.employment_status == EmploymentStatus.SELF_EMPLOYED:
        flags.append(
            PolicyFlag(
                code="SELF_EMPLOYED_INCOME_VARIABILITY",
                description=(
                    "Self-employed applicant: income may be more variable period-to-period. "
                    "Noted for context; not a basis for denial on its own."
                ),
                severity=PolicySeverity.INFO,
            )
        )

    passed = not any(f.severity == PolicySeverity.BLOCK for f in flags)

    return PolicyResult(
        passed=passed,
        flags=flags,
        permitted_factors_used=list(PERMITTED_FACTORS),
        excluded_factors_checked=list(EXCLUDED_FACTORS),
    )
