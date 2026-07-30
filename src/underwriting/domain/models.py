"""
Core domain models for the underwriting pipeline.

Every value that flows between graph nodes, crosses the FastAPI boundary, or
gets persisted to the checkpoint store is represented here as a Pydantic
model. This gives us validation "for free" at every boundary (requirement:
input validation at every boundary) and makes the LangGraph state
(`service/state.py`) a typed composition of these models rather than a bag
of loosely-typed dict keys.

All applicant data in this repository is synthetic and fictional, generated
for demonstration purposes against the invented "Northbridge Financial
Group" brand. Nothing here models a real person or a real institution's
policies.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    """Single source of truth for "now" so timestamps are consistently tz-aware."""
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    """Generate a short, prefixed, collision-resistant identifier.

    Prefixing (e.g. `app_`, `esc_`) makes IDs self-describing in logs and
    audit trails without needing a lookup.
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class EmploymentStatus(str, Enum):
    """Permitted, plainly-observable employment categories for the demo."""

    EMPLOYED_FULL_TIME = "employed_full_time"
    EMPLOYED_PART_TIME = "employed_part_time"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"


class LoanPurpose(str, Enum):
    """Synthetic taxonomy of loan purposes used to drive policy rules."""

    AUTO = "auto"
    HOME_IMPROVEMENT = "home_improvement"
    DEBT_CONSOLIDATION = "debt_consolidation"
    PERSONAL = "personal"
    SMALL_BUSINESS = "small_business"


class LoanApplication(BaseModel):
    """The inbound application payload submitted via `POST /applications`.

    `raw_financial_document_text` stands in for an uploaded pay-stub or
    financial-statement PDF: reviewers of this repo run it against plain
    text so the whole pipeline is exercisable with no file-parsing
    dependency and no real documents. The `document_extraction_agent` node
    is what turns this free text into structured `ExtractedFinancials`.
    """

    application_id: str = Field(default_factory=lambda: _new_id("app"))
    applicant_full_name: str = Field(..., min_length=1, max_length=200)
    requested_amount: float = Field(..., gt=0, le=1_000_000)
    loan_purpose: LoanPurpose
    employment_status: EmploymentStatus
    stated_annual_income: float = Field(..., ge=0, le=10_000_000)
    stated_monthly_debt: float = Field(..., ge=0, le=1_000_000)
    raw_financial_document_text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description=(
            "Synthetic pay-stub / financial-statement TEXT (not a real file "
            "upload) that the document_extraction_agent parses."
        ),
    )
    submitted_at: datetime = Field(default_factory=_utcnow)

    @field_validator("applicant_full_name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("applicant_full_name must not be blank")
        return v


class ExtractedFinancials(BaseModel):
    """Structured fields the document_extraction_agent produces from raw text.

    Confidence and source excerpts are kept alongside the numbers so a human
    reviewer (or the explainability agent) can trace a number back to the
    text it was derived from -- a small but deliberate transparency feature.
    """

    monthly_gross_income: float = Field(..., ge=0)
    monthly_net_income: float = Field(..., ge=0)
    existing_monthly_debt_payments: float = Field(..., ge=0)
    employer_name: str | None = None
    pay_frequency: str | None = None
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    source_excerpts: list[str] = Field(default_factory=list)


class CreditBand(str, Enum):
    """Illustrative credit-score bands used only for this demo's mock bureau."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class CreditBureauReport(BaseModel):
    """Response shape returned by the mock credit bureau tool."""

    report_id: str = Field(default_factory=lambda: _new_id("crb"))
    credit_score: int = Field(..., ge=300, le=850)
    credit_band: CreditBand
    open_tradelines: int = Field(..., ge=0)
    delinquencies_last_24mo: int = Field(..., ge=0)
    retrieved_at: datetime = Field(default_factory=_utcnow)


class RiskTier(str, Enum):
    """Coarse risk classification driving the routing decision in the graph."""

    LOW = "low"
    MEDIUM = "medium"
    BORDERLINE = "borderline"
    HIGH = "high"


class RiskAssessment(BaseModel):
    """Output of the risk_scoring_agent node."""

    debt_to_income_ratio: float = Field(..., ge=0)
    credit_report: CreditBureauReport
    risk_tier: RiskTier
    requires_human_escalation: bool
    rationale_factors: list[str] = Field(
        default_factory=list,
        description="Human-readable factors that drove the risk tier, for downstream explainability.",
    )


class PolicySeverity(str, Enum):
    """Severity of a policy flag; INFO flags do not block, WARN/BLOCK do."""

    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


class PolicyFlag(BaseModel):
    """A single finding raised by the policy_compliance_agent."""

    code: str
    description: str
    severity: PolicySeverity


class PolicyResult(BaseModel):
    """Output of the policy_compliance_agent node.

    `permitted_factors_used` / `excluded_factors_checked` exist to make the
    "only permitted factors drove this decision" claim auditable: the
    explainability agent is only allowed to cite factors present in
    `permitted_factors_used`. This is an illustrative, educational pattern
    for how a fair-lending-style consistency check might be represented in
    code -- it is not a certification of compliance with any specific law
    or regulator's rule.
    """

    passed: bool
    flags: list[PolicyFlag] = Field(default_factory=list)
    permitted_factors_used: list[str] = Field(default_factory=list)
    excluded_factors_checked: list[str] = Field(
        default_factory=list,
        description="Factors explicitly verified as NOT used in the rationale (e.g. age, zip code).",
    )


class DecisionOutcome(str, Enum):
    """Final adjudication outcomes."""

    APPROVED = "approved"
    DENIED = "denied"


class Decision(BaseModel):
    """Output of the explainability_agent + decision_output nodes.

    `adverse_action_reasons` is populated whenever `outcome == DENIED`,
    mirroring the real-world practice of providing specific, factor-level
    reasons for an adverse credit decision (again: illustrative pattern,
    not a compliance certification).
    """

    outcome: DecisionOutcome
    approved_amount: float | None = None
    rationale: str = Field(..., min_length=1)
    adverse_action_reasons: list[str] = Field(default_factory=list)
    decided_by: str = Field(
        default="explainability_agent",
        description="'explainability_agent' for straight-through decisions, or the reviewer id after escalation.",
    )
    decided_at: datetime = Field(default_factory=_utcnow)


class EscalationRequest(BaseModel):
    """Recorded when the graph pauses at the human_escalation node."""

    escalation_id: str = Field(default_factory=lambda: _new_id("esc"))
    reason: str
    risk_summary: str
    requested_at: datetime = Field(default_factory=_utcnow)


class EscalationResolution(BaseModel):
    """The payload a human reviewer submits via `/escalation-resolve`."""

    reviewer_id: str = Field(..., min_length=1, max_length=100)
    approve: bool
    notes: str = Field(default="", max_length=2000)
