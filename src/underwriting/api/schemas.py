"""
API request/response schemas.

Kept separate from `domain/models.py` even though there's overlap, because
the API's contract with clients (what shape a response takes over the
wire) and the domain's internal representation are allowed to evolve
independently. Reusing domain models directly as request bodies
(`LoanApplication`, `EscalationResolution`) is fine where the shapes
genuinely are the same thing; response envelopes get their own models here
so the HTTP contract doesn't silently change if a domain model gains an
internal-only field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApplicationStateResponse(BaseModel):
    """Full state/history snapshot of one application's graph run."""

    application_id: str
    status: str = Field(
        ..., description="in_progress | awaiting_escalation | escalation_resolved | completed"
    )
    application: dict[str, Any] | None = None
    extracted_financials: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None
    policy_result: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    pending_escalation: dict[str, Any] | None = Field(
        default=None,
        description="Present only while status == awaiting_escalation.",
    )
    escalation_request: dict[str, Any] | None = None
    escalation_resolution: dict[str, Any] | None = None
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    trace_spans: list[dict[str, Any]] = Field(default_factory=list)


class RationaleResponse(BaseModel):
    """Focused response for `GET /applications/{id}/rationale`."""

    application_id: str
    outcome: str
    rationale: str
    adverse_action_reasons: list[str]
    decided_by: str


class HealthResponse(BaseModel):
    """Response shape for both /healthz and /readyz."""

    status: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by exception handlers."""

    error: str
    detail: str
