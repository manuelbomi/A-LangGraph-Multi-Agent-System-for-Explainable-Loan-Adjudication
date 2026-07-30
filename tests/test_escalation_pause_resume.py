"""Escalation-path integration test: a borderline-risk applicant must
cause the graph run to durably PAUSE at the human_escalation node (via
LangGraph's interrupt/checkpoint mechanism), and must resume correctly
from the checkpoint once a reviewer's decision is submitted."""

from __future__ import annotations

import pytest

from tests.conftest import build_pay_stub_text
from underwriting.domain.errors import (
    ApplicationNotAwaitingEscalationError,
    ApplicationNotFoundError,
)
from underwriting.domain.models import EscalationResolution, LoanApplication


def _borderline_application_payload(make_application):
    # DTI = 2080 / (2400 * 26/12) = 2080 / 5200 = 40.0%, inside the
    # [0.36, 0.45) borderline band, with a 'good' credit band (Alex B.
    # Approved) so the credit band itself doesn't independently force
    # HIGH or BORDERLINE -- the DTI band is what's under test here.
    return make_application(
        applicant_full_name="Alex B. Approved",
        requested_amount=15000.0,
        stated_annual_income=62400.0,
        stated_monthly_debt=2080.0,
        raw_financial_document_text=build_pay_stub_text(existing_monthly_debt=2080.0),
    )


def test_run_pauses_at_human_escalation(service, make_application) -> None:
    application = LoanApplication.model_validate(_borderline_application_payload(make_application))

    result = service.start_application(application)

    assert result["status"] == "awaiting_escalation"
    assert result["risk_assessment"]["risk_tier"] == "borderline"
    assert result["decision"] is None  # graph genuinely stopped before explainability_agent
    assert result["pending_escalation"] is not None
    assert "borderline" in result["pending_escalation"]["reason"]


def test_resume_after_escalation_produces_approval(service, make_application) -> None:
    application = LoanApplication.model_validate(_borderline_application_payload(make_application))
    paused = service.start_application(application)
    assert paused["status"] == "awaiting_escalation"

    resolution = EscalationResolution(
        reviewer_id="jdoe123", approve=True, notes="Manually verified additional income."
    )
    resumed = service.resolve_escalation(application.application_id, resolution)

    assert resumed["status"] == "completed"
    assert resumed["decision"]["outcome"] == "approved"
    assert resumed["decision"]["decided_by"] == "human_reviewer:jdoe123"
    assert resumed["escalation_resolution"]["reviewer_id"] == "jdoe123"
    assert "jdoe123" in resumed["decision"]["rationale"]
    # The full audit trail (both pre- and post-pause) must be present.
    actions = [entry["action"] for entry in resumed["audit_trail"]]
    assert actions == [
        "application_received",
        "financials_extracted",
        "risk_scored",
        "policy_evaluated",
        "escalation_requested",
        "escalation_resolved",
        "rationale_generated",
        "decision_finalized",
    ]


def test_resume_after_escalation_can_produce_denial(service, make_application) -> None:
    application = LoanApplication.model_validate(_borderline_application_payload(make_application))
    service.start_application(application)

    resolution = EscalationResolution(
        reviewer_id="msmith456", approve=False, notes="Insufficient verifiable income."
    )
    resumed = service.resolve_escalation(application.application_id, resolution)

    assert resumed["status"] == "completed"
    assert resumed["decision"]["outcome"] == "denied"
    assert resumed["decision"]["approved_amount"] is None
    assert resumed["decision"]["decided_by"] == "human_reviewer:msmith456"


def test_resolving_an_unknown_application_raises(service) -> None:
    resolution = EscalationResolution(reviewer_id="jdoe123", approve=True)
    with pytest.raises(ApplicationNotFoundError):
        service.resolve_escalation("app_does_not_exist", resolution)


def test_resolving_a_non_paused_application_raises(service, make_application) -> None:
    # A LOW-risk application completes straight through -- never pauses.
    payload = make_application()
    application = LoanApplication.model_validate(payload)
    service.start_application(application)

    resolution = EscalationResolution(reviewer_id="jdoe123", approve=True)
    with pytest.raises(ApplicationNotAwaitingEscalationError):
        service.resolve_escalation(application.application_id, resolution)
