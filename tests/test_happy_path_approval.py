"""Happy-path integration test: a low-risk applicant should be approved
straight through, with no escalation, and a rationale naming only
favorable/permitted factors."""

from __future__ import annotations

from tests.conftest import build_pay_stub_text
from underwriting.domain.models import LoanApplication


def test_happy_path_produces_approval(service, make_application) -> None:
    payload = make_application(
        applicant_full_name="Alex B. Approved",  # deterministic mock bureau -> 'good' band
        requested_amount=15000.0,
        stated_annual_income=62400.0,
        stated_monthly_debt=400.0,
        raw_financial_document_text=build_pay_stub_text(
            gross_pay_per_period=2400.0, net_pay_per_period=1850.0, existing_monthly_debt=400.0
        ),
    )
    application = LoanApplication.model_validate(payload)

    result = service.start_application(application)

    assert result["status"] == "completed"
    assert result["risk_assessment"]["risk_tier"] == "low"
    assert result["decision"]["outcome"] == "approved"
    assert result["decision"]["approved_amount"] == 15000.0
    assert result["decision"]["adverse_action_reasons"] == []
    # No escalation should have occurred on this path.
    assert result["escalation_resolution"] is None
    assert result["pending_escalation"] is None


def test_happy_path_rationale_cites_permitted_factors(service, make_application) -> None:
    payload = make_application()
    application = LoanApplication.model_validate(payload)

    result = service.start_application(application)
    rationale = result["decision"]["rationale"]

    assert "APPROVED" in rationale
    assert "debt-to-income ratio" in rationale
    assert "credit band" in rationale
    # Every permitted factor name should be listed in the rationale text.
    for factor in result["policy_result"]["permitted_factors_used"]:
        assert factor in rationale


def test_happy_path_audit_trail_covers_every_node(service, make_application) -> None:
    payload = make_application()
    application = LoanApplication.model_validate(payload)

    result = service.start_application(application)
    actions = [entry["action"] for entry in result["audit_trail"]]

    assert actions == [
        "application_received",
        "financials_extracted",
        "risk_scored",
        "policy_evaluated",
        "rationale_generated",
        "decision_finalized",
    ]
    # Every audit entry should carry the same correlation id (one run == one correlation id).
    correlation_ids = {entry["correlation_id"] for entry in result["audit_trail"]}
    assert len(correlation_ids) == 1
