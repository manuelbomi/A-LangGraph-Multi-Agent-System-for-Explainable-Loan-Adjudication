"""Denial-path integration test: a high-DTI, poor-credit-band applicant
should be denied straight through (no escalation, since HIGH risk is
disqualifying rather than borderline), with a rationale that correctly
names the adverse factors present in the synthetic input."""

from __future__ import annotations

from tests.conftest import build_pay_stub_text
from underwriting.domain.models import LoanApplication


def test_denial_path_produces_denial(service, make_application) -> None:
    payload = make_application(
        applicant_full_name="Dana R. Applicant",  # deterministic mock bureau -> 'poor' band
        requested_amount=15000.0,
        stated_annual_income=62400.0,
        stated_monthly_debt=3000.0,
        raw_financial_document_text=build_pay_stub_text(
            gross_pay_per_period=2400.0, net_pay_per_period=1850.0, existing_monthly_debt=3000.0
        ),
    )
    application = LoanApplication.model_validate(payload)

    result = service.start_application(application)

    assert result["status"] == "completed"
    assert result["risk_assessment"]["risk_tier"] == "high"
    assert result["risk_assessment"]["credit_report"]["credit_band"] == "poor"
    assert result["decision"]["outcome"] == "denied"
    assert result["decision"]["approved_amount"] is None
    assert result["escalation_resolution"] is None  # HIGH risk denies straight through


def test_denial_rationale_references_adverse_factors(service, make_application) -> None:
    """The rationale and adverse_action_reasons must name the SPECIFIC
    factors that made this synthetic applicant's input adverse: a DTI
    figure above threshold and a poor credit band."""
    payload = make_application(
        applicant_full_name="Dana R. Applicant",
        stated_monthly_debt=3000.0,
        raw_financial_document_text=build_pay_stub_text(existing_monthly_debt=3000.0),
    )
    application = LoanApplication.model_validate(payload)

    result = service.start_application(application)
    decision = result["decision"]

    assert decision["outcome"] == "denied"
    reasons_text = " ".join(decision["adverse_action_reasons"])
    assert "debt-to-income ratio" in reasons_text.lower()
    assert "poor" in reasons_text.lower()

    rationale = decision["rationale"]
    assert "DENIED" in rationale
    # The rationale must be numerically consistent with the computed DTI.
    dti_pct = f"{result['risk_assessment']['debt_to_income_ratio']:.1%}"
    assert dti_pct in rationale
