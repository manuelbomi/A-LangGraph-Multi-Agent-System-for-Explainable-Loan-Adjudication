"""Unit tests for the synthetic lending policy ruleset (domain/policy.py).

Pure function tests -- no graph, no chat model, no I/O -- exercising the
policy logic directly against hand-built domain models.
"""

from __future__ import annotations

from underwriting.domain.models import (
    CreditBand,
    CreditBureauReport,
    EmploymentStatus,
    ExtractedFinancials,
    LoanApplication,
    LoanPurpose,
    PolicySeverity,
    RiskAssessment,
    RiskTier,
)
from underwriting.domain.policy import EXCLUDED_FACTORS, PERMITTED_FACTORS, evaluate_policy


def _application(**overrides) -> LoanApplication:
    defaults = dict(
        applicant_full_name="Test Applicant",
        requested_amount=10000.0,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_status=EmploymentStatus.EMPLOYED_FULL_TIME,
        stated_annual_income=60000.0,
        stated_monthly_debt=500.0,
        raw_financial_document_text="EMPLOYER: Acme\nPAY_FREQUENCY: monthly\n",
    )
    defaults.update(overrides)
    return LoanApplication(**defaults)


def _extracted(**overrides) -> ExtractedFinancials:
    defaults = dict(
        monthly_gross_income=5000.0,
        monthly_net_income=4000.0,
        existing_monthly_debt_payments=500.0,
        extraction_confidence=0.9,
    )
    defaults.update(overrides)
    return ExtractedFinancials(**defaults)


def _risk(**overrides) -> RiskAssessment:
    report = CreditBureauReport(
        credit_score=700, credit_band=CreditBand.GOOD, open_tradelines=5, delinquencies_last_24mo=0
    )
    defaults = dict(
        debt_to_income_ratio=0.10,
        credit_report=report,
        risk_tier=RiskTier.LOW,
        requires_human_escalation=False,
    )
    defaults.update(overrides)
    return RiskAssessment(**defaults)


def test_clean_application_passes_with_no_block_flags() -> None:
    result = evaluate_policy(_application(), _extracted(), _risk())
    assert result.passed is True
    assert all(f.severity != PolicySeverity.BLOCK for f in result.flags)


def test_over_leveraged_request_is_blocked() -> None:
    app = _application(requested_amount=400000.0, stated_annual_income=60000.0)  # > 5x income
    result = evaluate_policy(app, _extracted(), _risk())
    assert result.passed is False
    codes = [f.code for f in result.flags]
    assert "OVER_LEVERAGED_REQUEST" in codes


def test_low_extraction_confidence_is_blocked() -> None:
    result = evaluate_policy(_application(), _extracted(extraction_confidence=0.2), _risk())
    assert result.passed is False
    assert "LOW_EXTRACTION_CONFIDENCE" in [f.code for f in result.flags]


def test_stated_vs_extracted_debt_mismatch_is_warned_not_blocked() -> None:
    app = _application(stated_monthly_debt=500.0)
    extracted = _extracted(existing_monthly_debt_payments=2000.0)  # far off from stated
    result = evaluate_policy(app, extracted, _risk())
    flag = next(f for f in result.flags if f.code == "DEBT_STATEMENT_INCONSISTENCY")
    assert flag.severity == PolicySeverity.WARN
    # A WARN flag alone must not fail the overall policy check.
    assert result.passed is True


def test_self_employed_gets_informational_flag_only() -> None:
    app = _application(employment_status=EmploymentStatus.SELF_EMPLOYED)
    result = evaluate_policy(app, _extracted(), _risk())
    flag = next(f for f in result.flags if f.code == "SELF_EMPLOYED_INCOME_VARIABILITY")
    assert flag.severity == PolicySeverity.INFO
    assert result.passed is True


def test_permitted_and_excluded_factors_are_always_reported() -> None:
    result = evaluate_policy(_application(), _extracted(), _risk())
    assert result.permitted_factors_used == list(PERMITTED_FACTORS)
    assert result.excluded_factors_checked == list(EXCLUDED_FACTORS)
    # None of the excluded (protected-class-adjacent) factors overlap with
    # the permitted set -- a basic consistency guarantee about the ruleset.
    assert set(PERMITTED_FACTORS).isdisjoint(set(EXCLUDED_FACTORS))
