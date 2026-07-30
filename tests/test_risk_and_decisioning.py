"""Unit tests for domain/risk.py (DTI + tier classification) and
domain/decisioning.py (outcome determination). Pure functions, no I/O."""

from __future__ import annotations

from underwriting.config import Settings
from underwriting.domain.decisioning import determine_outcome
from underwriting.domain.models import (
    CreditBand,
    CreditBureauReport,
    DecisionOutcome,
    EmploymentStatus,
    EscalationResolution,
    LoanApplication,
    LoanPurpose,
    PolicyResult,
    RiskAssessment,
    RiskTier,
)
from underwriting.domain.risk import classify_risk_tier, compute_debt_to_income_ratio


def _settings() -> Settings:
    return Settings(
        checkpoint_db_path=":memory:",
        openai_api_key=None,
        anthropic_api_key=None,
        dti_borderline_low=0.36,
        dti_borderline_high=0.45,
        dti_hard_decline=0.55,
    )


def test_compute_dti_handles_zero_income_safely() -> None:
    assert compute_debt_to_income_ratio(monthly_income=0, monthly_debt_payments=500) == 1.0


def test_compute_dti_basic_math() -> None:
    assert compute_debt_to_income_ratio(monthly_income=5000, monthly_debt_payments=1000) == 0.2


def test_classify_risk_tier_low() -> None:
    tier = classify_risk_tier(dti=0.10, credit_band=CreditBand.EXCELLENT, settings=_settings())
    assert tier == RiskTier.LOW


def test_classify_risk_tier_borderline_by_dti() -> None:
    tier = classify_risk_tier(dti=0.40, credit_band=CreditBand.GOOD, settings=_settings())
    assert tier == RiskTier.BORDERLINE


def test_classify_risk_tier_borderline_by_fair_band() -> None:
    tier = classify_risk_tier(dti=0.10, credit_band=CreditBand.FAIR, settings=_settings())
    assert tier == RiskTier.BORDERLINE


def test_classify_risk_tier_high_by_dti() -> None:
    tier = classify_risk_tier(dti=0.60, credit_band=CreditBand.GOOD, settings=_settings())
    assert tier == RiskTier.HIGH


def test_classify_risk_tier_high_by_poor_band() -> None:
    tier = classify_risk_tier(dti=0.05, credit_band=CreditBand.POOR, settings=_settings())
    assert tier == RiskTier.HIGH


def _application() -> LoanApplication:
    return LoanApplication(
        applicant_full_name="Test Applicant",
        requested_amount=10000.0,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_status=EmploymentStatus.EMPLOYED_FULL_TIME,
        stated_annual_income=60000.0,
        stated_monthly_debt=500.0,
        raw_financial_document_text="EMPLOYER: Acme\n",
    )


def _risk(tier: RiskTier, band: CreditBand = CreditBand.GOOD) -> RiskAssessment:
    report = CreditBureauReport(
        credit_score=700, credit_band=band, open_tradelines=5, delinquencies_last_24mo=0
    )
    return RiskAssessment(
        debt_to_income_ratio=0.5,
        credit_report=report,
        risk_tier=tier,
        requires_human_escalation=(tier == RiskTier.BORDERLINE),
    )


def _policy(passed: bool = True) -> PolicyResult:
    return PolicyResult(passed=passed, flags=[], permitted_factors_used=["debt_to_income_ratio"])


def test_determine_outcome_approves_low_risk() -> None:
    outcome, amount, reasons = determine_outcome(_application(), _risk(RiskTier.LOW), _policy())
    assert outcome == DecisionOutcome.APPROVED
    assert amount == 10000.0
    assert reasons == []


def test_determine_outcome_denies_high_risk() -> None:
    outcome, amount, reasons = determine_outcome(
        _application(), _risk(RiskTier.HIGH, CreditBand.POOR), _policy()
    )
    assert outcome == DecisionOutcome.DENIED
    assert amount is None
    assert reasons  # at least one adverse reason present


def test_determine_outcome_denies_on_failed_policy_even_if_risk_is_low() -> None:
    outcome, amount, _reasons = determine_outcome(
        _application(), _risk(RiskTier.LOW), _policy(passed=False)
    )
    assert outcome == DecisionOutcome.DENIED
    assert amount is None


def test_determine_outcome_escalation_approve_overrides_borderline_tier() -> None:
    resolution = EscalationResolution(reviewer_id="jdoe123", approve=True)
    outcome, amount, reasons = determine_outcome(
        _application(), _risk(RiskTier.BORDERLINE), _policy(), resolution
    )
    assert outcome == DecisionOutcome.APPROVED
    assert amount == 10000.0
    assert reasons == []


def test_determine_outcome_escalation_deny_is_authoritative() -> None:
    resolution = EscalationResolution(reviewer_id="jdoe123", approve=False, notes="declined")
    outcome, amount, reasons = determine_outcome(
        _application(), _risk(RiskTier.LOW), _policy(), resolution
    )
    # Even though the automated risk tier was LOW, the human reviewer's
    # decision (once escalated) is authoritative.
    assert outcome == DecisionOutcome.DENIED
    assert amount is None
    assert reasons
