"""Unit tests for the mock credit bureau tool (infrastructure/credit_bureau.py)."""

from __future__ import annotations

from underwriting.domain.models import CreditBand
from underwriting.infrastructure.credit_bureau import get_credit_report


def test_credit_report_is_deterministic_for_same_name() -> None:
    r1 = get_credit_report("Jordan A. Applicant")
    r2 = get_credit_report("Jordan A. Applicant")
    assert r1.credit_score == r2.credit_score
    assert r1.credit_band == r2.credit_band


def test_credit_report_score_within_valid_range() -> None:
    report = get_credit_report("Any Applicant Name")
    assert 300 <= report.credit_score <= 850


def test_credit_report_band_matches_score_thresholds() -> None:
    report = get_credit_report("Riley T. Thin")
    if report.credit_score >= 740:
        assert report.credit_band == CreditBand.EXCELLENT
    elif report.credit_score >= 670:
        assert report.credit_band == CreditBand.GOOD
    elif report.credit_score >= 580:
        assert report.credit_band == CreditBand.FAIR
    else:
        assert report.credit_band == CreditBand.POOR


def test_credit_report_name_casing_and_whitespace_do_not_change_score() -> None:
    r1 = get_credit_report("Pat Q. Applicant")
    r2 = get_credit_report("  pat q. applicant  ")
    assert r1.credit_score == r2.credit_score
