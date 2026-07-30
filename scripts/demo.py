#!/usr/bin/env python
"""
Offline, end-to-end demo of the underwriting graph.

Runs all three decision paths (straight-through approval, straight-through
denial, and human-escalation pause + resume) directly against
`ApplicationService`, with no HTTP server and no LLM API key required. This
is the fastest way for a reviewer to see the whole system work end-to-end
after `make install` -- see README "Getting Started".

Usage:
    python scripts/demo.py
"""

from __future__ import annotations

import json

from underwriting.config import Settings
from underwriting.domain.models import (
    EmploymentStatus,
    EscalationResolution,
    LoanApplication,
    LoanPurpose,
)
from underwriting.infrastructure.checkpointer import close_checkpointer, open_checkpointer
from underwriting.infrastructure.llm import get_chat_model
from underwriting.service.application_service import ApplicationService


def build_pay_stub_text(
    *,
    employer: str = "Riverstone Logistics LLC",
    pay_frequency: str = "biweekly",
    gross_pay_per_period: float = 2400.0,
    net_pay_per_period: float = 1850.0,
    existing_monthly_debt: float = 400.0,
) -> str:
    """Synthetic pay-stub TEXT in the labeled format document_extraction_agent parses."""
    return (
        "NORTHBRIDGE FINANCIAL GROUP - SYNTHETIC PAY STATEMENT (DEMO DATA ONLY)\n"
        f"EMPLOYER: {employer}\n"
        f"PAY_FREQUENCY: {pay_frequency}\n"
        f"GROSS_PAY_PER_PERIOD: ${gross_pay_per_period:.2f}\n"
        f"NET_PAY_PER_PERIOD: ${net_pay_per_period:.2f}\n"
        f"EXISTING_MONTHLY_DEBT: ${existing_monthly_debt:.2f}\n"
    )


def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _print_summary(result: dict) -> None:
    print(f"status:           {result['status']}")
    if result.get("risk_assessment"):
        risk = result["risk_assessment"]
        print(f"risk_tier:        {risk['risk_tier']}  (DTI={risk['debt_to_income_ratio']:.1%}, "
              f"credit_band={risk['credit_report']['credit_band']})")
    if result.get("pending_escalation"):
        print(f"pending_escalation: {result['pending_escalation']['reason']}")
    if result.get("decision"):
        decision = result["decision"]
        print(f"decision:         {decision['outcome']} (decided_by={decision['decided_by']})")
        print(f"rationale:        {decision['rationale']}")
        if decision["adverse_action_reasons"]:
            print(f"adverse_reasons:  {json.dumps(decision['adverse_action_reasons'], indent=2)}")


def main() -> None:
    settings = Settings(checkpoint_db_path=":memory:", openai_api_key=None, anthropic_api_key=None)
    chat_model = get_chat_model(settings)
    checkpointer = open_checkpointer(":memory:")
    service = ApplicationService(settings=settings, chat_model=chat_model, checkpointer=checkpointer)

    _print_section("1. Straight-through APPROVAL (low risk)")
    approved_app = LoanApplication(
        applicant_full_name="Alex B. Approved",
        requested_amount=15000.0,
        loan_purpose=LoanPurpose.AUTO,
        employment_status=EmploymentStatus.EMPLOYED_FULL_TIME,
        stated_annual_income=62400.0,
        stated_monthly_debt=400.0,
        raw_financial_document_text=build_pay_stub_text(existing_monthly_debt=400.0),
    )
    _print_summary(service.start_application(approved_app))

    _print_section("2. Straight-through DENIAL (high risk)")
    denied_app = LoanApplication(
        applicant_full_name="Dana R. Applicant",
        requested_amount=15000.0,
        loan_purpose=LoanPurpose.DEBT_CONSOLIDATION,
        employment_status=EmploymentStatus.EMPLOYED_FULL_TIME,
        stated_annual_income=62400.0,
        stated_monthly_debt=3000.0,
        raw_financial_document_text=build_pay_stub_text(existing_monthly_debt=3000.0),
    )
    _print_summary(service.start_application(denied_app))

    _print_section("3. Borderline risk -> human ESCALATION -> resume")
    escalated_app = LoanApplication(
        applicant_full_name="Alex B. Approved",
        requested_amount=15000.0,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_status=EmploymentStatus.EMPLOYED_FULL_TIME,
        stated_annual_income=62400.0,
        stated_monthly_debt=2080.0,
        raw_financial_document_text=build_pay_stub_text(existing_monthly_debt=2080.0),
    )
    paused = service.start_application(escalated_app)
    print("--- run paused, awaiting human reviewer ---")
    _print_summary(paused)

    resolution = EscalationResolution(
        reviewer_id="jdoe123",
        approve=True,
        notes="Verified additional freelance income manually; approving.",
    )
    resumed = service.resolve_escalation(escalated_app.application_id, resolution)
    print("\n--- run resumed after reviewer decision ---")
    _print_summary(resumed)

    close_checkpointer()
    print("\nDone. All three decision paths executed fully offline, no API key required.")


if __name__ == "__main__":
    main()
