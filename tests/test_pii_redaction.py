"""Unit tests for the PII redaction utility (infrastructure/pii_redaction.py)."""

from __future__ import annotations

from underwriting.infrastructure.pii_redaction import redact_mapping, redact_text


def test_redact_text_masks_email() -> None:
    result = redact_text("Contact jane.doe@example.com for details.")
    assert "jane.doe@example.com" not in result
    assert "REDACTED" in result


def test_redact_text_masks_ssn() -> None:
    result = redact_text("SSN on file: 123-45-6789.")
    assert "123-45-6789" not in result
    assert "REDACTED" in result


def test_redact_text_masks_phone() -> None:
    result = redact_text("Call me at (555) 867-5309.")
    assert "867-5309" not in result


def test_redact_text_is_idempotent() -> None:
    once = redact_text("Email me at jane.doe@example.com")
    twice = redact_text(once)
    assert once == twice


def test_redact_text_leaves_non_pii_untouched() -> None:
    text = "Debt-to-income ratio is 42.0%, risk tier is borderline."
    assert redact_text(text) == text


def test_redact_mapping_masks_sensitive_keys() -> None:
    data = {
        "applicant_full_name": "Jane Q. Applicant",
        "requested_amount": 15000.0,
        "nested": {"employer_name": "Acme Corp", "note": "fine"},
    }
    result = redact_mapping(data)
    assert result["applicant_full_name"] == "***REDACTED***"
    assert result["requested_amount"] == 15000.0
    assert result["nested"]["employer_name"] == "***REDACTED***"
    assert result["nested"]["note"] == "fine"


def test_redact_mapping_scrubs_pii_inside_list_of_strings() -> None:
    data = {"notes": ["reach me at jane.doe@example.com", "no pii here"]}
    result = redact_mapping(data)
    assert "jane.doe@example.com" not in result["notes"][0]
    assert result["notes"][1] == "no pii here"
