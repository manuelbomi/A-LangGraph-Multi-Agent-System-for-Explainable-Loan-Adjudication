"""
PII redaction utilities.

Architectural role
-------------------
A regulated-bank-style system must never write raw personally identifiable
information into logs, traces, or audit records that might be shipped to a
lower-trust log aggregator or retained longer than the source system.
Every place in this codebase that logs applicant data (logging_config's
formatter, audit_log.py, the API's exception handlers) routes through
`redact_text` / `redact_mapping` here first.

This is a demo-grade implementation: regex-based pattern matching for
common structured PII (emails, phone numbers, SSNs, card-like numbers) plus
key-name-based redaction for known-sensitive dict fields. A production
deployment at a real bank would likely layer a vendor DLP library or a
named-entity-recognition model on top of this for freeform text; the
pattern-based approach here is intentionally simple, auditable, and
dependency-free.
"""

from __future__ import annotations

import re
from typing import Any

_REDACTED = "***REDACTED***"

# Ordered so more-specific patterns (SSN) are tried before looser ones
# (generic long-digit sequences) would be, if we added those later.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]

# Dict keys whose values are always fully redacted regardless of content,
# when logging structured payloads (e.g. an application dict) via
# `redact_mapping`.
SENSITIVE_KEYS = {
    "applicant_full_name",
    "ssn",
    "social_security_number",
    "email",
    "phone",
    "phone_number",
    "address",
    "employer_name",
    "reviewer_id",
}


def redact_text(text: str) -> str:
    """Scrub common PII patterns out of a free-text string.

    Returns a new string; never mutates the input. Safe to call on
    already-redacted text (idempotent, since `***REDACTED***` matches none
    of the patterns).
    """
    if not text:
        return text
    redacted = text
    for _label, pattern in _PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


def redact_mapping(data: dict[str, Any], *, extra_sensitive_keys: set[str] | None = None) -> dict[str, Any]:
    """Recursively redact a dict, masking known-sensitive keys and scrubbing text values.

    Used before any application/applicant payload is written to a log
    record or an audit entry. Non-string, non-dict, non-list values (ints,
    floats, bools, None) pass through unchanged -- they're either not PII
    (amounts, ratios, timestamps) or are already governed by key-name
    redaction.
    """
    sensitive = SENSITIVE_KEYS | (extra_sensitive_keys or set())
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in sensitive:
            out[key] = _REDACTED
        elif isinstance(value, dict):
            out[key] = redact_mapping(value, extra_sensitive_keys=extra_sensitive_keys)
        elif isinstance(value, list):
            out[key] = [
                redact_mapping(v, extra_sensitive_keys=extra_sensitive_keys)
                if isinstance(v, dict)
                else (redact_text(v) if isinstance(v, str) else v)
                for v in value
            ]
        elif isinstance(value, str):
            out[key] = redact_text(value)
        else:
            out[key] = value
    return out
