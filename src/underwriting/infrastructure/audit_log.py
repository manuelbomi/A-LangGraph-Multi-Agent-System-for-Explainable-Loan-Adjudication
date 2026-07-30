"""
Audit logging for agent/tool actions.

Architectural role
-------------------
A regulated-bank underwriting system needs a durable, append-only record of
*what each agent did and why*, independent of the final decision rationale
shown to an applicant. `record_audit_event` is called by every graph node
after it does anything decision-relevant (parsing a document, calling the
credit bureau, applying policy, escalating, deciding). Entries are:

  * Emitted to the structured logger (so they land in whatever log
    aggregation a deployment wires up), and
  * Returned to the caller so the node can also append them to the graph
    state's `audit_trail` list -- making the audit trail retrievable
    directly from `GET /applications/{id}` without a separate log query.

Every event is passed through `redact_mapping` before it is logged or
persisted, so no PII from the synthetic applicant payload leaks into the
audit trail in the clear.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from underwriting.infrastructure.pii_redaction import redact_mapping

logger = logging.getLogger("underwriting.audit")


def record_audit_event(
    *,
    application_id: str,
    correlation_id: str,
    actor: str,
    action: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, log, and return a single redacted audit event.

    Args:
        application_id: The loan application this event pertains to.
        correlation_id: Request/run correlation id, ties the event back to
            a specific API call and to the structured log lines emitted
            around it.
        actor: Which agent/node/human performed the action, e.g.
            "document_extraction_agent" or "human_reviewer:jdoe123".
        action: A short, stable event code, e.g. "financials_extracted",
            "credit_report_retrieved", "policy_evaluated", "escalated",
            "escalation_resolved", "decision_finalized".
        details: Arbitrary structured context. Redacted before logging.
    """
    event = {
        "event_time": datetime.now(timezone.utc).isoformat(),
        "application_id": application_id,
        "correlation_id": correlation_id,
        "actor": actor,
        "action": action,
        "details": redact_mapping(details or {}),
    }
    logger.info("audit_event", extra={"audit": event})
    return event
