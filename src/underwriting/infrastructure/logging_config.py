"""
Structured (JSON) logging configuration.

Architectural role
-------------------
Configures the root logger once, at process startup (`configure_logging`,
called from `api/main.py`'s FastAPI lifespan), so every log line emitted
anywhere in the app -- graph nodes, infrastructure clients, API routes --
comes out as a single-line JSON object suitable for ingestion by a log
aggregator (CloudWatch, Splunk, ELK, Datadog, etc.) in a real deployment.

Correlation IDs
---------------
`correlation_id_var` is a `contextvars.ContextVar` set once per API request
(see `api/main.py`'s middleware) and automatically attached to every log
record via `CorrelationIdFilter`, without every call site having to thread
a `correlation_id` kwarg through manually. This is what lets an operator
grep one request's entire multi-agent execution out of a shared log stream.

Secret safety
-------------
This module never logs configuration values directly -- `Settings` keeps
API keys as `SecretStr` (see config.py), and any application/applicant data
passed as `extra=` is expected to already be redacted via
`infrastructure/pii_redaction.py` before it reaches the logger (see
`infrastructure/audit_log.py` for the canonical example).
"""

from __future__ import annotations

import contextvars
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from pythonjsonlogger import jsonlogger

# Per-request correlation id, read by CorrelationIdFilter on every log
# record. Defaults to None outside of a request context (e.g. at startup).
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIdFilter(logging.Filter):
    """Attaches the current request's correlation id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get()
        return True


class _JsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter with a consistent field set across every log line."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        # Ensure the key is always present (even as null) so downstream log
        # queries can filter/group on it consistently.
        log_record.setdefault("correlation_id", getattr(record, "correlation_id", None))


def configure_logging(*, service_name: str, log_level: str = "INFO") -> None:
    """Configure the root logger for JSON structured output to stdout.

    Idempotent-ish: clears any handlers previously attached by this
    function (relevant in tests, which may call it multiple times across
    the FastAPI TestClient lifespan) before attaching a fresh handler.
    """
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    # Remove handlers this function previously added, identified by a
    # marker attribute, so repeated calls (e.g. across pytest test
    # functions that each spin up a TestClient) don't duplicate log lines.
    for h in list(root.handlers):
        if getattr(h, "_underwriting_managed", False):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.addFilter(CorrelationIdFilter())
    formatter = _JsonFormatter(fmt="%(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    handler.__dict__["_underwriting_managed"] = True
    root.addHandler(handler)

    logging.getLogger(__name__).info(
        "logging_configured", extra={"service_name": service_name, "log_level": log_level}
    )
