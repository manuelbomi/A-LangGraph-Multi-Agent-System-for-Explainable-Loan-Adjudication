"""
Lightweight OpenTelemetry-style tracing spans.

Architectural role
-------------------
Every graph node wraps its work in `start_span(...)`, producing a span
record shaped like a simplified OTel span (name, span_id, start/end time,
duration, attributes, status). Spans accumulate in the graph state's
`trace_spans` list (see service/state.py) so the full execution trace of a
run is visible via `GET /applications/{id}` -- genuinely useful for
debugging *and* for governance review ("show me exactly what each agent did
and how long it took").

This module deliberately does NOT depend on the `opentelemetry-sdk`
package: pulling in a full OTel SDK plus exporter config would be
disproportionate for a portfolio demo whose priority is running with zero
external services. Instead it produces the same *shape* of data
(span name, id, timing, attributes) so wiring in a real OTel SDK later is a
mechanical swap -- see the README's Observability section for exactly how
this would plug into an OTLP exporter feeding Prometheus/Grafana/Datadog in
a production deployment.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Context manager producing a mutable span record.

    Usage::

        with start_span("risk_scoring_agent", application_id=app_id) as span:
            ... do work ...
            span["attributes"]["risk_tier"] = "borderline"

    The yielded dict is the span record itself -- callers may add
    attributes during the `with` block. On exit, `duration_ms`, `end_time`,
    and `status` are filled in automatically. Exceptions are recorded on the
    span (status="error") and then re-raised so node error handling is
    unaffected by tracing.
    """
    span: dict[str, Any] = {
        "span_id": uuid.uuid4().hex[:16],
        "name": name,
        "attributes": dict(attributes),
        "start_time": _utcnow_iso(),
        "status": "ok",
    }
    perf_start = time.perf_counter()
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = repr(exc)
        raise
    finally:
        span["duration_ms"] = round((time.perf_counter() - perf_start) * 1000, 3)
        span["end_time"] = _utcnow_iso()
