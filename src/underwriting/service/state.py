"""
LangGraph state schema for the underwriting graph.

Architectural role
-------------------
LangGraph threads a single mutable "state" object through every node. We
define it as a `TypedDict` of plain JSON-compatible values (dicts, strings,
lists) rather than passing live Pydantic model instances around. Two
reasons:

  1. The SQLite checkpointer (infrastructure/checkpointer.py) serializes
     state on every superstep so a run can pause at `human_escalation` and
     be resumed -- potentially by a different process -- after the
     checkpoint was written to disk. Sticking to plain JSON-serializable
     types avoids any ambiguity about how custom objects round-trip through
     that serializer.
  2. It makes the state trivially inspectable via `GET /applications/{id}`
     without a custom encoder: `dict(state)` is already an API response.

Node functions (service/nodes.py) convert to/from the typed Pydantic
domain models (domain/models.py) internally, at the boundary of each node,
so business logic still gets full validation -- only the state envelope
itself is "just dicts".

`audit_trail` and `trace_spans` use LangGraph's `Annotated[..., operator.add]`
reducer convention: instead of each node's return value overwriting the
whole list, LangGraph concatenates it with the existing one. That's what
lets every node append its own audit/trace entries without needing to know
about entries other nodes already wrote.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class UnderwritingState(TypedDict, total=False):
    """The full state threaded through the LangGraph underwriting graph.

    All fields are optional (`total=False`) because the state is built up
    incrementally as the run progresses through nodes -- e.g.
    `risk_assessment` doesn't exist until `risk_scoring_agent` has run.
    """

    # --- Identity / correlation ---
    application_id: str
    correlation_id: str

    # --- Domain payloads, stored as JSON-mode dicts (see module docstring) ---
    #
    # Deliberately typed as plain `dict[str, Any]` rather than
    # `dict[str, Any] | None`: `total=False` already models "not yet
    # present" (a node hasn't run yet) via key absence, which is the only
    # state these ever take on -- no node ever sets one of these keys to a
    # literal `None`. This keeps every node's `state["risk_assessment"]`
    # etc. accesses type-checking cleanly once that node's position in the
    # graph guarantees the key exists.
    application: dict[str, Any]
    extracted_financials: dict[str, Any]
    risk_assessment: dict[str, Any]
    policy_result: dict[str, Any]
    decision: dict[str, Any]
    escalation_request: dict[str, Any]
    escalation_resolution: dict[str, Any]

    # --- Governance / observability trails (append-only via reducer) ---
    audit_trail: Annotated[list[dict[str, Any]], operator.add]
    trace_spans: Annotated[list[dict[str, Any]], operator.add]

    # --- Control flow bookkeeping ---
    status: str  # e.g. "in_progress" | "awaiting_escalation" | "completed"
