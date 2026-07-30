"""
Application-level orchestration sitting between FastAPI routes and the
compiled LangGraph graph.

Architectural role
-------------------
`ApplicationService` is the only thing `api/main.py` talks to. It owns:

  * Starting a new graph run for a submitted `LoanApplication`.
  * Reading back full run state/history for `GET /applications/{id}`.
  * Resuming a paused run with a reviewer's `EscalationResolution`.
  * Extracting just the rationale for `GET /applications/{id}/rationale`.

Keeping this logic out of `api/main.py` means the orchestration is testable
without spinning up FastAPI/uvicorn at all (see tests/test_escalation_*.py),
and means a future second transport (a CLI, a gRPC service, a Slack bot)
could reuse this class unchanged.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from underwriting.config import Settings
from underwriting.domain.errors import (
    ApplicationNotAwaitingEscalationError,
    ApplicationNotFoundError,
    DecisionNotAvailableError,
)
from underwriting.domain.models import EscalationResolution, LoanApplication
from underwriting.infrastructure.llm import ResilientChatModel
from underwriting.service.graph import compile_graph
from underwriting.service.state import UnderwritingState

logger = logging.getLogger(__name__)


class ApplicationService:
    """Orchestrates the underwriting graph lifecycle for one process.

    One instance is created at FastAPI startup (see `api/main.py`) bound to
    the process-wide checkpointer and chat model, and reused across all
    requests -- the compiled graph itself is stateless; all per-application
    state lives in the checkpointer, keyed by `thread_id == application_id`.
    """

    def __init__(self, settings: Settings, chat_model: ResilientChatModel, checkpointer: Any) -> None:
        self._settings = settings
        self._graph = compile_graph(settings, chat_model, checkpointer)

    @staticmethod
    def _config(application_id: str) -> RunnableConfig:
        """LangGraph thread config: `thread_id` is the checkpoint partition key."""
        return RunnableConfig(configurable={"thread_id": application_id})

    def start_application(self, application: LoanApplication) -> dict[str, Any]:
        """Kick off a new graph run for a freshly submitted application.

        Runs synchronously to completion, UNLESS the graph routes into
        `human_escalation`, in which case `graph.invoke` returns as soon as
        the run durably pauses (see service/nodes.py for the interrupt
        mechanics). Either way, the caller gets back the current summary.
        """
        correlation_id = f"corr_{uuid.uuid4().hex[:12]}"
        initial_state: UnderwritingState = {
            "application_id": application.application_id,
            "correlation_id": correlation_id,
            "application": application.model_dump(mode="json"),
            "audit_trail": [],
            "trace_spans": [],
            "status": "in_progress",
        }
        logger.info(
            "application_run_starting",
            extra={"application_id": application.application_id, "correlation_id": correlation_id},
        )
        self._graph.invoke(initial_state, config=self._config(application.application_id))
        return self.get_application(application.application_id)

    def get_application(self, application_id: str) -> dict[str, Any]:
        """Return the full current state + status + audit/trace history.

        Raises `ApplicationNotFoundError` if no run has ever been started
        for this id (LangGraph returns an empty snapshot rather than
        raising for an unknown thread_id, so we translate that here).
        """
        snapshot = self._graph.get_state(self._config(application_id))
        values = snapshot.values or {}
        if not values:
            raise ApplicationNotFoundError(application_id)

        paused = bool(snapshot.next)
        pending_escalation: dict[str, Any] | None = None
        if paused:
            for task in snapshot.tasks:
                for intr in task.interrupts:
                    pending_escalation = intr.value

        status = "awaiting_escalation" if paused else values.get("status", "unknown")

        return {
            "application_id": application_id,
            "status": status,
            "application": values.get("application"),
            "extracted_financials": values.get("extracted_financials"),
            "risk_assessment": values.get("risk_assessment"),
            "policy_result": values.get("policy_result"),
            "decision": values.get("decision"),
            "pending_escalation": pending_escalation,
            "escalation_request": values.get("escalation_request"),
            "escalation_resolution": values.get("escalation_resolution"),
            "audit_trail": values.get("audit_trail", []),
            "trace_spans": values.get("trace_spans", []),
        }

    def resolve_escalation(
        self, application_id: str, resolution: EscalationResolution
    ) -> dict[str, Any]:
        """Resume a paused run with a reviewer's decision.

        Raises `ApplicationNotFoundError` if the id is unknown, or
        `ApplicationNotAwaitingEscalationError` if the run exists but isn't
        currently paused at `human_escalation` (e.g. it already completed,
        or was never routed to escalation).
        """
        config = self._config(application_id)
        snapshot = self._graph.get_state(config)
        if not snapshot.values:
            raise ApplicationNotFoundError(application_id)
        if not snapshot.next:
            raise ApplicationNotAwaitingEscalationError(application_id)

        logger.info(
            "escalation_resolving",
            extra={"application_id": application_id, "reviewer_id": resolution.reviewer_id},
        )
        self._graph.invoke(Command(resume=resolution.model_dump(mode="json")), config=config)
        return self.get_application(application_id)

    def get_rationale(self, application_id: str) -> dict[str, Any]:
        """Return just the decision outcome + rationale + adverse-action reasons.

        Raises `DecisionNotAvailableError` if the run hasn't reached
        `decision_output` yet (e.g. still awaiting escalation).
        """
        summary = self.get_application(application_id)
        decision = summary.get("decision")
        if decision is None:
            raise DecisionNotAvailableError(application_id)
        return {
            "application_id": application_id,
            "outcome": decision["outcome"],
            "rationale": decision["rationale"],
            "adverse_action_reasons": decision["adverse_action_reasons"],
            "decided_by": decision["decided_by"],
        }
