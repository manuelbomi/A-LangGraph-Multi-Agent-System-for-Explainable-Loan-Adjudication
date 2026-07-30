"""
LangGraph node implementations for the underwriting graph.

Architectural role
-------------------
Each `make_*_node(...)` function is a small factory that closes over the
dependencies a node needs (a chat model, `Settings`) and returns the actual
node callable LangGraph invokes with `(state) -> partial_state_update`.
Using factories rather than module-level functions is the dependency
injection seam: `service/graph.py` decides which chat model (mock or real)
and which settings to bind, and tests can bind a `MockChatModel` explicitly
without any monkeypatching.

Every node follows the same shape:

  1. Open a tracing span (`infrastructure/tracing.py`) named after the node.
  2. Validate/deserialize the relevant slice of state into Pydantic domain
     models (input validation at every internal boundary, not just the
     HTTP edge).
  3. Do the node's actual work (deterministic domain logic and/or a chat
     model call).
  4. Record an audit event (`infrastructure/audit_log.py`).
  5. Return a partial state update -- LangGraph merges this into the
     overall state (list-typed fields like `audit_trail`/`trace_spans` are
     concatenated per the `operator.add` reducer in `service/state.py`).

`human_escalation` is the one exception to "no side-pausing": it calls
LangGraph's `interrupt()` mid-function, which durably pauses the run (see
infrastructure/checkpointer.py) until an external `Command(resume=...)` is
sent via `ApplicationService.resolve_escalation`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from underwriting.config import Settings
from underwriting.domain.decisioning import determine_outcome
from underwriting.domain.models import (
    Decision,
    EscalationResolution,
    ExtractedFinancials,
    LoanApplication,
    PolicyResult,
    RiskAssessment,
    RiskTier,
)
from underwriting.domain.policy import evaluate_policy
from underwriting.domain.risk import (
    build_rationale_factors,
    classify_risk_tier,
    compute_debt_to_income_ratio,
)
from underwriting.infrastructure.audit_log import record_audit_event
from underwriting.infrastructure.credit_bureau import get_credit_report
from underwriting.infrastructure.llm import (
    EXTRACTION_TASK_MARKER,
    RATIONALE_TASK_MARKER,
    ResilientChatModel,
)
from underwriting.infrastructure.prompt_guard import sanitize_untrusted_text
from underwriting.infrastructure.tracing import start_span
from underwriting.service.state import UnderwritingState

logger = logging.getLogger(__name__)


def make_intake_node() -> Any:
    """intake: validate/normalize the inbound application payload.

    The application arrives in `state["application"]` already once-validated
    by FastAPI's request model, but we re-validate here deliberately: graph
    nodes should never trust that upstream validation is still intact by
    the time a run resumes from a checkpoint (potentially minutes or days
    later, potentially after a code deploy that tightened validation
    rules).
    """

    def node(state: UnderwritingState) -> dict[str, Any]:
        application_id = state["application_id"]
        correlation_id = state["correlation_id"]
        with start_span("intake", application_id=application_id) as span:
            application = LoanApplication.model_validate(state["application"])
            audit = record_audit_event(
                application_id=application_id,
                correlation_id=correlation_id,
                actor="intake",
                action="application_received",
                details={
                    "loan_purpose": application.loan_purpose.value,
                    "requested_amount": application.requested_amount,
                    "employment_status": application.employment_status.value,
                },
            )
            span["attributes"]["loan_purpose"] = application.loan_purpose.value

        return {
            "application": application.model_dump(mode="json"),
            "status": "in_progress",
            "audit_trail": [audit],
            "trace_spans": [span],
        }

    return node


def make_document_extraction_node(chat_model: ResilientChatModel) -> Any:
    """document_extraction_agent: turn raw pay-stub/financial-statement text
    into structured `ExtractedFinancials`, via a simulated LLM call.

    The chat model is invoked with a system message carrying
    `EXTRACTION_TASK_MARKER` (see infrastructure/llm.py) so the offline
    `MockChatModel` can deterministically parse the synthetic document
    format; a real provider would follow the same instruction as a normal
    structured-extraction prompt.
    """

    def node(state: UnderwritingState) -> dict[str, Any]:
        application_id = state["application_id"]
        correlation_id = state["correlation_id"]
        with start_span("document_extraction_agent", application_id=application_id) as span:
            application = LoanApplication.model_validate(state["application"])

            system = SystemMessage(
                content=(
                    f"{EXTRACTION_TASK_MARKER}\n"
                    "You are a financial document extraction assistant for a synthetic "
                    "bank demo (Northbridge Financial Group, fictional). Extract "
                    "monthly_gross_income, monthly_net_income, "
                    "existing_monthly_debt_payments, employer_name, and pay_frequency "
                    "from the applicant's submitted document text below. Respond ONLY "
                    "with a single JSON object matching those field names."
                )
            )
            # Applicant-submitted text is untrusted input to the LLM prompt --
            # sanitize it against common prompt-injection phrasing first.
            safe_document_text = sanitize_untrusted_text(
                application.raw_financial_document_text, source="raw_financial_document_text"
            )
            human = HumanMessage(content=safe_document_text)
            ai_message = chat_model.invoke([system, human])
            extracted = ExtractedFinancials.model_validate_json(str(ai_message.content))

            audit = record_audit_event(
                application_id=application_id,
                correlation_id=correlation_id,
                actor="document_extraction_agent",
                action="financials_extracted",
                details={
                    "extraction_confidence": extracted.extraction_confidence,
                    "monthly_gross_income": extracted.monthly_gross_income,
                    "existing_monthly_debt_payments": extracted.existing_monthly_debt_payments,
                },
            )
            span["attributes"]["extraction_confidence"] = extracted.extraction_confidence

        return {
            "extracted_financials": extracted.model_dump(mode="json"),
            "audit_trail": [audit],
            "trace_spans": [span],
        }

    return node


def make_risk_scoring_node(settings: Settings) -> Any:
    """risk_scoring_agent: compute DTI + credit-band risk tier via the mock bureau.

    Calls `infrastructure.credit_bureau.get_credit_report` (itself wrapped
    in retry/circuit-breaker resilience) as the "external tool" call, then
    applies the deterministic classification rules in `domain/risk.py`.
    """

    def node(state: UnderwritingState) -> dict[str, Any]:
        application_id = state["application_id"]
        correlation_id = state["correlation_id"]
        with start_span("risk_scoring_agent", application_id=application_id) as span:
            application = LoanApplication.model_validate(state["application"])
            extracted = ExtractedFinancials.model_validate(state["extracted_financials"])

            credit_report = get_credit_report(application.applicant_full_name)

            # Prefer the document-extracted income (what the applicant can
            # actually evidence) over the self-reported figure; fall back to
            # the stated annual income only if extraction yielded nothing.
            monthly_income = extracted.monthly_gross_income or (
                application.stated_annual_income / 12.0
            )
            dti = compute_debt_to_income_ratio(
                monthly_income=monthly_income,
                monthly_debt_payments=extracted.existing_monthly_debt_payments,
            )
            risk_tier = classify_risk_tier(
                dti=dti, credit_band=credit_report.credit_band, settings=settings
            )

            risk = RiskAssessment(
                debt_to_income_ratio=dti,
                credit_report=credit_report,
                risk_tier=risk_tier,
                requires_human_escalation=(risk_tier == RiskTier.BORDERLINE),
                rationale_factors=build_rationale_factors(
                    dti=dti,
                    credit_band=credit_report.credit_band,
                    credit_score=credit_report.credit_score,
                ),
            )

            audit = record_audit_event(
                application_id=application_id,
                correlation_id=correlation_id,
                actor="risk_scoring_agent",
                action="risk_scored",
                details={
                    "debt_to_income_ratio": dti,
                    "risk_tier": risk_tier.value,
                    "credit_band": credit_report.credit_band.value,
                    "credit_score": credit_report.credit_score,
                },
            )
            span["attributes"].update({"risk_tier": risk_tier.value, "dti": dti})

        return {
            "risk_assessment": risk.model_dump(mode="json"),
            "audit_trail": [audit],
            "trace_spans": [span],
        }

    return node


def make_policy_compliance_node() -> Any:
    """policy_compliance_agent: run the synthetic lending policy ruleset.

    Pure deterministic evaluation (`domain/policy.py`) -- no LLM call. This
    node's output (`PolicyResult.passed`, `.flags`) feeds the conditional
    routing edge that decides whether to escalate.
    """

    def node(state: UnderwritingState) -> dict[str, Any]:
        application_id = state["application_id"]
        correlation_id = state["correlation_id"]
        with start_span("policy_compliance_agent", application_id=application_id) as span:
            application = LoanApplication.model_validate(state["application"])
            extracted = ExtractedFinancials.model_validate(state["extracted_financials"])
            risk = RiskAssessment.model_validate(state["risk_assessment"])

            policy_result = evaluate_policy(application, extracted, risk)

            audit = record_audit_event(
                application_id=application_id,
                correlation_id=correlation_id,
                actor="policy_compliance_agent",
                action="policy_evaluated",
                details={
                    "passed": policy_result.passed,
                    "flag_codes": [f.code for f in policy_result.flags],
                },
            )
            span["attributes"]["passed"] = policy_result.passed
            span["attributes"]["flag_count"] = len(policy_result.flags)

        return {
            "policy_result": policy_result.model_dump(mode="json"),
            "audit_trail": [audit],
            "trace_spans": [span],
        }

    return node


def route_after_policy(state: UnderwritingState) -> str:
    """Conditional edge: borderline risk OR a failed policy check -> escalate.

    A BLOCK-severity policy flag (e.g. unreliable document extraction, an
    over-leveraged request) means the system cannot responsibly decide
    automatically even if the risk tier itself looked fine -- so escalation
    is an OR of both signals, not just the risk tier.
    """
    risk_assessment = state["risk_assessment"]
    policy_result = state["policy_result"]
    if risk_assessment["requires_human_escalation"] or not policy_result["passed"]:
        return "human_escalation"
    return "explainability_agent"


def make_human_escalation_node() -> Any:
    """human_escalation: durably pause the run for a human reviewer's decision.

    This is the key human-in-the-loop mechanism: `interrupt(...)` suspends
    graph execution at this exact point. Because the graph is compiled with
    a `SqliteSaver` checkpointer (infrastructure/checkpointer.py), the pause
    survives an API process restart -- resuming is a fresh `graph.invoke(
    Command(resume=...))` call against the persisted thread, not a
    continuation of the same Python call stack.

    Everything before `interrupt()` runs once, when the run first reaches
    this node. Everything after it runs once, when a reviewer resolves the
    escalation -- `resolution_input` is exactly the payload passed to
    `Command(resume=...)`.
    """

    def node(state: UnderwritingState) -> dict[str, Any]:
        application_id = state["application_id"]
        correlation_id = state["correlation_id"]
        risk = state["risk_assessment"]
        policy = state["policy_result"]

        with start_span("human_escalation_request", application_id=application_id) as request_span:
            reasons = []
            if risk["requires_human_escalation"]:
                reasons.append(
                    f"risk tier classified as '{risk['risk_tier']}' (borderline band)"
                )
            if not policy["passed"]:
                blocking = [f["code"] for f in policy["flags"] if f["severity"] == "block"]
                reasons.append(f"policy block flag(s): {', '.join(blocking)}")

            escalation_payload = {
                "reason": "; ".join(reasons) or "escalation routing triggered",
                "risk_summary": (
                    f"debt_to_income_ratio={risk['debt_to_income_ratio']:.1%}, "
                    f"credit_band={risk['credit_report']['credit_band']}, "
                    f"risk_tier={risk['risk_tier']}"
                ),
                "application_id": application_id,
            }
            request_audit = record_audit_event(
                application_id=application_id,
                correlation_id=correlation_id,
                actor="human_escalation",
                action="escalation_requested",
                details=escalation_payload,
            )

        # --- Durable pause point ---------------------------------------
        # Execution genuinely stops here until a `Command(resume=...)` call
        # is made against this thread_id (see
        # ApplicationService.resolve_escalation). The value returned by
        # `interrupt()` is whatever was passed to `Command(resume=...)`.
        resolution_input: dict[str, Any] = interrupt(escalation_payload)

        with start_span("human_escalation_resume", application_id=application_id) as resume_span:
            resolution = EscalationResolution.model_validate(resolution_input)
            resume_audit = record_audit_event(
                application_id=application_id,
                correlation_id=correlation_id,
                actor=f"human_reviewer:{resolution.reviewer_id}",
                action="escalation_resolved",
                details={"approve": resolution.approve, "notes": resolution.notes},
            )
            resume_span["attributes"]["approve"] = resolution.approve

        return {
            "escalation_request": escalation_payload,
            "escalation_resolution": resolution.model_dump(mode="json"),
            "status": "escalation_resolved",
            "audit_trail": [request_audit, resume_audit],
            "trace_spans": [request_span, resume_span],
        }

    return node


def make_explainability_node(chat_model: ResilientChatModel) -> Any:
    """explainability_agent: produce a plain-English rationale naming the driving factors.

    Outcome determination itself is deterministic (`domain/decisioning.py`)
    -- the chat model's job is purely to render that already-decided
    outcome and its already-computed factors as readable prose. This
    separation is the crux of this repo's governance story: the LLM never
    decides approve/deny, it only explains a decision made by auditable
    code (see README "Key Design Decisions").
    """

    def node(state: UnderwritingState) -> dict[str, Any]:
        application_id = state["application_id"]
        correlation_id = state["correlation_id"]
        with start_span("explainability_agent", application_id=application_id) as span:
            application = LoanApplication.model_validate(state["application"])
            risk = RiskAssessment.model_validate(state["risk_assessment"])
            policy = PolicyResult.model_validate(state["policy_result"])
            escalation_resolution: EscalationResolution | None = None
            if state.get("escalation_resolution"):
                escalation_resolution = EscalationResolution.model_validate(
                    state["escalation_resolution"]
                )

            outcome, approved_amount, adverse_reasons = determine_outcome(
                application, risk, policy, escalation_resolution
            )

            payload = {
                "decision_outcome": outcome.value,
                "debt_to_income_ratio": risk.debt_to_income_ratio,
                "risk_tier": risk.risk_tier.value,
                "credit_band": risk.credit_report.credit_band.value,
                "requested_amount": application.requested_amount,
                "approved_amount": approved_amount,
                "permitted_factors_used": policy.permitted_factors_used,
                "policy_flags": [f.model_dump(mode="json") for f in policy.flags],
                "escalation": (
                    escalation_resolution.model_dump(mode="json")
                    if escalation_resolution
                    else None
                ),
            }
            system = SystemMessage(
                content=(
                    f"{RATIONALE_TASK_MARKER}\n"
                    "Write a plain-English rationale for this credit decision. Cite ONLY "
                    "the factors listed in permitted_factors_used. Do not invent factors "
                    "not present in the payload. Respond in prose."
                )
            )
            human = HumanMessage(content=json.dumps(payload))
            ai_message = chat_model.invoke([system, human])
            rationale_text = str(ai_message.content)

            decided_by = (
                f"human_reviewer:{escalation_resolution.reviewer_id}"
                if escalation_resolution
                else "explainability_agent"
            )
            decision = Decision(
                outcome=outcome,
                approved_amount=approved_amount,
                rationale=rationale_text,
                adverse_action_reasons=adverse_reasons,
                decided_by=decided_by,
            )

            audit = record_audit_event(
                application_id=application_id,
                correlation_id=correlation_id,
                actor="explainability_agent",
                action="rationale_generated",
                details={"outcome": outcome.value, "decided_by": decided_by},
            )
            span["attributes"]["outcome"] = outcome.value

        return {
            "decision": decision.model_dump(mode="json"),
            "audit_trail": [audit],
            "trace_spans": [span],
        }

    return node


def make_decision_output_node() -> Any:
    """decision_output: finalize and durably persist the decision.

    Persistence itself is handled by the LangGraph checkpointer (every
    superstep is already written to SQLite) -- this node's job is to mark
    the run `completed` and emit the terminal audit event, giving a clean,
    single audit-log line that a compliance reviewer can search for to
    confirm a given application reached a final state.
    """

    def node(state: UnderwritingState) -> dict[str, Any]:
        application_id = state["application_id"]
        correlation_id = state["correlation_id"]
        decision = state["decision"]
        with start_span("decision_output", application_id=application_id) as span:
            audit = record_audit_event(
                application_id=application_id,
                correlation_id=correlation_id,
                actor="decision_output",
                action="decision_finalized",
                details={"outcome": decision["outcome"]},
            )
            span["attributes"]["outcome"] = decision["outcome"]

        return {
            "status": "completed",
            "audit_trail": [audit],
            "trace_spans": [span],
        }

    return node
