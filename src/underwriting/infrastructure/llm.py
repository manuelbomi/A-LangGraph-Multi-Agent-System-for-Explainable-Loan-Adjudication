"""
Pluggable chat-model factory.

Architectural role
-------------------
Graph nodes that need an "LLM call" (document_extraction_agent,
explainability_agent) depend only on LangChain's `Runnable` interface --
never on a concrete provider SDK. `get_chat_model()` is the single seam
that decides, at process start, which concrete implementation to hand
back:

  * No `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` configured (the default,
    and the path used by every test and the CI pipeline) -> `MockChatModel`,
    a fully offline, deterministic stand-in that still honors the Runnable
    contract (`.invoke`, `.batch`, `.stream` via the BaseChatModel base
    class).
  * A key is configured -> a real `ChatOpenAI` or `ChatAnthropic` instance,
    imported lazily so those SDKs are not a hard dependency of the base
    install (`pip install -e .` works with zero network calls; real
    providers are an opt-in extra: `pip install -e ".[providers]"`).

Because both branches satisfy the same `Runnable[list[BaseMessage],
BaseMessage]` interface, every node function is provider-agnostic: swapping
mock for real is a config change, not a code change.

Determinism strategy for the mock
----------------------------------
Rather than returning a single canned string regardless of input (which
would make it useless for testing denial-vs-approval rationale content),
`MockChatModel` inspects a small "task marker" placed in the system message
by the calling node and applies deterministic, rule-based text generation /
extraction keyed off the structured content of the human message. This
keeps `pytest` fully offline while still letting tests assert that, e.g., a
denial rationale actually names the adverse factor that caused it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from underwriting.config import Settings
from underwriting.infrastructure.resilience import CircuitBreaker, with_retry

logger = logging.getLogger(__name__)

# --- Task markers ---------------------------------------------------------
# Node functions embed one of these strings at the start of the system
# message to tell the mock model which deterministic behavior to run. Real
# providers ignore the marker entirely -- it's just plain text to them, part
# of a normal system prompt instructing the model what to do.
EXTRACTION_TASK_MARKER = "TASK:DOCUMENT_EXTRACTION"
RATIONALE_TASK_MARKER = "TASK:EXPLAINABILITY_RATIONALE"

# Pay-frequency -> multiplier to normalize a per-period figure to a monthly
# figure. Used by the mock extraction handler when parsing synthetic
# pay-stub text (see domain fixtures / scripts/demo.py for the text format).
_PAY_FREQUENCY_MONTHLY_MULTIPLIER = {
    "monthly": 1.0,
    "semimonthly": 2.0,
    "semi-monthly": 2.0,
    "biweekly": 26 / 12,
    "bi-weekly": 26 / 12,
    "weekly": 52 / 12,
}

_MONEY_FIELD_PATTERN = r"{label}\s*:\s*\$?\s*([\d,]+(?:\.\d+)?)"


def _parse_money_field(text: str, label: str) -> float | None:
    """Extract a labeled dollar figure like `GROSS_PAY_PER_PERIOD: $3,200.00`."""
    match = re.search(_MONEY_FIELD_PATTERN.format(label=re.escape(label)), text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _parse_text_field(text: str, label: str) -> str | None:
    """Extract a labeled free-text field like `EMPLOYER: Riverstone Logistics LLC`."""
    match = re.search(rf"{re.escape(label)}\s*:\s*(.+)", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _mock_document_extraction(document_text: str) -> str:
    """Deterministically parse the synthetic pay-stub/financial-statement text.

    Returns a JSON string matching `ExtractedFinancials` field names so the
    calling node can `json.loads` it exactly as it would parse a real
    provider's structured-output response. Missing fields lower the
    reported confidence, simulating an LLM's uncertainty on messy input.
    """
    fields_found = 0
    fields_expected = 4

    frequency_raw = (_parse_text_field(document_text, "PAY_FREQUENCY") or "monthly").lower()
    multiplier = _PAY_FREQUENCY_MONTHLY_MULTIPLIER.get(frequency_raw, 1.0)
    if frequency_raw in _PAY_FREQUENCY_MONTHLY_MULTIPLIER:
        fields_found += 1

    gross_per_period = _parse_money_field(document_text, "GROSS_PAY_PER_PERIOD")
    net_per_period = _parse_money_field(document_text, "NET_PAY_PER_PERIOD")
    existing_debt = _parse_money_field(document_text, "EXISTING_MONTHLY_DEBT")
    employer = _parse_text_field(document_text, "EMPLOYER")

    for value in (gross_per_period, net_per_period, existing_debt):
        if value is not None:
            fields_found += 1

    monthly_gross = round((gross_per_period or 0.0) * multiplier, 2)
    monthly_net = round((net_per_period or 0.0) * multiplier, 2)

    confidence = round(0.55 + 0.4 * (fields_found / fields_expected), 2)

    payload = {
        "monthly_gross_income": monthly_gross,
        "monthly_net_income": monthly_net,
        "existing_monthly_debt_payments": existing_debt or 0.0,
        "employer_name": employer,
        "pay_frequency": frequency_raw,
        "extraction_confidence": min(confidence, 0.99),
        "source_excerpts": [
            line.strip() for line in document_text.splitlines() if ":" in line and line.strip()
        ][:6],
    }
    return json.dumps(payload)


def _mock_explainability_rationale(payload_json: str) -> str:
    """Deterministically render a plain-English rationale from a structured payload.

    The calling node passes a JSON object with the decision facts (outcome,
    risk tier, DTI, credit band, permitted factors, policy flags, amounts).
    This function turns those facts into readable prose, always naming only
    the factors present in `permitted_factors_used` -- mirroring how a real
    LLM call would be constrained via prompt instructions, but doing so
    deterministically so tests can assert on exact factor mentions.
    """
    data: dict[str, Any] = json.loads(payload_json)
    outcome = data["decision_outcome"]
    dti = data["debt_to_income_ratio"]
    tier = data["risk_tier"]
    band = data["credit_band"]
    requested = data["requested_amount"]
    approved_amount = data.get("approved_amount")
    permitted = data.get("permitted_factors_used", [])
    flags = data.get("policy_flags", [])
    escalation = data.get("escalation")

    factor_summary = (
        f"a debt-to-income ratio of {dti:.1%}, a '{band}' credit band, "
        f"a '{tier}' overall risk tier, and a requested amount of ${requested:,.0f}"
    )

    if outcome == "approved":
        amount_clause = (
            f" for ${approved_amount:,.0f}" if approved_amount is not None else ""
        )
        lines = [
            f"This application was APPROVED{amount_clause}.",
            f"The decision was driven by {factor_summary}, all of which fall within the "
            f"permitted factor set ({', '.join(permitted)}).",
        ]
    else:
        lines = [
            "This application was DENIED.",
            f"The decision was driven by {factor_summary}, all of which fall within the "
            f"permitted factor set ({', '.join(permitted)}).",
        ]

    blocking_flags = [f for f in flags if f.get("severity") in ("warn", "block")]
    if blocking_flags:
        flag_text = "; ".join(f"{f['code']}: {f['description']}" for f in blocking_flags)
        lines.append(f"Policy review noted the following: {flag_text}.")

    if escalation:
        reviewer = escalation.get("reviewer_id", "a human reviewer")
        notes = escalation.get("notes") or "no additional notes provided"
        approve = escalation.get("approve")
        lines.append(
            f"This case was escalated for human review; {reviewer} "
            f"{'approved' if approve else 'denied'} it after automated risk scoring "
            f"placed it in the borderline band. Reviewer notes: {notes}."
        )

    return " ".join(lines)


class MockChatModel(BaseChatModel):
    """Deterministic, fully offline stand-in for a hosted chat model.

    Implements LangChain's chat-model contract (which is itself a
    `Runnable[LanguageModelInput, BaseMessage]`) so it is a drop-in
    replacement for `ChatOpenAI` / `ChatAnthropic` anywhere in the graph.
    No network access, no API key, and fully deterministic given the same
    input -- which is what lets `pytest` and the demo scripts run offline
    and produce reproducible assertions.
    """

    model_name: str = "mock-deterministic-v1"

    @property
    def _llm_type(self) -> str:  # required by BaseChatModel
        return "mock-deterministic"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Dispatch on the task marker embedded in the leading system message."""
        system_content = ""
        human_content = ""
        for m in messages:
            if m.type == "system" and not system_content:
                system_content = str(m.content)
            elif m.type == "human":
                human_content = str(m.content)

        if EXTRACTION_TASK_MARKER in system_content:
            response_text = _mock_document_extraction(human_content)
        elif RATIONALE_TASK_MARKER in system_content:
            response_text = _mock_explainability_rationale(human_content)
        else:
            # Generic fallback so the mock never hard-fails on an
            # unrecognized prompt -- useful during development of new nodes.
            response_text = json.dumps({"note": "mock model: no task marker recognized"})

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


class ResilientChatModel:
    """Thin resilience wrapper around any Runnable chat model.

    Applies retry-with-backoff-and-jitter and a circuit breaker around
    `.invoke()`, per the requirement that every external call in this
    system carries timeouts/retries/circuit-breaking. Wrapping (rather than
    subclassing each provider) lets the same policy apply uniformly to the
    mock and to any real provider without duplicating logic in three
    places.
    """

    def __init__(self, inner: BaseChatModel, *, breaker_name: str = "llm") -> None:
        self._inner = inner
        self._breaker = CircuitBreaker(name=breaker_name, failure_threshold=3,
                                        recovery_timeout_seconds=15.0)

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
        @with_retry(max_attempts=3, retry_on=(Exception,))
        def _call() -> BaseMessage:
            return self._breaker.call(lambda: self._inner.invoke(messages, **kwargs))

        return _call()


def get_chat_model(settings: Settings) -> ResilientChatModel:
    """Build the process-wide chat model per configured provider, resilience-wrapped.

    This is the ONLY place in the codebase that decides mock-vs-real. Every
    node calls `get_chat_model(settings)` (or receives it via dependency
    injection in `service/application_service.py`) rather than constructing
    a provider client itself.
    """
    if not settings.has_real_llm_provider():
        logger.info("llm_provider_selected", extra={"provider": "mock"})
        return ResilientChatModel(MockChatModel())

    if settings.openai_api_key is not None:
        try:
            from langchain_openai import ChatOpenAI  # lazy import: optional extra
        except ImportError as exc:  # pragma: no cover - exercised only with extras installed
            raise RuntimeError(
                "OPENAI_API_KEY is set but langchain-openai is not installed. "
                'Install it with: pip install -e ".[providers]"'
            ) from exc
        logger.info("llm_provider_selected", extra={"provider": "openai"})
        model = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=30,
            max_retries=0,  # retries are handled uniformly by ResilientChatModel
        )
        return ResilientChatModel(model, breaker_name="llm-openai")

    if settings.anthropic_api_key is not None:
        try:
            from langchain_anthropic import ChatAnthropic  # lazy import: optional extra
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set but langchain-anthropic is not installed. "
                'Install it with: pip install -e ".[providers]"'
            ) from exc
        logger.info("llm_provider_selected", extra={"provider": "anthropic"})
        model = ChatAnthropic(
            model=settings.llm_model_name,
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=30,
            max_retries=0,
        )
        return ResilientChatModel(model, breaker_name="llm-anthropic")

    # Unreachable given has_real_llm_provider(), but keeps mypy/pyright happy.
    return ResilientChatModel(MockChatModel())
