"""Unit tests for the pluggable chat-model factory (infrastructure/llm.py)."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from underwriting.config import Settings
from underwriting.infrastructure.llm import (
    EXTRACTION_TASK_MARKER,
    RATIONALE_TASK_MARKER,
    MockChatModel,
    get_chat_model,
)


def test_get_chat_model_defaults_to_mock_when_no_keys_configured() -> None:
    settings = Settings(checkpoint_db_path=":memory:", openai_api_key=None, anthropic_api_key=None)
    chat_model = get_chat_model(settings)
    assert isinstance(chat_model._inner, MockChatModel)


def test_get_chat_model_raises_helpfully_if_openai_extra_missing() -> None:
    settings = Settings(
        checkpoint_db_path=":memory:", openai_api_key="sk-fake-not-real", anthropic_api_key=None
    )
    with pytest.raises(RuntimeError, match="langchain-openai is not installed"):
        get_chat_model(settings)


def test_mock_model_extraction_is_deterministic_and_json_parseable() -> None:
    model = MockChatModel()
    system = SystemMessage(content=f"{EXTRACTION_TASK_MARKER}\nExtract fields.")
    human = HumanMessage(
        content=(
            "EMPLOYER: Acme Corp\n"
            "PAY_FREQUENCY: monthly\n"
            "GROSS_PAY_PER_PERIOD: $5000.00\n"
            "NET_PAY_PER_PERIOD: $3900.00\n"
            "EXISTING_MONTHLY_DEBT: $600.00\n"
        )
    )
    result1 = model.invoke([system, human])
    result2 = model.invoke([system, human])
    assert result1.content == result2.content  # deterministic

    parsed = json.loads(result1.content)
    assert parsed["monthly_gross_income"] == 5000.0
    assert parsed["existing_monthly_debt_payments"] == 600.0
    assert parsed["employer_name"] == "Acme Corp"
    assert 0.0 <= parsed["extraction_confidence"] <= 1.0


def test_mock_model_rationale_generation_cites_outcome() -> None:
    model = MockChatModel()
    system = SystemMessage(content=f"{RATIONALE_TASK_MARKER}\nWrite a rationale.")
    payload = {
        "decision_outcome": "denied",
        "debt_to_income_ratio": 0.6,
        "risk_tier": "high",
        "credit_band": "poor",
        "requested_amount": 10000.0,
        "approved_amount": None,
        "permitted_factors_used": ["debt_to_income_ratio", "credit_band"],
        "policy_flags": [],
        "escalation": None,
    }
    human = HumanMessage(content=json.dumps(payload))
    result = model.invoke([system, human])
    assert "DENIED" in result.content
    assert "60.0%" in result.content
