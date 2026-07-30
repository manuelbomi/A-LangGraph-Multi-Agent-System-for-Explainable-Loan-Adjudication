"""
Shared pytest fixtures.

The most important fixture here is `_no_ambient_provider_keys`: it is
`autouse=True` so EVERY test in the suite runs with `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY` forcibly unset, regardless of what's present in the
developer's (or CI runner's) actual shell environment. Without this, a
developer machine that happens to export `OPENAI_API_KEY` for unrelated
tools would cause `underwriting.infrastructure.llm.get_chat_model` to try
to select a real provider during tests -- breaking the "runs fully offline,
no paid API keys required" guarantee this repo is built around.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from underwriting.config import Settings
from underwriting.infrastructure.checkpointer import close_checkpointer, open_checkpointer
from underwriting.infrastructure.llm import get_chat_model
from underwriting.service.application_service import ApplicationService


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the offline MockChatModel path for every test (see module docstring)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def settings() -> Settings:
    """Settings pinned to an in-memory checkpoint DB and no real LLM provider."""
    return Settings(
        checkpoint_db_path=":memory:",
        openai_api_key=None,
        anthropic_api_key=None,
    )


@pytest.fixture
def service(settings: Settings):
    """A fully wired ApplicationService backed by an in-memory checkpointer.

    Closes the module-level checkpointer singleton on teardown so the next
    test (which requests a fresh `service` fixture) gets a clean SQLite
    connection rather than reusing this test's.
    """
    chat_model = get_chat_model(settings)
    checkpointer = open_checkpointer(":memory:")
    svc = ApplicationService(settings=settings, chat_model=chat_model, checkpointer=checkpointer)
    yield svc
    close_checkpointer()


@pytest.fixture
def api_client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A FastAPI TestClient with a real (temp-file) SQLite checkpointer.

    Drives the full `lifespan` startup/shutdown path (unlike the `service`
    fixture, which wires dependencies directly), so this is what exercises
    the actual HTTP surface, middleware, and exception handlers.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(db_path))

    from underwriting.config import get_settings

    get_settings.cache_clear()
    close_checkpointer()

    from underwriting.api.main import app

    with TestClient(app) as client:
        yield client

    close_checkpointer()
    get_settings.cache_clear()


def build_pay_stub_text(
    *,
    employer: str = "Riverstone Logistics LLC",
    pay_frequency: str = "biweekly",
    gross_pay_per_period: float = 2400.0,
    net_pay_per_period: float = 1850.0,
    existing_monthly_debt: float = 400.0,
) -> str:
    """Build synthetic pay-stub TEXT in the labeled format the mock model parses.

    Not a real file -- this is the "document" the demo passes as
    `raw_financial_document_text`. All values are synthetic and reference
    the fictional Northbridge Financial Group brand only in the header.
    """
    return (
        "NORTHBRIDGE FINANCIAL GROUP - SYNTHETIC PAY STATEMENT (DEMO DATA ONLY)\n"
        f"EMPLOYER: {employer}\n"
        f"PAY_FREQUENCY: {pay_frequency}\n"
        f"GROSS_PAY_PER_PERIOD: ${gross_pay_per_period:.2f}\n"
        f"NET_PAY_PER_PERIOD: ${net_pay_per_period:.2f}\n"
        f"EXISTING_MONTHLY_DEBT: ${existing_monthly_debt:.2f}\n"
    )


@pytest.fixture
def make_application() -> Callable[..., dict[str, Any]]:
    """Factory fixture returning a JSON-ready LoanApplication payload dict.

    Returns a plain dict (rather than a `LoanApplication` instance) so it
    can be used interchangeably as an API request body or passed through
    `LoanApplication.model_validate(...)` in service-layer tests.
    """

    def _make(**overrides: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "applicant_full_name": "Alex B. Approved",
            "requested_amount": 15000.0,
            "loan_purpose": "auto",
            "employment_status": "employed_full_time",
            "stated_annual_income": 62400.0,
            "stated_monthly_debt": 400.0,
            "raw_financial_document_text": build_pay_stub_text(),
        }
        defaults.update(overrides)
        return defaults

    return _make
