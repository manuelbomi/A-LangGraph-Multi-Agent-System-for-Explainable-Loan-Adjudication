"""
FastAPI application entry point.

Architectural role
-------------------
This module wires together configuration, logging, the LLM factory, the
SQLite checkpointer, and `ApplicationService` at process startup (via
FastAPI's `lifespan` context manager), then exposes a small, thin set of
HTTP routes over that service. Routes do input validation via Pydantic and
translate domain exceptions to HTTP status codes -- they contain no
LangGraph or business logic themselves (see service/application_service.py
for that).

Run locally with:  uvicorn underwriting.api.main:app --reload
(or `make run`, or `docker compose up` -- see README "Getting Started").
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from underwriting.api.schemas import (
    ApplicationStateResponse,
    ErrorResponse,
    HealthResponse,
    RationaleResponse,
)
from underwriting.config import get_settings
from underwriting.domain.errors import (
    ApplicationNotAwaitingEscalationError,
    ApplicationNotFoundError,
    DecisionNotAvailableError,
)
from underwriting.domain.models import EscalationResolution, LoanApplication
from underwriting.infrastructure.checkpointer import close_checkpointer, open_checkpointer
from underwriting.infrastructure.llm import get_chat_model
from underwriting.infrastructure.logging_config import configure_logging, correlation_id_var
from underwriting.service.application_service import ApplicationService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown: configure logging, open the checkpointer, build the service."""
    settings = get_settings()
    configure_logging(service_name=settings.service_name, log_level=settings.log_level)

    checkpointer = open_checkpointer(settings.checkpoint_db_path)
    chat_model = get_chat_model(settings)
    service = ApplicationService(settings=settings, chat_model=chat_model, checkpointer=checkpointer)

    app.state.settings = settings
    app.state.service = service
    logger.info("application_startup_complete", extra={"service_name": settings.service_name})

    yield

    close_checkpointer()
    logger.info("application_shutdown_complete")


app = FastAPI(
    title="Credit Underwriting Decision Graph",
    description=(
        "A LangGraph multi-agent system for explainable, human-in-the-loop loan "
        "adjudication. Demo/portfolio project using synthetic data for a fictional "
        "bank, Northbridge Financial Group."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Attach a per-request correlation id to context (for logging) and to the response.

    Accepts an inbound `X-Correlation-Id` header so a caller (or an
    upstream gateway) can propagate its own id; otherwise generates one.
    Every log line emitted while handling this request carries the id via
    `infrastructure.logging_config.CorrelationIdFilter`.
    """
    incoming = request.headers.get("x-correlation-id")
    correlation_id = incoming or f"req_{uuid.uuid4().hex[:12]}"
    token = correlation_id_var.set(correlation_id)
    try:
        response = await call_next(request)
    finally:
        correlation_id_var.reset(token)
    response.headers["X-Correlation-Id"] = correlation_id
    return response


def _service(request: Request) -> ApplicationService:
    """Fetch the process-wide ApplicationService from app state."""
    service: ApplicationService = request.app.state.service
    return service


# --- Exception handlers: translate domain errors into HTTP responses -------


@app.exception_handler(ApplicationNotFoundError)
async def _not_found_handler(request: Request, exc: ApplicationNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="application_not_found", detail=f"No application found with id '{exc}'."
        ).model_dump(),
    )


@app.exception_handler(ApplicationNotAwaitingEscalationError)
async def _not_awaiting_escalation_handler(
    request: Request, exc: ApplicationNotAwaitingEscalationError
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content=ErrorResponse(
            error="not_awaiting_escalation",
            detail=f"Application '{exc}' is not currently paused awaiting escalation review.",
        ).model_dump(),
    )


@app.exception_handler(DecisionNotAvailableError)
async def _decision_not_available_handler(
    request: Request, exc: DecisionNotAvailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content=ErrorResponse(
            error="decision_not_available",
            detail=f"Application '{exc}' has not reached a final decision yet.",
        ).model_dump(),
    )


# --- Health / readiness ------------------------------------------------------


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz() -> HealthResponse:
    """Liveness probe: process is up and able to answer HTTP requests."""
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=HealthResponse, tags=["health"])
async def readyz(request: Request) -> HealthResponse:
    """Readiness probe: dependencies (checkpointer, chat model, service) are initialized."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="service not yet initialized")
    return HealthResponse(status="ok", detail="checkpointer and chat model initialized")


# --- Core application endpoints ----------------------------------------------


@app.post("/applications", response_model=ApplicationStateResponse, status_code=201, tags=["applications"])
async def submit_application(
    application: LoanApplication, request: Request
) -> ApplicationStateResponse:
    """Start a new underwriting graph run for a synthetic loan application.

    Runs the graph through to completion, or until it durably pauses at
    `human_escalation` -- either way, the response reflects the run's
    current state.
    """
    service = _service(request)
    result = service.start_application(application)
    return ApplicationStateResponse.model_validate(result)


@app.get("/applications/{application_id}", response_model=ApplicationStateResponse, tags=["applications"])
async def get_application(application_id: str, request: Request) -> ApplicationStateResponse:
    """Fetch the current state, decision (if any), and audit/trace history."""
    service = _service(request)
    result = service.get_application(application_id)
    return ApplicationStateResponse.model_validate(result)


@app.post(
    "/applications/{application_id}/escalation-resolve",
    response_model=ApplicationStateResponse,
    tags=["applications"],
)
async def resolve_escalation(
    application_id: str, resolution: EscalationResolution, request: Request
) -> ApplicationStateResponse:
    """Reviewer endpoint: resume a paused run with an approve/deny decision.

    404 if the application doesn't exist; 409 if it exists but isn't
    currently paused awaiting escalation (already decided, or never
    escalated).
    """
    service = _service(request)
    result = service.resolve_escalation(application_id, resolution)
    return ApplicationStateResponse.model_validate(result)


@app.get(
    "/applications/{application_id}/rationale", response_model=RationaleResponse, tags=["applications"]
)
async def get_rationale(application_id: str, request: Request) -> RationaleResponse:
    """Fetch just the plain-English decision rationale and adverse-action reasons."""
    service = _service(request)
    result = service.get_rationale(application_id)
    return RationaleResponse.model_validate(result)
