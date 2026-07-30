"""Integration tests for the FastAPI HTTP surface, exercised through
`TestClient` end-to-end against the real lifespan (SQLite checkpointer on a
temp file, offline MockChatModel). This is the "at least one
integration-style test using the mock model" requirement, applied to every
public endpoint."""

from __future__ import annotations

from tests.conftest import build_pay_stub_text


def _application_payload(**overrides):
    payload = {
        "applicant_full_name": "Alex B. Approved",
        "requested_amount": 15000.0,
        "loan_purpose": "auto",
        "employment_status": "employed_full_time",
        "stated_annual_income": 62400.0,
        "stated_monthly_debt": 400.0,
        "raw_financial_document_text": build_pay_stub_text(),
    }
    payload.update(overrides)
    return payload


def test_healthz(api_client) -> None:
    response = api_client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz(api_client) -> None:
    response = api_client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_response_carries_correlation_id_header(api_client) -> None:
    response = api_client.get("/healthz")
    assert "x-correlation-id" in response.headers


def test_submit_and_fetch_approved_application(api_client) -> None:
    submit_response = api_client.post("/applications", json=_application_payload())
    assert submit_response.status_code == 201
    body = submit_response.json()
    assert body["status"] == "completed"
    assert body["decision"]["outcome"] == "approved"

    application_id = body["application_id"]
    fetch_response = api_client.get(f"/applications/{application_id}")
    assert fetch_response.status_code == 200
    assert fetch_response.json()["application_id"] == application_id


def test_get_unknown_application_returns_404(api_client) -> None:
    response = api_client.get("/applications/app_does_not_exist")
    assert response.status_code == 404
    assert response.json()["error"] == "application_not_found"


def test_rationale_endpoint(api_client) -> None:
    submit_response = api_client.post("/applications", json=_application_payload())
    application_id = submit_response.json()["application_id"]

    rationale_response = api_client.get(f"/applications/{application_id}/rationale")
    assert rationale_response.status_code == 200
    body = rationale_response.json()
    assert body["outcome"] == "approved"
    assert "APPROVED" in body["rationale"]


def test_rationale_unavailable_while_awaiting_escalation(api_client) -> None:
    borderline_payload = _application_payload(
        stated_monthly_debt=2080.0,
        raw_financial_document_text=build_pay_stub_text(existing_monthly_debt=2080.0),
    )
    submit_response = api_client.post("/applications", json=borderline_payload)
    body = submit_response.json()
    assert body["status"] == "awaiting_escalation"
    application_id = body["application_id"]

    rationale_response = api_client.get(f"/applications/{application_id}/rationale")
    assert rationale_response.status_code == 409
    assert rationale_response.json()["error"] == "decision_not_available"


def test_full_escalation_resolve_flow_via_api(api_client) -> None:
    borderline_payload = _application_payload(
        stated_monthly_debt=2080.0,
        raw_financial_document_text=build_pay_stub_text(existing_monthly_debt=2080.0),
    )
    submit_response = api_client.post("/applications", json=borderline_payload)
    body = submit_response.json()
    assert body["status"] == "awaiting_escalation"
    application_id = body["application_id"]

    resolve_response = api_client.post(
        f"/applications/{application_id}/escalation-resolve",
        json={"reviewer_id": "jdoe123", "approve": True, "notes": "Verified manually."},
    )
    assert resolve_response.status_code == 200
    resolved_body = resolve_response.json()
    assert resolved_body["status"] == "completed"
    assert resolved_body["decision"]["outcome"] == "approved"
    assert resolved_body["decision"]["decided_by"] == "human_reviewer:jdoe123"


def test_resolving_non_escalated_application_returns_409(api_client) -> None:
    submit_response = api_client.post("/applications", json=_application_payload())
    application_id = submit_response.json()["application_id"]

    resolve_response = api_client.post(
        f"/applications/{application_id}/escalation-resolve",
        json={"reviewer_id": "jdoe123", "approve": True},
    )
    assert resolve_response.status_code == 409
    assert resolve_response.json()["error"] == "not_awaiting_escalation"


def test_invalid_application_payload_returns_422(api_client) -> None:
    bad_payload = _application_payload(requested_amount=-5.0)  # violates gt=0
    response = api_client.post("/applications", json=bad_payload)
    assert response.status_code == 422
