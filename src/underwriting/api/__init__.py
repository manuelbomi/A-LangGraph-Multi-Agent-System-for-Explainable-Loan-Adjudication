"""API layer: FastAPI application exposing the underwriting graph over HTTP.

Deliberately thin -- routes validate input via Pydantic (domain models and
`api/schemas.py`), delegate all real work to
`service.application_service.ApplicationService`, and translate domain
exceptions into HTTP status codes. No LangGraph or business-logic imports
belong directly in a route handler.
"""
