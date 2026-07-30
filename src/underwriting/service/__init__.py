"""Service layer: the LangGraph graph definition, node implementations, and
the application-level orchestration that FastAPI routes call into.

This is where domain models (domain/) and infrastructure clients
(infrastructure/) get wired together into the actual multi-agent workflow.
Nothing in `api/` talks to LangGraph directly -- it always goes through
`ApplicationService` in `application_service.py`, keeping the HTTP layer
thin and the orchestration logic independently testable.
"""
