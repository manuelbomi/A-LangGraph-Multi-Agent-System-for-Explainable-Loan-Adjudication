"""Domain layer: framework-agnostic data models and business rules.

Nothing in this package imports LangGraph, FastAPI, or any LLM SDK. It
defines the vocabulary the rest of the system speaks (LoanApplication,
ExtractedFinancials, RiskAssessment, PolicyResult, Decision) and the
synthetic lending policy rules that operate purely on that vocabulary. This
separation lets `service/` and `api/` change frameworks without touching
business logic, and lets `tests/` exercise policy logic without spinning up
a graph or a server.
"""
