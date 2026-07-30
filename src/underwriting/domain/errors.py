"""Domain-level exceptions.

Kept separate from `models.py` and `policy.py` so both the API layer
(`api/main.py`, mapping these to HTTP status codes) and the service layer
(`service/application_service.py`, raising them) can import a small,
stable set of error types without a circular import.
"""

from __future__ import annotations


class ApplicationNotFoundError(Exception):
    """Raised when an `application_id` has no corresponding graph run."""


class ApplicationNotAwaitingEscalationError(Exception):
    """Raised when a resolve-escalation call targets a run that isn't paused."""


class DecisionNotAvailableError(Exception):
    """Raised when a rationale/decision is requested before the run has completed."""
