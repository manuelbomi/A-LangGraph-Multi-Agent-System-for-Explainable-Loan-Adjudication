"""
Durable checkpointing for the LangGraph run.

Architectural role
-------------------
The `human_escalation` node uses LangGraph's `interrupt()` mechanism to
pause a run mid-graph and wait for an external decision. For that pause to
be *durable* -- i.e. survive an API server restart between the escalation
firing and a reviewer resolving it, which is realistic (reviews can take
hours or days) -- the graph must be compiled with a checkpointer that
persists state to disk, not just to an in-memory dict.

This module owns the single SQLite-backed `SqliteSaver` connection used by
the whole process. It is opened once at FastAPI startup (see
`api/main.py`'s lifespan) and closed at shutdown, rather than opened per
request, both for performance and because `SqliteSaver` wraps a single
`sqlite3.Connection` (with `check_same_thread=False`) that is safe to share
across the app's request-handling threads.

Production note: SQLite is the right default for a single-process demo. A
horizontally-scaled deployment (see deploy/k8s/) would swap this for
LangGraph's Postgres checkpointer against a shared database so any replica
can resume any paused run -- see the README's "Key Design Decisions"
section for that tradeoff.
"""

from __future__ import annotations

import logging
import os
from contextlib import ExitStack

from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)

# Module-level handle to the open checkpointer + the ExitStack that owns its
# underlying context manager, so `close_checkpointer` can cleanly tear it
# down. A module-level singleton (rather than FastAPI `app.state` only) also
# lets test fixtures reach in and reset it between tests.
_stack: ExitStack | None = None
_saver: SqliteSaver | None = None


def open_checkpointer(db_path: str) -> SqliteSaver:
    """Open (or return the already-open) SqliteSaver for `db_path`.

    Creates the parent directory if needed (a fresh clone of this repo has
    no `data/` directory yet) and runs `saver.setup()` to create the
    checkpoint tables on first use.
    """
    global _stack, _saver
    if _saver is not None:
        return _saver

    if db_path != ":memory:":
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    _stack = ExitStack()
    saver = _stack.enter_context(SqliteSaver.from_conn_string(db_path))
    saver.setup()
    _saver = saver
    logger.info("checkpointer_opened", extra={"db_path": db_path})
    return saver


def close_checkpointer() -> None:
    """Close the checkpointer's underlying connection, if open.

    Safe to call multiple times (idempotent) -- used in both FastAPI's
    shutdown lifespan and test teardown.
    """
    global _stack, _saver
    if _stack is not None:
        _stack.close()
    _stack = None
    _saver = None
    logger.info("checkpointer_closed")
