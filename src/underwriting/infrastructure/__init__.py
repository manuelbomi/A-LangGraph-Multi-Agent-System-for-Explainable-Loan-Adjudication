"""Infrastructure layer: everything that talks to the outside world.

LLM providers, the mock credit bureau "tool", the SQLite checkpointer,
structured logging, PII redaction, audit logging, and lightweight tracing
all live here. The domain and service layers depend on abstractions
exposed by this package (e.g. a `Runnable` chat model, a `get_credit_report`
function) but never on a specific vendor SDK directly -- that indirection is
what makes the whole system runnable offline by default.
"""
