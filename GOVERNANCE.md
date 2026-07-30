# Governance & Guardrails

This document describes the governance patterns implemented in this repo.
It is a **portfolio/demonstration project** illustrating engineering
patterns relevant to regulated-industry AI systems -- it is **not** a
compliance certification, does not implement any specific named regulation
(e.g. ECOA, Regulation B, FCRA, GDPR), and should not be represented as
such. All applicant data is synthetic; the example bank, "Northbridge
Financial Group," is entirely fictional.

## 1. Decisions are made by code, explained by an LLM

The single most important governance decision in this system: **the LLM
never decides approve/deny**. `domain/decisioning.py::determine_outcome`
is a deterministic function of risk tier, policy flags, and (if escalated)
a human reviewer's explicit decision. The chat model
(`explainability_agent`) is invoked strictly *after* that outcome is fixed,
and its only job is to render already-computed facts as readable prose.

This means:
- The approve/deny decision is 100% reproducible and unit-testable without
  mocking an LLM (see `tests/test_risk_and_decisioning.py`).
- An LLM hallucination in the rationale text cannot flip a decision -- at
  worst it produces an awkwardly-worded but factually-grounded rationale
  attached to a decision the LLM had no part in making.

## 2. Permitted-factor allow-listing

`domain/policy.py` defines `PERMITTED_FACTORS`: the exhaustive set of
factors `explainability_agent` is allowed to cite (income, DTI, credit
band, requested-amount-to-income ratio, employment status, delinquency
history). Every `PolicyResult` carries this list forward, and the
rationale-generation prompt is explicitly instructed to cite only factors
present in it.

Separately, `EXCLUDED_FACTORS` documents protected-class-adjacent factors
(age, race, gender, zip code, marital status, national origin) as
*structurally* excluded -- `domain/models.py::LoanApplication` never
collects them in the first place. The safest way to guarantee a factor
never influences a decision is to never capture it.

## 3. Explainability / adverse-action rationale

Every decision -- approved or denied -- produces a `Decision.rationale`
(plain-English, naming the driving factors) and, for denials,
`Decision.adverse_action_reasons` (a list of specific, factor-level
reasons). This mirrors the real-world practice of giving applicants
concrete reasons for an adverse credit decision. See
`GET /applications/{id}/rationale`.

## 4. Human-in-the-loop escalation

Applications with a `borderline` risk tier, or that trip a BLOCK-severity
policy flag (e.g. unreliably-extracted financials, an over-leveraged
request), are routed to `human_escalation` instead of being decided
automatically. The graph run **durably pauses** there via LangGraph's
`interrupt()`/checkpoint mechanism -- a reviewer's decision, submitted via
`POST /applications/{id}/escalation-resolve`, is authoritative once a case
has been escalated (see `domain/decisioning.py::determine_outcome`).

## 5. Audit logging

Every node emits a structured audit event via
`infrastructure/audit_log.py::record_audit_event` -- one entry per
decision-relevant action (document parsed, credit report retrieved, policy
evaluated, escalated, resolved, rationale generated, decision finalized).
The full audit trail for a run is retrievable via
`GET /applications/{id}` (`audit_trail` field) and is also emitted to the
structured JSON application log, tagged with the run's correlation id.

## 6. PII redaction

`infrastructure/pii_redaction.py` scrubs common PII patterns (emails,
phone numbers, SSNs, card-like numbers) and masks known-sensitive dict keys
(`applicant_full_name`, `employer_name`, `reviewer_id`, etc.) before any
structured payload is written to the audit log. This is a demo-grade,
pattern-based implementation -- see the module docstring for what a
production deployment would add (vendor DLP, NER-based detection).

## 7. Prompt-injection guardrail

Applicant-submitted document text is untrusted input to an LLM prompt.
`infrastructure/prompt_guard.py::sanitize_untrusted_text` strips common
injection phrasing (`"ignore previous instructions"`, fake `SYSTEM:` role
markers, etc.) before that text reaches `document_extraction_agent`'s
prompt, and logs a security-relevant warning when it fires. See the
module docstring and README "Security" section for what this does and
does not defend against.

## 8. Observability as a governance tool

Every node wraps its execution in a tracing span
(`infrastructure/tracing.py`), accumulated in `trace_spans` and returned
via `GET /applications/{id}`. Beyond debugging, this gives a compliance
reviewer a literal, timestamped record of what each agent did, in what
order, and how long it took, for any given application -- see the
README's "Observability" section for how this maps onto a production
OpenTelemetry/Prometheus/Grafana stack.
