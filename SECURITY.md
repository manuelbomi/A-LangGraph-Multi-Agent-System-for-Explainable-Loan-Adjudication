# Security Policy

This is a portfolio/demonstration project, not a production service with
an active user base. That said, it follows practices appropriate to
software intended as a credible example for a regulated-industry
engineering role.

## Supported Versions

This repo targets a single `main` branch; there is no maintained release
branch matrix. Security-relevant fixes, if any are identified, would land
on `main`.

## Reporting a Vulnerability

If you find a security issue in this repository (e.g. a dependency with a
known CVE, an injection vector, a secret accidentally committed), please
open a GitHub issue on the repository describing the concern. As a
portfolio project with no production deployment or user data, there is no
formal SLA, but reports are welcome and will be triaged.

## Practices Followed in This Repo

- **No secrets in source control.** `.env.example` ships placeholder-only
  values; real secrets are only ever read from environment variables at
  runtime via `pydantic-settings` (`src/underwriting/config.py`), typed as
  `SecretStr` so they never render in logs, `repr()`, or error messages.
  `.gitignore` excludes `.env` and any `*.sqlite` checkpoint data.
- **Pinned dependencies.** `pyproject.toml` pins exact versions for every
  direct dependency (no `>=` / wildcard ranges) so builds are reproducible
  and a supply-chain issue in a transitive dependency doesn't silently
  change what gets installed.
- **No LLM provider required by default.** The application runs fully
  offline against a deterministic mock model unless a real provider API
  key is explicitly configured (see `infrastructure/llm.py`) -- there is
  no code path where a key is required just to run the demo or test suite.
- **Prompt-injection mitigation.** Applicant-submitted text is treated as
  untrusted input and passed through
  `infrastructure/prompt_guard.py::sanitize_untrusted_text` before being
  interpolated into any LLM prompt. This is a pattern-based, demo-grade
  mitigation -- see that module's docstring for its limits and what a
  production system would add on top (instruction-hierarchy-aware models,
  dual-LLM "quarantine" patterns, strict output schemas).
- **Least-privilege container.** The Docker image (`Dockerfile`) runs as a
  dedicated non-root user, drops all Linux capabilities, and disables
  privilege escalation (see `deploy/k8s/deployment.yaml`'s
  `securityContext`).
- **Resilience against dependency failures.** Every external call (LLM
  provider, mock credit bureau) is wrapped in retry-with-backoff-and-jitter
  and a circuit breaker (`infrastructure/resilience.py`), reducing the
  blast radius of a flaky or unavailable dependency and avoiding
  thundering-herd retry storms.
- **PII-aware logging.** See `GOVERNANCE.md` section 6 -- structured logs
  and audit entries are redacted before being written.
- **Input validation at every boundary.** Every HTTP request body and every
  inter-node data handoff is validated through a Pydantic model (see
  `domain/models.py`), rejecting malformed input before it reaches
  business logic.

## Known Limitations (by design, for a demo)

- The mock credit bureau and mock LLM are intentionally simplistic --
  this repo optimizes for reviewability and offline reproducibility, not
  for defending against an adversarial red team.
- The SQLite checkpointer is a single-writer store; see the README's "Key
  Design Decisions" for the production swap to a Postgres-backed
  checkpointer for multi-replica deployments.
