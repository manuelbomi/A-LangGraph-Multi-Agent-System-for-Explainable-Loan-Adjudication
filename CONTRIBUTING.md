# Contributing

This is primarily a portfolio project, but it's built to normal
open-source-quality standards and welcomes issues/PRs.

## Development setup

```bash
git clone <this-repo-url>
cd langgraph-multiagent-underwriting
make install     # creates .venv/ and installs the package + dev/test deps
make test         # runs the full pytest suite (fully offline)
make lint         # ruff
make typecheck    # mypy
```

## Before opening a PR

1. `make lint && make typecheck && make test` all pass locally.
2. New behavior has a corresponding test -- see `tests/` for the existing
   pattern (pure `domain/` unit tests, `service`-level integration tests
   using the `service` fixture, HTTP-level tests using the `api_client`
   fixture).
3. Public functions/classes have docstrings explaining their role, per the
   convention already used throughout `src/underwriting/`.
4. No real employer, bank, or person's name anywhere in code, comments,
   fixtures, or commit messages. All example data must reference only the
   fictional "Northbridge Financial Group" brand, clearly synthetic.
5. No secrets, API keys, or `.env` files committed. `.env.example` should
   only ever contain placeholder values.

## Commit style

Concise, present-tense summaries (`add retry policy to credit bureau
client`, not `added` or `adds`). Reference the relevant module/area when it
isn't obvious from context.

## Code style

- `ruff` governs formatting/lint rules (see `pyproject.toml`'s
  `[tool.ruff]` section); run `make format` to auto-fix what's fixable.
- `mypy` runs in non-strict-but-meaningful mode against `src/` (tests are
  excluded from type-checking to keep test-writing friction low).
- Follow the existing layering: `domain/` has no framework dependencies,
  `infrastructure/` wraps external I/O, `service/` wires domain +
  infrastructure into the LangGraph graph, `api/` is a thin FastAPI shell
  over `service/`.
