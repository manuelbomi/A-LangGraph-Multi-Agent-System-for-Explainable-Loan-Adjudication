# Makefile: standard entry points for local development and CI.
#
# Every target assumes a Python virtual environment at .venv/ (see `make
# install`). Targets are intentionally thin wrappers around the same
# commands documented in README.md's "Getting Started" section -- if a
# target ever drifts from what's documented there, trust this file and
# update the README to match.

.PHONY: install test lint typecheck format run docker-build compose-up clean

VENV := .venv
PYTHON := $(VENV)/Scripts/python.exe
PIP := $(VENV)/Scripts/pip.exe

install: ## Create a virtualenv and install the package + dev/test dependencies.
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test: ## Run the full pytest suite (fully offline, no API keys required).
	$(PYTHON) -m pytest -v

lint: ## Run ruff static analysis.
	$(PYTHON) -m ruff check src tests

format: ## Auto-fix ruff-fixable lint issues.
	$(PYTHON) -m ruff check src tests --fix

typecheck: ## Run mypy static type checking.
	$(PYTHON) -m mypy src

run: ## Run the FastAPI app locally with auto-reload.
	$(PYTHON) -m uvicorn underwriting.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Build the production Docker image.
	docker build -t credit-underwriting-decision-graph:local .

compose-up: ## Start the service via docker-compose (one-command local spin-up).
	docker compose up --build

clean: ## Remove caches and the local SQLite checkpoint DB.
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf data/checkpoints.sqlite
	find . -type d -name __pycache__ -exec rm -rf {} +
