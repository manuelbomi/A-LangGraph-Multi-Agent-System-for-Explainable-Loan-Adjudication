# syntax=docker/dockerfile:1
#
# Multi-stage build: a "builder" stage compiles the virtualenv (so build
# tooling and pip caches never reach the final image), and a slim "runtime"
# stage copies only the venv + application source, runs as a non-root user,
# and exposes /healthz + /readyz for container orchestrators.

# ---- Builder stage -----------------------------------------------------
FROM python:3.10.15-slim-bookworm AS builder

# Keep pip quiet/deterministic; avoid writing .pyc cache clutter into layers.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Copy only dependency-relevant files first so Docker's layer cache is
# reused across source-code-only changes.
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# ---- Runtime stage -------------------------------------------------------
FROM python:3.10.15-slim-bookworm AS runtime

# Never run as root inside the container -- create a dedicated, unprivileged
# user/group for the application process.
RUN groupadd --system app && useradd --system --gid app --home /app app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    CHECKPOINT_DB_PATH=/app/data/checkpoints.sqlite

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src ./src

# The checkpoint DB directory must exist and be writable by the non-root user.
RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "underwriting.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
