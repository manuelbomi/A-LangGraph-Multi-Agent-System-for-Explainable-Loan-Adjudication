"""
Centralized application configuration.

Architectural role
-------------------
This module is the single source of truth for runtime configuration. Every
other module that needs a configurable value (log level, DB path, risk
thresholds, provider keys) imports `get_settings()` from here instead of
reading `os.environ` directly. That keeps configuration:

  * discoverable  -- one place to see every knob the app exposes
  * validated     -- pydantic-settings parses/type-checks env vars at
                      startup and fails fast on nonsense values
  * secret-safe   -- secrets are `SecretStr` so they never render in `repr()`
                      or accidentally land in a log line via `str(settings)`

Values are sourced from process environment variables first and a local
`.env` file second (see `.env.example` for the documented template). No
secret is ever hardcoded here or anywhere else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from environment / .env.

    Grouped loosely by concern: app metadata, LLM provider selection,
    persistence, risk-policy thresholds, and API server binding.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application metadata ---
    app_env: str = Field(default="local", description="Deployment environment label.")
    log_level: str = Field(default="INFO", description="Python logging level name.")
    service_name: str = Field(
        default="credit-underwriting-decision-graph",
        description="Service identifier used in structured logs and traces.",
    )

    # --- LLM provider selection (all optional -> defaults to offline mock) ---
    openai_api_key: SecretStr | None = Field(default=None)
    anthropic_api_key: SecretStr | None = Field(default=None)
    llm_model_name: str = Field(default="gpt-4o-mini")

    # --- Persistence ---
    checkpoint_db_path: str = Field(
        default="./data/checkpoints.sqlite",
        description="SQLite file backing the LangGraph durable checkpointer.",
    )

    # --- Risk policy thresholds (synthetic / illustrative only) ---
    dti_borderline_low: float = Field(
        default=0.36,
        description="DTI at/above this value enters the borderline review band.",
    )
    dti_borderline_high: float = Field(
        default=0.45,
        description="DTI at/above this value exits the borderline band toward decline.",
    )
    dti_hard_decline: float = Field(
        default=0.55,
        description="DTI at/above this value is an automatic decline, no escalation.",
    )

    # --- API server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    def has_real_llm_provider(self) -> bool:
        """True if a real hosted-LLM API key is configured.

        Used by the LLM factory (infrastructure/llm.py) to decide between
        wiring up ChatOpenAI/ChatAnthropic vs. falling back to the offline
        deterministic MockChatModel. Kept here (rather than duplicated at
        each call site) so the "what counts as configured" rule lives in
        exactly one place.
        """
        return bool(self.openai_api_key or self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance.

    `lru_cache` gives us a cheap singleton without needing a global mutable
    variable or a DI container -- calling `get_settings()` anywhere in the
    app returns the same validated object. Tests can bypass the cache by
    constructing `Settings(...)` directly with overrides.
    """
    return Settings()
