"""Application configuration, loaded from environment variables / `.env`.

Nothing secret is ever hardcoded: the OpenAI key is read from the environment
and is never logged or returned in an API response.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to `default` when unset or malformed."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid integer for %s=%r; using default %s", name, raw, default
        )
        return default


class Settings(BaseModel):
    """Runtime settings for the agent, backend client and server."""

    # --- OpenAI -----------------------------------------------------------
    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-4o-mini"
    # Leave empty for OpenAI itself. Set it to use any OpenAI-compatible
    # endpoint (Groq, OpenRouter, Together, a local server) with the same key.
    openai_base_url: str = ""
    openai_temperature: float = 0.0
    openai_timeout_seconds: int = 30
    openai_max_retries: int = 2

    # --- Internal customer backend ---------------------------------------
    backend_base_url: str = "http://127.0.0.1:8000"
    backend_timeout_seconds: float = 5.0
    # "none" | "timeout" | "server_error" | "malformed" — demo switch used to
    # exercise the error-handling paths without editing code.
    backend_failure_mode: str = "none"

    # --- Agent ------------------------------------------------------------
    agent_max_iterations: int = 4
    history_max_messages: int = 12
    session_ttl_seconds: int = 3600

    # --- Server -----------------------------------------------------------
    log_level: str = "INFO"

    @property
    def llm_configured(self) -> bool:
        """True when an OpenAI key is present, so the real LLM can be used."""
        return bool(self.openai_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings built from the process environment."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0") or 0),
        openai_timeout_seconds=_env_int("OPENAI_TIMEOUT_SECONDS", 30),
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 2),
        backend_base_url=os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000"),
        backend_timeout_seconds=float(os.getenv("BACKEND_TIMEOUT_SECONDS", "5") or 5),
        backend_failure_mode=os.getenv("BACKEND_FAILURE_MODE", "none").strip().lower(),
        agent_max_iterations=_env_int("AGENT_MAX_ITERATIONS", 4),
        history_max_messages=_env_int("HISTORY_MAX_MESSAGES", 12),
        session_ttl_seconds=_env_int("SESSION_TTL_SECONDS", 3600),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, at process start."""
    logging.basicConfig(
        level=level or get_settings().log_level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
