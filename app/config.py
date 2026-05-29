"""Runtime configuration, loaded from environment variables.

Kept dependency-free (no pydantic-settings) so the sample stays small.
Offline defaults (`stub` LLM + `hash` embeddings) mean the app boots and
all tests pass with no API keys set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "stub"
    embedding_provider: str = "hash"
    top_k: int = 3
    chunk_size: int = 400
    chunk_overlap: int = 40

    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None

    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"


def load_settings() -> Settings:
    """Build Settings from the current environment."""
    def _int(name: str, default: int) -> int:
        raw = os.getenv(name)
        try:
            return int(raw) if raw not in (None, "") else default
        except ValueError:
            return default

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "stub").strip().lower() or "stub",
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower() or "hash",
        top_k=_int("TOP_K", 3),
        chunk_size=_int("CHUNK_SIZE", 400),
        chunk_overlap=_int("CHUNK_OVERLAP", 40),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
