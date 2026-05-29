"""Embedding providers behind a tiny interface.

`HashingEmbedder` is deterministic and dependency-free so the whole RAG
pipeline can run and be tested offline. Real providers (OpenAI / Gemini)
are imported lazily and only used when a key is configured.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import Settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


class Embedder(Protocol):
    """Maps texts to fixed-length vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder:
    """Offline, deterministic bag-of-tokens hashing embedder.

    Not semantically as strong as a learned model, but it is stable, free,
    and good enough to demonstrate and test retrieval end-to-end. Tokens are
    hashed into a fixed number of buckets (the "hashing trick") and the
    resulting vector is L2-normalized so cosine similarity is meaningful.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        # Term-frequency over hashed buckets (the "hashing trick"), then
        # L2-normalize. We intentionally use non-negative counts (no sign
        # hashing) so that texts sharing tokens get an intuitive, clearly
        # positive cosine similarity — important for a readable demo.
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return _normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """Real embeddings via OpenAI (lazy import)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI  # lazy: only needed when selected

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


class GeminiEmbedder:
    """Real embeddings via Gemini (lazy import)."""

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        import google.generativeai as genai  # lazy

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            r = self._genai.embed_content(model=self._model, content=t)
            out.append(r["embedding"])
        return out


def get_embedder(settings: Settings) -> Embedder:
    """Select an embedder, falling back to offline hashing if unavailable."""
    provider = settings.embedding_provider
    try:
        if provider == "openai" and settings.openai_api_key:
            return OpenAIEmbedder(settings.openai_api_key)
        if provider == "gemini" and settings.gemini_api_key:
            return GeminiEmbedder(settings.gemini_api_key)
    except Exception:
        # Missing SDK or bad key → stay bootable with the offline embedder.
        pass
    return HashingEmbedder()
