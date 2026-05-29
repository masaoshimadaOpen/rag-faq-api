"""A dependency-free in-memory vector store with cosine search.

The interface is deliberately tiny (`add`, `search`) so it can be swapped
for pgvector / FAISS / a managed vector DB in production without touching
the RAG pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class StoredChunk:
    id: str            # parent document id
    text: str
    vector: list[float]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchHit:
    chunk: StoredChunk
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Minimal cosine-similarity store. Not for millions of vectors — for
    clarity and offline testing. Swap behind the same interface in prod."""

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunk: StoredChunk) -> None:
        self._chunks.append(chunk)

    def search(self, query_vector: list[float], top_k: int = 3) -> list[SearchHit]:
        scored = [
            SearchHit(chunk=c, score=cosine_similarity(query_vector, c.vector))
            for c in self._chunks
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: max(0, top_k)]

    def clear(self) -> None:
        self._chunks.clear()
