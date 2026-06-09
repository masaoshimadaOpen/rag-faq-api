"""Retrieval evaluation harness — measure and compare RAG retrieval quality.

The single most important thing you can do to *improve* a RAG system is to
*measure* it. This module scores retrieval against a labelled dataset
(questions → the document ids that should be retrieved) using the standard
information-retrieval metrics:

  - Hit@k       : did any relevant doc appear in the top-k?  (coverage)
  - MRR         : 1 / rank of the first relevant doc         (ranking)
  - Recall@k    : fraction of relevant docs found in top-k   (completeness)
  - Precision@k : fraction of top-k that were relevant       (noise)

Everything runs on retrieval only (no LLM call), so evaluation is free and
fast. With the offline `HashingEmbedder` the whole harness runs with no API
keys, which keeps it usable in CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .embeddings import HashingEmbedder
from .llm import StubLLM
from .rag import RAGEngine
from .vectorstore import InMemoryVectorStore, SearchHit


@dataclass(frozen=True)
class EvalExample:
    """A labelled query: which parent doc ids count as correct retrieval."""

    question: str
    relevant_ids: tuple[str, ...]


@dataclass
class ExampleResult:
    question: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    hit: bool
    reciprocal_rank: float
    recall: float
    precision: float


@dataclass
class RetrievalReport:
    top_k: int
    num_examples: int
    hit_rate: float
    mrr: float
    recall_at_k: float
    precision_at_k: float
    per_example: list[ExampleResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"n={self.num_examples} top_k={self.top_k} | "
            f"Hit@{self.top_k}={self.hit_rate:.3f} "
            f"MRR={self.mrr:.3f} "
            f"Recall@{self.top_k}={self.recall_at_k:.3f} "
            f"Precision@{self.top_k}={self.precision_at_k:.3f}"
        )


def ranked_parent_ids(hits: list[SearchHit]) -> list[str]:
    """Collapse chunk hits to a ranked list of unique parent doc ids.

    Chunks share their parent's id, so the same document can be retrieved
    several times; we keep only its best (first) appearance to rank at the
    document level.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for h in hits:
        if h.chunk.id not in seen:
            seen.add(h.chunk.id)
            ranked.append(h.chunk.id)
    return ranked


def evaluate_example(engine: RAGEngine, example: EvalExample, top_k: int) -> ExampleResult:
    hits = engine.retrieve(example.question, top_k=top_k)
    retrieved = ranked_parent_ids(hits)
    relevant = set(example.relevant_ids)

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    found = [d for d in retrieved if d in relevant]
    recall = len(set(found)) / len(relevant) if relevant else 0.0
    precision = len(found) / len(retrieved) if retrieved else 0.0

    return ExampleResult(
        question=example.question,
        retrieved_ids=retrieved,
        relevant_ids=list(example.relevant_ids),
        hit=bool(found),
        reciprocal_rank=reciprocal_rank,
        recall=recall,
        precision=precision,
    )


def evaluate_retrieval(
    engine: RAGEngine, dataset: list[EvalExample], top_k: int = 3
) -> RetrievalReport:
    """Score an already-ingested engine against a labelled dataset."""
    results = [evaluate_example(engine, ex, top_k) for ex in dataset]
    n = len(results) or 1
    return RetrievalReport(
        top_k=top_k,
        num_examples=len(results),
        hit_rate=sum(r.hit for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        recall_at_k=sum(r.recall for r in results) / n,
        precision_at_k=sum(r.precision for r in results) / n,
        per_example=results,
    )


# ─── Chunking strategy sweep ──────────────────────────────────────────────

@dataclass
class ChunkingResult:
    chunk_size: int
    chunk_overlap: int
    report: RetrievalReport


def default_engine_factory(chunk_size: int, chunk_overlap: int) -> RAGEngine:
    """Offline engine for a given chunking config (no keys, no network)."""
    return RAGEngine(
        embedder=HashingEmbedder(dim=256),
        store=InMemoryVectorStore(),
        llm=StubLLM(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def sweep_chunking(
    documents: list[dict],
    dataset: list[EvalExample],
    configs: list[tuple[int, int]],
    top_k: int = 3,
    engine_factory: Callable[[int, int], RAGEngine] = default_engine_factory,
) -> list[ChunkingResult]:
    """Re-ingest under each (chunk_size, overlap) config and score retrieval.

    Returns one result per config, sorted best-first by MRR then Hit-rate —
    the empirical way to *pick* a chunking strategy instead of guessing.
    """
    results: list[ChunkingResult] = []
    for chunk_size, chunk_overlap in configs:
        engine = engine_factory(chunk_size, chunk_overlap)
        engine.ingest(documents)
        report = evaluate_retrieval(engine, dataset, top_k=top_k)
        results.append(ChunkingResult(chunk_size, chunk_overlap, report))

    results.sort(key=lambda r: (r.report.mrr, r.report.hit_rate), reverse=True)
    return results


def best_chunking(results: list[ChunkingResult]) -> ChunkingResult:
    """The top config from a sweep (results are pre-sorted best-first)."""
    if not results:
        raise ValueError("no chunking results to choose from")
    return results[0]
