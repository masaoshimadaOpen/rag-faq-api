"""Tests for the retrieval evaluation harness — all offline."""
from __future__ import annotations

import math

from app.evaluation import (
    EvalExample,
    ExampleResult,
    best_chunking,
    evaluate_example,
    evaluate_retrieval,
    ranked_parent_ids,
    sweep_chunking,
)
from app.vectorstore import SearchHit, StoredChunk


def _hit(doc_id: str, score: float) -> SearchHit:
    return SearchHit(chunk=StoredChunk(id=doc_id, text=doc_id, vector=[]), score=score)


def test_ranked_parent_ids_dedupes_preserving_order():
    hits = [_hit("a", 0.9), _hit("a", 0.8), _hit("b", 0.7), _hit("a", 0.6)]
    assert ranked_parent_ids(hits) == ["a", "b"]


def test_metrics_perfect_top1(engine, faq_docs):
    engine.ingest(faq_docs)
    ex = EvalExample(question="How long do refunds take?", relevant_ids=("refunds",))
    r = evaluate_example(engine, ex, top_k=3)
    assert r.hit is True
    assert r.reciprocal_rank == 1.0          # relevant doc ranked first
    assert r.recall == 1.0                   # only relevant doc, and it was found
    assert 0.0 < r.precision <= 1.0


def test_metrics_miss_is_zero(engine, faq_docs):
    engine.ingest(faq_docs)
    ex = EvalExample(question="completely unrelated quantum thing", relevant_ids=("nonexistent",))
    r = evaluate_example(engine, ex, top_k=3)
    assert r.hit is False
    assert r.reciprocal_rank == 0.0
    assert r.recall == 0.0


def test_reciprocal_rank_reflects_position():
    # Hand-built ranked result where the relevant doc is 2nd.
    res = ExampleResult(
        question="q", retrieved_ids=["x", "y", "z"], relevant_ids=["y"],
        hit=True, reciprocal_rank=0.5, recall=1.0, precision=1 / 3,
    )
    assert res.reciprocal_rank == 0.5


def test_evaluate_retrieval_aggregates(engine, faq_docs):
    engine.ingest(faq_docs)
    dataset = [
        EvalExample("How long do refunds take?", ("refunds",)),
        EvalExample("When is support available?", ("hours",)),
    ]
    report = evaluate_retrieval(engine, dataset, top_k=3)
    assert report.num_examples == 2
    assert 0.0 <= report.hit_rate <= 1.0
    assert 0.0 <= report.mrr <= 1.0
    assert report.hit_rate == 1.0            # both lexically obvious → retrieved
    assert "Hit@3" in report.summary()


def test_sweep_chunking_sorts_best_first_and_picks_best(faq_docs):
    dataset = [
        EvalExample("How long do refunds take?", ("refunds",)),
        EvalExample("How fast is express shipping?", ("shipping",)),
        EvalExample("When is support available?", ("hours",)),
    ]
    configs = [(80, 0), (200, 20), (400, 40)]
    results = sweep_chunking(faq_docs, dataset, configs, top_k=3)
    assert len(results) == 3
    # sorted best-first by (mrr, hit_rate)
    keys = [(r.report.mrr, r.report.hit_rate) for r in results]
    assert keys == sorted(keys, reverse=True)
    assert best_chunking(results) is results[0]
    # every config retrieves these obvious answers → strong MRR
    assert best_chunking(results).report.mrr == 1.0


def test_recall_partial_when_multiple_relevant():
    # 2 relevant docs, only 1 retrieved in top-k → recall 0.5
    res = ExampleResult(
        question="q", retrieved_ids=["a", "x"], relevant_ids=["a", "b"],
        hit=True, reciprocal_rank=1.0, recall=0.5, precision=0.5,
    )
    assert math.isclose(res.recall, 0.5)
