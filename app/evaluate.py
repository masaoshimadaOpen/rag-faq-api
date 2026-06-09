"""CLI: run the retrieval eval harness against a labelled dataset.

    python -m app.evaluate                          # uses examples/eval_dataset.json
    python -m app.evaluate path/to/dataset.json     # custom dataset

Dataset JSON shape:
    {
      "documents": [{"id": "...", "text": "...", "metadata": {...}?}, ...],
      "examples":  [{"question": "...", "relevant_ids": ["id", ...]}, ...]
    }

Runs fully offline (hashing embedder + stub LLM) with no API keys, prints a
retrieval report and a chunking-strategy sweep so you can see which chunk
size wins on *this* corpus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .evaluation import (
    EvalExample,
    evaluate_retrieval,
    sweep_chunking,
)
from .rag import RAGEngine
from .embeddings import HashingEmbedder
from .llm import StubLLM
from .vectorstore import InMemoryVectorStore

_DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "examples" / "eval_dataset.json"


def load_dataset(path: Path) -> tuple[list[dict], list[EvalExample]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data["documents"]
    examples = [
        EvalExample(question=e["question"], relevant_ids=tuple(e["relevant_ids"]))
        for e in data["examples"]
    ]
    return documents, examples


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else _DEFAULT_DATASET
    if not path.exists():
        print(f"dataset not found: {path}", file=sys.stderr)
        return 1

    documents, dataset = load_dataset(path)

    engine = RAGEngine(
        embedder=HashingEmbedder(dim=256),
        store=InMemoryVectorStore(),
        llm=StubLLM(),
        chunk_size=400,
        chunk_overlap=40,
    )
    engine.ingest(documents)

    print("== Retrieval evaluation (offline hashing embedder) ==")
    for top_k in (1, 3, 5):
        report = evaluate_retrieval(engine, dataset, top_k=top_k)
        print("  " + report.summary())

    print("\n== Chunking strategy sweep (sorted best-first by MRR) ==")
    configs = [(120, 0), (200, 20), (400, 40), (800, 80)]
    results = sweep_chunking(documents, dataset, configs, top_k=3)
    for r in results:
        print(
            f"  size={r.chunk_size:>4} overlap={r.chunk_overlap:>3} | "
            f"{r.report.summary()}"
        )
    best = results[0]
    print(f"\n  -> best: size={best.chunk_size} overlap={best.chunk_overlap} "
          f"(MRR={best.report.mrr:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
