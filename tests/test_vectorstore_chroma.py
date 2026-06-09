"""Chroma store parity tests — skipped unless `chromadb` is installed.

Keeps the default offline suite green (no heavy dependency required) while
proving the Chroma store honours the same interface as the in-memory one.
"""
from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from app.embeddings import HashingEmbedder
from app.llm import StubLLM
from app.rag import RAGEngine
from app.vectorstore import InMemoryVectorStore, StoredChunk
from app.vectorstore_chroma import ChromaVectorStore


@pytest.fixture
def chroma_store():
    store = ChromaVectorStore(collection_name="test_parity")
    store.clear()  # ensure empty across test runs
    yield store
    store.clear()


def test_add_and_len(chroma_store):
    emb = HashingEmbedder(dim=64)
    vec = emb.embed(["hello world"])[0]
    chroma_store.add(StoredChunk(id="d1", text="hello world", vector=vec))
    assert len(chroma_store) == 1


def test_search_matches_in_memory_store(chroma_store):
    # Parity: given identical vectors + query, Chroma ranks the same top doc
    # as the reference in-memory cosine store. This verifies the store honours
    # the interface/semantics, independent of any embedder quirk.
    emb = HashingEmbedder(dim=256)
    docs = {
        "refunds": "Refunds are processed within 5 business days after approval.",
        "shipping": "Express shipping is delivered the next business day.",
        "hours": "Support is available Monday to Friday, 9am to 6pm JST.",
    }
    mem = InMemoryVectorStore()
    for doc_id, text in docs.items():
        chunk = StoredChunk(id=doc_id, text=text, vector=emb.embed([text])[0])
        chroma_store.add(chunk)
        mem.add(chunk)

    q = emb.embed(["How long do refunds take?"])[0]
    chroma_hits = chroma_store.search(q, top_k=3)
    mem_hits = mem.search(q, top_k=3)

    assert chroma_hits, "expected hits"
    assert chroma_hits[0].chunk.id == mem_hits[0].chunk.id   # same winner
    assert -1.0 <= chroma_hits[0].score <= 1.0


def test_engine_works_with_chroma_store(faq_docs):
    # Same RAG pipeline, only the store swapped → identical top-1 behaviour.
    engine = RAGEngine(
        embedder=HashingEmbedder(dim=256),
        store=ChromaVectorStore(collection_name="test_engine"),
        llm=StubLLM(),
        chunk_size=200,
        chunk_overlap=20,
    )
    engine.store.clear()
    engine.ingest(faq_docs)
    ans = engine.ask("How long do refunds take?", top_k=3)
    assert ans.sources[0]["id"] == "refunds"
    engine.store.clear()
