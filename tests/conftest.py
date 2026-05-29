"""Shared fixtures. Everything runs offline with the hashing embedder and
the stub LLM — no API keys, no network."""
from __future__ import annotations

import pytest

from app.embeddings import HashingEmbedder
from app.llm import StubLLM
from app.rag import RAGEngine
from app.vectorstore import InMemoryVectorStore


@pytest.fixture
def engine() -> RAGEngine:
    return RAGEngine(
        embedder=HashingEmbedder(dim=256),
        store=InMemoryVectorStore(),
        llm=StubLLM(),
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.fixture
def faq_docs() -> list[dict]:
    return [
        {"id": "refunds", "text": "Refunds are processed within 5 business days after approval."},
        {"id": "shipping", "text": "Standard shipping takes 3 to 7 days. Express shipping is next day."},
        {"id": "hours", "text": "Our support team is available Monday to Friday, 9am to 6pm JST."},
    ]
