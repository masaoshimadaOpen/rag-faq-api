"""LangChain retriever adapter — skipped unless `langchain_core` is installed."""
from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from app.langchain_adapter import as_langchain_retriever


def test_retriever_returns_documents(engine, faq_docs):
    engine.ingest(faq_docs)
    retriever = as_langchain_retriever(engine, top_k=3)

    docs = retriever.invoke("How long do refunds take?")
    assert docs, "expected retrieved documents"
    # top result is the refunds doc, metadata carries our id + score
    assert docs[0].metadata["id"] == "refunds"
    assert "score" in docs[0].metadata
    assert isinstance(docs[0].page_content, str) and docs[0].page_content
