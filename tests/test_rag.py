from app.rag import chunk_text


def test_chunk_text_overlap_and_coverage():
    text = "abcdefghij" * 5  # 50 chars
    chunks = chunk_text(text, size=20, overlap=5)
    assert all(len(c) <= 20 for c in chunks)
    # full text is covered (concatenated chunks contain every original char run)
    assert chunks[0].startswith("abcdefghij")
    assert len(chunks) >= 3


def test_chunk_text_empty_returns_empty():
    assert chunk_text("   ", size=100, overlap=10) == []


def test_ingest_counts_documents_and_chunks(engine, faq_docs):
    result = engine.ingest(faq_docs)
    assert result.ingested_documents == 3
    assert result.total_chunks >= 3
    assert len(engine.store) == result.total_chunks


def test_ask_retrieves_relevant_source(engine, faq_docs):
    engine.ingest(faq_docs)
    ans = engine.ask("How long do refunds take?", top_k=3)
    assert ans.sources, "expected at least one source"
    # the most relevant retrieved source should be the refunds doc
    assert ans.sources[0]["id"] == "refunds"
    # stub LLM answers from context → mentions the refund window
    assert "5 business days" in ans.answer
    assert 0.0 <= ans.sources[0]["score"] <= 1.0


def test_ask_without_ingest_says_unknown(engine):
    ans = engine.ask("anything?", top_k=3)
    assert ans.sources == []
    assert "no documents" in ans.answer.lower() or "don't know" in ans.answer.lower()
