"""End-to-end API tests via FastAPI TestClient (offline providers)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _reset_index():
    # Each test starts from a clean in-memory index.
    app.state.engine.store.clear()


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body and "embedding_provider" in body


def test_ingest_then_ask_flow():
    _reset_index()
    ingest = client.post(
        "/ingest",
        json={"documents": [
            {"id": "refunds", "text": "Refunds are processed within 5 business days."},
            {"id": "shipping", "text": "Express shipping arrives next day."},
        ]},
    )
    assert ingest.status_code == 200
    assert ingest.json()["ingested_documents"] == 2

    # Offline hashing embedder matches lexically, so the question shares the
    # "refunds" token with the target doc (real embeddings handle synonyms).
    ask = client.post("/ask", json={"question": "How long do refunds take?", "top_k": 2})
    assert ask.status_code == 200
    body = ask.json()
    assert body["sources"][0]["id"] == "refunds"
    assert "5 business days" in body["answer"]


def test_ingest_validation_rejects_empty_documents():
    r = client.post("/ingest", json={"documents": []})
    assert r.status_code == 422


def test_ask_validation_rejects_blank_question():
    r = client.post("/ask", json={"question": "", "top_k": 3})
    assert r.status_code == 422
