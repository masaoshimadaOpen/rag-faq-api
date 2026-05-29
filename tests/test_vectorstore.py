from app.embeddings import HashingEmbedder
from app.vectorstore import InMemoryVectorStore, StoredChunk, cosine_similarity


def test_cosine_identical_is_one():
    v = [0.1, 0.2, 0.3]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_dimension_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])


def test_search_ranks_most_similar_first():
    emb = HashingEmbedder(dim=128)
    store = InMemoryVectorStore()
    texts = {
        "cats": "cats are small domestic animals that purr",
        "finance": "interest rates affect bond prices and yields",
        "weather": "rain and clouds are expected tomorrow afternoon",
    }
    for doc_id, text in texts.items():
        vec = emb.embed([text])[0]
        store.add(StoredChunk(id=doc_id, text=text, vector=vec))

    query_vec = emb.embed(["how do bond yields move with interest rates"])[0]
    hits = store.search(query_vec, top_k=3)

    assert len(hits) == 3
    assert hits[0].chunk.id == "finance"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_top_k_limits_results():
    emb = HashingEmbedder(dim=64)
    store = InMemoryVectorStore()
    for i in range(5):
        store.add(StoredChunk(id=str(i), text=f"document number {i}", vector=emb.embed([str(i)])[0]))
    assert len(store.search(emb.embed(["doc"])[0], top_k=2)) == 2
