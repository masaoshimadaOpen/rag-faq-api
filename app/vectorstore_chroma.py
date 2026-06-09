"""A Chroma-backed vector store implementing the same tiny interface as
`InMemoryVectorStore` (`add` / `search` / `__len__` / `clear`).

This is the "swap the store behind the same interface" promise made good:
the RAG pipeline doesn't change at all, only the persistence layer does.
`chromadb` is imported lazily, so the rest of the package (and its offline
tests) keep working with no extra dependency installed.

We supply our own embedding vectors (from the project's Embedder), so Chroma
is used purely as an ANN index — its built-in embedding function is bypassed.
Cosine space is selected to match the in-memory store's similarity.
"""
from __future__ import annotations

from .vectorstore import SearchHit, StoredChunk

_PARENT_KEY = "__parent_id__"


class ChromaVectorStore:
    def __init__(self, collection_name: str = "rag_faq", persist_path: str | None = None) -> None:
        import chromadb  # lazy: only needed when this store is selected

        self._client = (
            chromadb.PersistentClient(path=persist_path)
            if persist_path
            else chromadb.EphemeralClient()
        )
        self._name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._auto = 0

    def __len__(self) -> int:
        return self._collection.count()

    def add(self, chunk: StoredChunk) -> None:
        # Chroma needs a globally-unique id per row and rejects empty metadata,
        # so we namespace the row id and stash the parent doc id in metadata.
        meta = {**chunk.metadata, _PARENT_KEY: chunk.id}
        self._collection.add(
            ids=[f"{chunk.id}::{self._auto}"],
            embeddings=[chunk.vector],
            documents=[chunk.text],
            metadatas=[meta],
        )
        self._auto += 1

    def search(self, query_vector: list[float], top_k: int = 3) -> list[SearchHit]:
        if self._collection.count() == 0 or top_k <= 0:
            return []
        res = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        hits: list[SearchHit] = []
        for text, meta, dist in zip(docs, metas, dists):
            parent_id = str(meta.get(_PARENT_KEY, ""))
            clean_meta = {k: v for k, v in meta.items() if k != _PARENT_KEY}
            # Chroma returns cosine *distance*; convert back to similarity.
            score = 1.0 - float(dist)
            hits.append(
                SearchHit(
                    chunk=StoredChunk(id=parent_id, text=text, vector=[], metadata=clean_meta),
                    score=score,
                )
            )
        return hits

    def clear(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )
        self._auto = 0
