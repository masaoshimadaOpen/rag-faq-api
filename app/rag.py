"""The RAG pipeline: chunk → embed → store → retrieve → answer.

`RAGEngine` wires together an Embedder, a vector store, and an LLMProvider.
All three are injected, so the engine is trivially testable with offline
fakes and identical in behavior with real providers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder
from .llm import LLMProvider, SYSTEM_PROMPT, build_prompt
from .vectorstore import InMemoryVectorStore, SearchHit, StoredChunk


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows.

    Overlap preserves context that would otherwise be cut at a boundary.
    """
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        return [text]
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


@dataclass
class IngestResult:
    ingested_documents: int
    total_chunks: int


@dataclass
class Answer:
    answer: str
    sources: list[dict]


class RAGEngine:
    def __init__(
        self,
        embedder: Embedder,
        store: InMemoryVectorStore,
        llm: LLMProvider,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, documents: list[dict]) -> IngestResult:
        """documents: [{"id", "text", "metadata"?}]"""
        all_chunks: list[tuple[str, str, dict]] = []
        for doc in documents:
            for piece in chunk_text(doc["text"], self.chunk_size, self.chunk_overlap):
                all_chunks.append((doc["id"], piece, doc.get("metadata", {})))

        if all_chunks:
            vectors = self.embedder.embed([c[1] for c in all_chunks])
            for (doc_id, piece, meta), vec in zip(all_chunks, vectors):
                self.store.add(StoredChunk(id=doc_id, text=piece, vector=vec, metadata=meta))

        return IngestResult(ingested_documents=len(documents), total_chunks=len(all_chunks))

    def retrieve(self, question: str, top_k: int = 3) -> list[SearchHit]:
        """Retrieve top-k chunks for a question — no LLM call.

        Separated from `ask` so retrieval quality can be evaluated on its own
        (the eval harness runs this thousands of times at zero LLM cost).
        """
        query_vec = self.embedder.embed([question])[0]
        return self.store.search(query_vec, top_k=top_k)

    def ask(self, question: str, top_k: int = 3) -> Answer:
        hits = self.retrieve(question, top_k=top_k)
        contexts = [h.chunk.text for h in hits]

        if not contexts:
            return Answer(answer="I don't know — no documents have been ingested.", sources=[])

        prompt = build_prompt(question, contexts)
        answer_text = self.llm.complete(SYSTEM_PROMPT, prompt)

        sources = [
            {
                "id": h.chunk.id,
                "score": round(h.score, 4),
                "snippet": h.chunk.text[:200],
                "metadata": h.chunk.metadata,
            }
            for h in hits
        ]
        return Answer(answer=answer_text, sources=sources)
