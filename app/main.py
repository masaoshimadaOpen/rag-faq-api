"""FastAPI application exposing the RAG pipeline.

Endpoints:
  GET  /health  — liveness + active providers
  POST /ingest  — add documents to the knowledge base
  POST /ask     — ask a grounded question, get an answer + sources
"""
from __future__ import annotations

from fastapi import FastAPI

from .config import load_settings
from .embeddings import get_embedder
from .llm import get_llm
from .rag import RAGEngine
from .schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    Source,
)
from .vectorstore import InMemoryVectorStore


def make_store(settings):
    """Pick a vector store, falling back to in-memory if Chroma is unavailable."""
    if settings.vector_store == "chroma":
        try:
            from .vectorstore_chroma import ChromaVectorStore  # lazy/optional

            return ChromaVectorStore(persist_path=settings.chroma_path)
        except Exception:
            # Missing chromadb or bad path → stay bootable in-memory.
            pass
    return InMemoryVectorStore()


def build_engine() -> RAGEngine:
    """Construct a RAGEngine from current settings (providers auto-selected)."""
    settings = load_settings()
    return RAGEngine(
        embedder=get_embedder(settings),
        store=make_store(settings),
        llm=get_llm(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


app = FastAPI(title="rag-faq-api", version="0.1.0")
app.state.engine = build_engine()
app.state.settings = load_settings()


@app.get("/health")
def health() -> dict:
    s = app.state.settings
    return {
        "status": "ok",
        "llm_provider": s.llm_provider,
        "embedding_provider": s.embedding_provider,
        "vector_store": type(app.state.engine.store).__name__,
        "indexed_chunks": len(app.state.engine.store),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    result = app.state.engine.ingest([d.model_dump() for d in req.documents])
    return IngestResponse(
        ingested_documents=result.ingested_documents,
        total_chunks=result.total_chunks,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    ans = app.state.engine.ask(req.question, top_k=req.top_k)
    return AskResponse(
        answer=ans.answer,
        sources=[Source(**s) for s in ans.sources],
    )
