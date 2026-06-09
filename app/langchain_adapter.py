"""Expose the RAG engine's retrieval as a LangChain Retriever.

Lets the project's retrieval plug into LangChain chains/agents (LCEL,
`create_retrieval_chain`, tool-calling agents, …) without rewriting the
pipeline. `langchain_core` is imported lazily so it stays an optional
dependency — nothing else in the package needs it.
"""
from __future__ import annotations

from .rag import RAGEngine


def as_langchain_retriever(engine: RAGEngine, top_k: int = 3):
    """Wrap a RAGEngine as a `langchain_core` BaseRetriever.

    Returns an object usable anywhere LangChain expects a retriever, e.g.::

        retriever = as_langchain_retriever(engine, top_k=4)
        docs = retriever.invoke("How long do refunds take?")
    """
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class _EngineRetriever(BaseRetriever):
        # `engine`/`top_k` are captured from the closure rather than declared
        # as pydantic fields, so we don't need arbitrary-type model config.
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> list:
            hits = engine.retrieve(query, top_k=top_k)
            return [
                Document(
                    page_content=h.chunk.text,
                    metadata={"id": h.chunk.id, "score": h.score, **h.chunk.metadata},
                )
                for h in hits
            ]

    return _EngineRetriever()
