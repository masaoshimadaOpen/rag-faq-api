"""Pydantic request/response models — the typed edge of the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(..., min_length=1, description="Caller-supplied unique id")
    text: str = Field(..., min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Document] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ingested_documents: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class Source(BaseModel):
    id: str
    score: float
    snippet: str
    metadata: dict[str, str] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
