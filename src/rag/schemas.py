"""Typed RAG records and citations."""

from pydantic import BaseModel, ConfigDict


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: str
    source_path: str
    heading: str
    text: str
    content_hash: str
    vector: tuple[float, ...] = ()


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)
    citation_id: str
    source_path: str
    heading: str
    excerpt: str


class RetrievalResult(BaseModel):
    query: str
    citations: tuple[Citation, ...] = ()
    scores: tuple[float, ...] = ()
