"""Retrieval configuration."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class RAGBackend(StrEnum):
    LOCAL = "local"
    PGVECTOR = "pgvector"


@dataclass(frozen=True)
class RAGConfig:
    backend: RAGBackend = RAGBackend.LOCAL
    index_path: Path = Path(".rag/index.json")
    embedding_model: str = "deterministic-sha256-v1"
    embedding_dimension: int = 64
    default_top_k: int = 4
