"""Retrieval interface used by Copilot tools."""

from typing import Protocol

from rag.schemas import RetrievalResult


class Retriever(Protocol):
    def search_knowledge_base(self, query: str, limit: int = 4) -> RetrievalResult: ...
