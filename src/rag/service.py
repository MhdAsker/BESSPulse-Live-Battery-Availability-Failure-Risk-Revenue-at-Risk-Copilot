"""Hybrid semantic/lexical retrieval with stable citations."""

import math
import re

from rag.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from rag.index import LocalVectorIndex
from rag.schemas import Citation, KnowledgeChunk, RetrievalResult


class KnowledgeService:
    def __init__(
        self,
        index: LocalVectorIndex | None = None,
        provider: EmbeddingProvider | None = None,
        auto_build: bool = False,
    ) -> None:
        self.index = index or LocalVectorIndex()
        self.provider = provider or DeterministicEmbeddingProvider()
        self.auto_build = auto_build

    def search_knowledge_base(self, query: str, limit: int = 4) -> RetrievalResult:
        if not query.strip():
            return RetrievalResult(query=query)
        chunks = self.index.load()
        if not chunks and self.auto_build:
            chunks = self.index.build(self.provider)
        if not chunks:
            return RetrievalResult(query=query)
        query_vector = self.provider.embed([query])[0]
        query_terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
        ranked = sorted(
            ((self._score(item, query_vector, query_terms), item) for item in chunks),
            key=lambda pair: (-pair[0], pair[1].chunk_id),
        )[: max(1, min(limit, 10))]
        citations = tuple(self._citation(item) for score, item in ranked if score > 0)
        scores = tuple(round(score, 6) for score, _ in ranked if score > 0)
        return RetrievalResult(query=query, citations=citations, scores=scores)

    @staticmethod
    def _score(chunk: KnowledgeChunk, query_vector: list[float], query_terms: set[str]) -> float:
        semantic = sum(a * b for a, b in zip(query_vector, chunk.vector, strict=False))
        terms = set(re.findall(r"[a-z0-9_]+", f"{chunk.heading} {chunk.text}".lower()))
        lexical = len(query_terms & terms) / math.sqrt(max(1, len(query_terms) * len(terms)))
        return 0.65 * max(0.0, semantic) + 0.35 * lexical

    @staticmethod
    def _citation(chunk: KnowledgeChunk) -> Citation:
        excerpt = " ".join(chunk.text.split())[:300]
        return Citation(
            citation_id=chunk.chunk_id,
            source_path=chunk.source_path,
            heading=chunk.heading,
            excerpt=excerpt,
        )
