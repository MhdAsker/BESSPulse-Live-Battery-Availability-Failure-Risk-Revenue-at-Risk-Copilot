"""Hybrid semantic/lexical retrieval with stable citations."""

import re

from rag.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from rag.index import LocalVectorIndex
from rag.schemas import Citation, KnowledgeChunk, RetrievalResult

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "is",
    "of",
    "the",
    "to",
    "what",
}


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
        query_terms = self._terms(query)
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
        terms = KnowledgeService._terms(f"{chunk.heading} {chunk.text}")
        heading_terms = KnowledgeService._terms(chunk.heading)
        lexical = len(query_terms & terms) / max(1, len(query_terms))
        heading = len(query_terms & heading_terms) / max(1, len(query_terms))
        return 0.25 * max(0.0, semantic) + 0.55 * lexical + 0.20 * heading

    @staticmethod
    def _terms(text: str) -> set[str]:
        expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
        expanded = re.sub(r"\b(\d+)\s*[- ]\s*hours?\b", r"\1h", expanded, flags=re.I)
        tokens = [
            token
            for token in re.findall(r"[a-z0-9_]+", expanded.lower())
            if token not in STOP_WORDS
        ]
        terms = set(tokens)
        terms.update(
            "".join(tokens[index : index + size])
            for size in (2, 3)
            for index in range(len(tokens) - size + 1)
        )
        return terms

    @staticmethod
    def _citation(chunk: KnowledgeChunk) -> Citation:
        excerpt = " ".join(chunk.text.split())[:300]
        return Citation(
            citation_id=chunk.chunk_id,
            source_path=chunk.source_path,
            heading=chunk.heading,
            excerpt=excerpt,
        )
