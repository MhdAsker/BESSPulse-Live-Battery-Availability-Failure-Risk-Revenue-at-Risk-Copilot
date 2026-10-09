"""Incremental local JSON vector index and CLI."""

import json
from pathlib import Path

from rag.documents import chunk_document, discover_documents
from rag.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from rag.schemas import KnowledgeChunk


class LocalVectorIndex:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(".rag/index.json")

    def build(self, provider: EmbeddingProvider | None = None) -> tuple[KnowledgeChunk, ...]:
        embedder = provider or DeterministicEmbeddingProvider()
        chunks = [chunk for path in discover_documents() for chunk in chunk_document(path)]
        existing = {item.content_hash: item for item in self.load()}
        missing = [item for item in chunks if item.content_hash not in existing]
        vectors = embedder.embed([item.text for item in missing]) if missing else []
        refreshed = [
            existing.get(item.content_hash) or item.model_copy(update={"vector": tuple(vector)})
            for item, vector in zip(missing, vectors, strict=True)
        ]
        by_hash = {item.content_hash: item for item in existing.values()}
        by_hash.update({item.content_hash: item for item in refreshed})
        result = tuple(by_hash[item.content_hash] for item in chunks)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([item.model_dump() for item in result]), encoding="utf-8")
        return result

    def load(self) -> tuple[KnowledgeChunk, ...]:
        if not self.path.exists():
            return ()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return tuple(KnowledgeChunk.model_validate(item) for item in payload)


def main() -> None:
    chunks = LocalVectorIndex().build()
    print(f"Indexed {len(chunks)} approved chunks.")


if __name__ == "__main__":
    main()
