"""Citation validation helpers."""

from rag.schemas import Citation


def validate_citations(citations: tuple[Citation, ...]) -> bool:
    return bool(citations) and all(
        item.citation_id.startswith("kb-")
        and item.source_path.endswith(".md")
        and bool(item.heading)
        for item in citations
    )
