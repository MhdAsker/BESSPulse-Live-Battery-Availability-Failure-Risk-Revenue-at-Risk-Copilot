"""Strict document discovery and heading-aware deterministic chunking."""

import hashlib
import re
from pathlib import Path

from rag.schemas import KnowledgeChunk

ROOT = Path(__file__).resolve().parents[2]
APPROVED_FILES = (ROOT / "README.md",)
APPROVED_DIRS = (ROOT / "docs",)
EXCLUDED_PARTS = {".git", ".env", "data", "artifacts", "mlruns", ".rag", "secrets"}


def discover_documents() -> tuple[Path, ...]:
    candidates = list(APPROVED_FILES)
    for directory in APPROVED_DIRS:
        candidates.extend(directory.rglob("*.md"))
    return tuple(
        sorted(
            path.resolve()
            for path in candidates
            if path.is_file() and not any(part.lower() in EXCLUDED_PARTS for part in path.parts)
        )
    )


def load_approved_document(path: Path) -> str:
    resolved = path.resolve()
    if resolved not in discover_documents():
        raise ValueError("Document is outside the approved knowledge-base allowlist.")
    return resolved.read_text(encoding="utf-8")


def chunk_document(path: Path, max_chars: int = 1600) -> tuple[KnowledgeChunk, ...]:
    text = load_approved_document(path)
    relative = path.resolve().relative_to(ROOT).as_posix()
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+)$", line)
        if match:
            if lines:
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = match.group(1).strip(), []
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines).strip()))
    chunks: list[KnowledgeChunk] = []
    for section_heading, body in sections:
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", body) if item.strip()]
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 2 > max_chars:
                chunks.append(_chunk(relative, section_heading, current, len(chunks)))
                current = paragraph
            else:
                current = f"{current}\n\n{paragraph}".strip()
        if current:
            chunks.append(_chunk(relative, section_heading, current, len(chunks)))
    return tuple(chunks)


def _chunk(source: str, heading: str, text: str, ordinal: int) -> KnowledgeChunk:
    digest = hashlib.sha256(f"{source}\0{heading}\0{text}".encode()).hexdigest()
    return KnowledgeChunk(
        chunk_id=f"kb-{digest[:16]}-{ordinal}",
        source_path=source,
        heading=heading,
        text=text,
        content_hash=digest,
    )
