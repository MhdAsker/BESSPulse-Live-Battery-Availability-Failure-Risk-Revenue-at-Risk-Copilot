# Grounded knowledge retrieval

The knowledge base accepts only an explicit allowlist of root methodology documents, `README.md`, and Markdown files under `docs/`. Data, artifacts, environment files, secrets, indexes, and arbitrary caller-provided paths are excluded. Documents are split on headings and paragraph boundaries; source-relative path, heading, content hash, and stable chunk ID are retained.

Build the local index explicitly with `python -m rag.index`. API startup and runtime retrieval never rebuild it. The default embedding provider is deterministic and offline, making CI reproducible. `GeminiEmbeddingProvider` is optional, imports the SDK lazily, and reads `GEMINI_API_KEY` only from configuration. The local JSON index lives in `.rag/`, is content-hash incremental, and is ignored by Git. `LocalVectorIndex` is the storage abstraction boundary for a future pgvector backend.

Retrieval combines cosine similarity with lexical overlap and returns structured citations. With `RAG_ENABLED=1`, the existing Copilot endpoint exposes a retrieval-only compatibility response. Prompt 13 agent/tool orchestration is absent from this repository, so this phase does not claim Gemini synthesis or operational tool routing.
