"""Embedding provider abstraction with offline deterministic default."""

import hashlib
import importlib
import math
import os
import re
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicEmbeddingProvider:
    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += -1.0 if digest[4] & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class GeminiEmbeddingProvider:
    """Optional provider; import and credential access are deliberately lazy."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-embedding-001") -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        genai = importlib.import_module("google.genai")
        self.client = genai.Client(api_key=key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.client.models.embed_content(model=self.model, contents=texts)
        return [list(item.values) for item in result.embeddings]
