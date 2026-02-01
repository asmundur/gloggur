from __future__ import annotations

import os
from typing import Iterable, List

from gloggur.embeddings.base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini embeddings")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini embeddings") from exc
        self._client = genai.Client(api_key=self.api_key)
        self._dimension: int | None = None

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_batch([text])
        if not vectors:
            raise RuntimeError("Gemini embeddings returned no vectors")
        return vectors[0]

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        payload = list(texts)
        response = self._client.models.embed_content(model=self.model, contents=payload)
        vectors = self._extract_vectors(response)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def get_dimension(self) -> int:
        if self._dimension is None:
            _ = self.embed_text("dimension probe")
        return self._dimension or 0

    @staticmethod
    def _extract_vectors(response: object) -> List[List[float]]:
        embeddings = getattr(response, "embeddings", None) or getattr(response, "embedding", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings") or response.get("embedding")
        if embeddings is None:
            return []
        vectors: List[List[float]] = []
        for item in embeddings:
            if isinstance(item, list):
                vectors.append(item)
            elif isinstance(item, dict):
                if "values" in item:
                    vectors.append(list(item["values"]))
                elif "embedding" in item:
                    vectors.append(list(item["embedding"]))
            elif hasattr(item, "values"):
                vectors.append(list(item.values))
        return vectors
