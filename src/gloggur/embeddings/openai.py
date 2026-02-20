from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from gloggur.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls the OpenAI embeddings API."""
    def __init__(self, model: str) -> None:
        """Initialize the OpenAI client and model selection."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self._dimension: int | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        response = self.client.embeddings.create(model=self.model, input=text)
        vector = response.data[0].embedding
        self._dimension = len(vector)
        return vector

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        vectors = [item.embedding for item in response.data]
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def get_dimension(self) -> int:
        """Return the embedding dimension (probe if unknown)."""
        if self._dimension is None:
            _ = self.embed_text("dimension probe")
        return self._dimension or 0
