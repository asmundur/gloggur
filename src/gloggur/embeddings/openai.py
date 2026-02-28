from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from gloggur.embeddings.base import EmbeddingProvider


def _normalize_vector(vector: object, *, model: str, context: str) -> list[float]:
    """Validate and normalize one embedding vector from the OpenAI client."""
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise RuntimeError(
            f"OpenAI embeddings returned invalid vector "
            f"payload for model '{model}' during {context}"
        )
    normalized: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"OpenAI embeddings returned non-numeric "
                f"vector value for model '{model}' "
                f"during {context}"
            )
        normalized.append(float(value))
    if not normalized:
        raise RuntimeError(
            f"OpenAI embeddings returned an empty vector for model '{model}' during {context}"
        )
    return normalized


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls the OpenAI embeddings API."""

    def __init__(self, model: str) -> None:
        """Initialize the OpenAI client and model selection."""
        self.provider = "openai"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self._dimension: int | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI embedding request failed for model '{self.model}': {exc}"
            ) from exc
        data = getattr(response, "data", None)
        if not isinstance(data, list) or len(data) != 1:
            raise RuntimeError(
                f"OpenAI embeddings returned invalid item "
                f"count for model '{self.model}' during "
                f"single-text embedding"
            )
        vector = _normalize_vector(
            getattr(data[0], "embedding", None),
            model=self.model,
            context="single-text embedding",
        )
        self._dimension = len(vector)
        return vector

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        payload = list(texts)
        if not payload:
            return []
        try:
            response = self.client.embeddings.create(model=self.model, input=payload)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI embedding request failed for model '{self.model}': {exc}"
            ) from exc
        data = getattr(response, "data", None)
        if not isinstance(data, list):
            raise RuntimeError(
                f"OpenAI embeddings returned invalid response "
                f"payload for model '{self.model}' "
                f"during batch embedding"
            )
        vectors = [
            _normalize_vector(
                getattr(item, "embedding", None),
                model=self.model,
                context="batch embedding",
            )
            for item in data
        ]
        if len(vectors) != len(payload):
            raise RuntimeError(
                f"OpenAI embeddings returned "
                f"{len(vectors)} vectors for "
                f"{len(payload)} inputs with "
                f"model '{self.model}'"
            )
        expected_dimension = len(vectors[0]) if vectors else 0
        if any(len(vector) != expected_dimension for vector in vectors):
            raise RuntimeError(
                f"OpenAI embeddings returned inconsistent "
                f"vector dimensions for model '{self.model}'"
            )
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def get_dimension(self) -> int:
        """Return the embedding dimension (probe if unknown)."""
        if self._dimension is None:
            _ = self.embed_text("dimension probe")
        return self._dimension or 0
