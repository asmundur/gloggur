from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the vector dimensionality for embeddings."""
        raise NotImplementedError
