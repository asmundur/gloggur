from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        raise NotImplementedError
