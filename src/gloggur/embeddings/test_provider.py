from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

from gloggur.embeddings.base import EmbeddingProvider


class DeterministicTestEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline embedding provider intended for tests only."""

    def __init__(self, *, dimension: int = 256) -> None:
        self.provider = "test"
        self._dimension = dimension
        self._token_pattern = re.compile(r"[A-Za-z0-9_]+")
        self._token_vector_cache: dict[str, list[float]] = {}

    def embed_text(self, text: str) -> list[float]:
        tokens = self._token_pattern.findall(text.lower())
        if not tokens:
            tokens = [text]
        values = [0.0] * self._dimension
        for token in tokens:
            token_vector = self._token_vector_cache.get(token)
            if token_vector is None:
                token_vector = self._vector_from_seed(token.encode("utf8"))
                self._token_vector_cache[token] = token_vector
            for idx, token_value in enumerate(token_vector):
                values[idx] += token_value
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        return self._dimension

    def _vector_from_seed(self, seed: bytes) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self._dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(digest), 4):
                if len(values) >= self._dimension:
                    break
                chunk = digest[i : i + 4]
                if len(chunk) < 4:
                    continue
                number = int.from_bytes(chunk, "big")
                values.append((number / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        return values
