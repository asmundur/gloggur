from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from gloggur.models import Symbol


@dataclass
class VectorStoreConfig:
    """Configuration for the vector index store (FAISS and fallback paths)."""
    cache_dir: str
    index_name: str = "vectors.index"
    id_map_name: str = "vectors.json"

    @property
    def index_path(self) -> str:
        """Return the path to the FAISS index."""
        return os.path.join(self.cache_dir, self.index_name)

    @property
    def id_map_path(self) -> str:
        """Return the path to the id map JSON file."""
        return os.path.join(self.cache_dir, self.id_map_name)


class VectorStore:
    """Vector index backed by FAISS with a numpy fallback."""
    def __init__(self, config: VectorStoreConfig) -> None:
        """Initialize the vector store and load existing data."""
        self.config = config
        self._index = None
        self._id_map: List[str] = []
        self._fallback_vectors: List[List[float]] = []
        self._fallback_path = os.path.join(self.config.cache_dir, "vectors.npy")
        self._faiss_available = self._check_faiss()
        os.makedirs(self.config.cache_dir, exist_ok=True)
        self.load()

    def add_vectors(self, symbols: Iterable[Symbol]) -> None:
        """Add Symbol.embedding_vector values to FAISS or numpy fallback."""
        vectors = []
        ids = []
        existing = set(self._id_map)
        for symbol in symbols:
            if symbol.embedding_vector is None:
                continue
            if symbol.id in existing:
                continue
            vectors.append(symbol.embedding_vector)
            ids.append(symbol.id)
        if not vectors:
            return
        matrix = np.asarray(vectors, dtype="float32")
        if self._faiss_available:
            if self._index is None:
                self._index = self._create_index(matrix.shape[1])
            self._index.add(matrix)
        else:
            self._fallback_vectors.extend(vectors)
        self._id_map.extend(ids)
        self._persist_id_map()

    def search(self, query_vector: List[float], k: int) -> List[tuple[str, float]]:
        """Return k nearest symbol ids for a query vector."""
        if not self._id_map:
            return []
        if self._faiss_available and self._index is not None:
            vector = np.asarray([query_vector], dtype="float32")
            distances, indices = self._index.search(vector, k)
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                results.append((self._id_map[idx], float(distance)))
            return results
        if not self._fallback_vectors:
            return []
        matrix = np.asarray(self._fallback_vectors, dtype="float32")
        vector = np.asarray(query_vector, dtype="float32")
        diffs = matrix - vector
        distances = np.sum(diffs * diffs, axis=1)
        if k <= 0:
            return []
        top_indices = np.argsort(distances)[: min(k, len(distances))]
        return [(self._id_map[idx], float(distances[idx])) for idx in top_indices]

    def save(self) -> None:
        """Persist the FAISS index and id map to disk."""
        self._persist_id_map()
        if self._faiss_available and self._index is not None:
            import faiss

            faiss.write_index(self._index, self.config.index_path)
            return
        if self._fallback_vectors:
            np.save(self._fallback_path, np.asarray(self._fallback_vectors, dtype="float32"))
        self._touch_index_placeholder()

    def load(self) -> None:
        """Load FAISS index and id map from disk if present."""
        self._load_id_map()
        if self._faiss_available and os.path.exists(self.config.index_path):
            import faiss

            self._index = faiss.read_index(self.config.index_path)
            return
        if os.path.exists(self._fallback_path):
            try:
                matrix = np.load(self._fallback_path)
            except OSError:
                matrix = np.asarray([], dtype="float32")
            self._fallback_vectors = matrix.tolist() if matrix.size else []

    def clear(self) -> None:
        """Remove all persisted vectors and metadata files."""
        self._index = None
        self._id_map = []
        self._fallback_vectors = []
        for path in (self.config.index_path, self.config.id_map_path, self._fallback_path):
            if os.path.exists(path):
                os.remove(path)

    def _persist_id_map(self) -> None:
        """Write the id map to disk."""
        with open(self.config.id_map_path, "w", encoding="utf8") as handle:
            json.dump(self._id_map, handle)

    def _load_id_map(self) -> None:
        """Read the id map from disk."""
        if not os.path.exists(self.config.id_map_path):
            self._id_map = []
            return
        with open(self.config.id_map_path, "r", encoding="utf8") as handle:
            self._id_map = json.load(handle)

    @staticmethod
    def _create_index(dimension: int):
        """Create a FAISS index for the given dimension."""
        import faiss

        return faiss.IndexFlatL2(dimension)

    @staticmethod
    def _check_faiss() -> bool:
        """Return True if FAISS can be imported."""
        try:
            import faiss  # noqa: F401
        except Exception:
            return False
        return True

    def _touch_index_placeholder(self) -> None:
        """Create a placeholder index file when FAISS is unavailable."""
        if os.path.exists(self.config.index_path):
            return
        try:
            with open(self.config.index_path, "wb") as handle:
                handle.write(b"FAISS_UNAVAILABLE\n")
        except OSError:
            pass
