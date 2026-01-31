from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from gloggur.models import Symbol


@dataclass
class VectorStoreConfig:
    cache_dir: str
    index_name: str = "vectors.index"
    id_map_name: str = "vectors.json"

    @property
    def index_path(self) -> str:
        return os.path.join(self.cache_dir, self.index_name)

    @property
    def id_map_path(self) -> str:
        return os.path.join(self.cache_dir, self.id_map_name)


class VectorStore:
    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self._index = None
        self._id_map: List[str] = []
        os.makedirs(self.config.cache_dir, exist_ok=True)
        self.load()

    def add_vectors(self, symbols: Iterable[Symbol]) -> None:
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
        if self._index is None:
            self._index = self._create_index(matrix.shape[1])
        self._index.add(matrix)
        self._id_map.extend(ids)
        self._persist_id_map()

    def search(self, query_vector: List[float], k: int) -> List[tuple[str, float]]:
        if self._index is None or not self._id_map:
            return []
        vector = np.asarray([query_vector], dtype="float32")
        distances, indices = self._index.search(vector, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(distance)))
        return results

    def save(self) -> None:
        if self._index is None:
            return
        self._persist_id_map()
        import faiss

        faiss.write_index(self._index, self.config.index_path)

    def load(self) -> None:
        if not os.path.exists(self.config.index_path):
            self._load_id_map()
            return
        import faiss

        self._index = faiss.read_index(self.config.index_path)
        self._load_id_map()

    def clear(self) -> None:
        self._index = None
        self._id_map = []
        for path in (self.config.index_path, self.config.id_map_path):
            if os.path.exists(path):
                os.remove(path)

    def _persist_id_map(self) -> None:
        with open(self.config.id_map_path, "w", encoding="utf8") as handle:
            json.dump(self._id_map, handle)

    def _load_id_map(self) -> None:
        if not os.path.exists(self.config.id_map_path):
            self._id_map = []
            return
        with open(self.config.id_map_path, "r", encoding="utf8") as handle:
            self._id_map = json.load(handle)

    @staticmethod
    def _create_index(dimension: int):
        import faiss

        return faiss.IndexFlatL2(dimension)
