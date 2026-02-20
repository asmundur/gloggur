from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

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
        self._symbol_to_vector_id: Dict[str, int] = {}
        self._vector_id_to_symbol: Dict[int, str] = {}
        self._next_vector_id = 1
        self._fallback_vectors: Dict[str, List[float]] = {}
        self._fallback_order: List[str] = []
        self._fallback_path = os.path.join(self.config.cache_dir, "vectors.npy")
        self._faiss_available = self._check_faiss()
        os.makedirs(self.config.cache_dir, exist_ok=True)
        self.load()

    def add_vectors(self, symbols: Iterable[Symbol]) -> None:
        """Backward-compatible alias for upsert semantics."""

        self.upsert_vectors(symbols)

    def upsert_vectors(self, symbols: Iterable[Symbol]) -> None:
        """Insert or replace symbol vectors in the underlying index."""

        updates: List[tuple[str, List[float]]] = []
        remove_candidates: List[str] = []
        for symbol in symbols:
            if symbol.embedding_vector is None:
                continue
            remove_candidates.append(symbol.id)
            updates.append((symbol.id, symbol.embedding_vector))
        if not updates:
            return

        self.remove_ids(remove_candidates)
        if self._faiss_available:
            self._upsert_faiss_vectors(updates)
        else:
            for symbol_id, vector in updates:
                self._ensure_vector_id(symbol_id)
                self._fallback_vectors[symbol_id] = vector
        self._persist_id_map()

    def remove_ids(self, symbol_ids: Iterable[str]) -> None:
        """Remove vectors by symbol ids."""

        unique_ids = {symbol_id for symbol_id in symbol_ids if symbol_id}
        if not unique_ids:
            return

        vector_ids = [
            self._symbol_to_vector_id[symbol_id]
            for symbol_id in unique_ids
            if symbol_id in self._symbol_to_vector_id
        ]
        if not vector_ids:
            return

        if self._faiss_available and self._index is not None and vector_ids:
            ids_array = np.asarray(vector_ids, dtype="int64")
            self._index.remove_ids(ids_array)
        for symbol_id in unique_ids:
            self._fallback_vectors.pop(symbol_id, None)
            vector_id = self._symbol_to_vector_id.pop(symbol_id, None)
            if vector_id is not None:
                self._vector_id_to_symbol.pop(vector_id, None)
        self._persist_id_map()

    def search(self, query_vector: List[float], k: int) -> List[tuple[str, float]]:
        """Return k nearest symbol ids for a query vector."""

        if k <= 0:
            return []
        if self._faiss_available and self._index is not None:
            vector = np.asarray([query_vector], dtype="float32")
            distances, indices = self._index.search(vector, k)
            results = []
            for vector_id, distance in zip(indices[0], distances[0]):
                if vector_id < 0:
                    continue
                symbol_id = self._vector_id_to_symbol.get(int(vector_id))
                if not symbol_id:
                    continue
                results.append((symbol_id, float(distance)))
            return results

        if not self._fallback_vectors:
            return []
        symbol_ids = list(self._fallback_vectors.keys())
        matrix = np.asarray(
            [self._fallback_vectors[symbol_id] for symbol_id in symbol_ids],
            dtype="float32",
        )
        vector = np.asarray(query_vector, dtype="float32")
        diffs = matrix - vector
        distances = np.sum(diffs * diffs, axis=1)
        top_indices = np.argsort(distances)[: min(k, len(distances))]
        return [(symbol_ids[idx], float(distances[idx])) for idx in top_indices]

    def save(self) -> None:
        """Persist vectors and id mapping to disk."""

        self._persist_id_map()
        if self._faiss_available and self._index is not None:
            import faiss

            faiss.write_index(self._index, self.config.index_path)
            self._remove_file(self._fallback_path)
            return

        if self._fallback_vectors:
            self._fallback_order = sorted(
                self._fallback_vectors,
                key=lambda symbol_id: self._symbol_to_vector_id.get(symbol_id, 0),
            )
            matrix = np.asarray(
                [self._fallback_vectors[symbol_id] for symbol_id in self._fallback_order],
                dtype="float32",
            )
            np.save(self._fallback_path, matrix)
        else:
            self._remove_file(self._fallback_path)
        self._touch_index_placeholder()

    def load(self) -> None:
        """Load vectors and id mapping from disk."""

        legacy_map = self._load_id_map()
        if self._faiss_available and os.path.exists(self.config.index_path):
            import faiss

            self._index = faiss.read_index(self.config.index_path)
        else:
            self._load_fallback_vectors()

        if legacy_map:
            self._migrate_legacy_vectors()

    def clear(self) -> None:
        """Remove all persisted vectors and metadata files."""

        self._index = None
        self._symbol_to_vector_id = {}
        self._vector_id_to_symbol = {}
        self._next_vector_id = 1
        self._fallback_vectors = {}
        self._fallback_order = []
        for path in (self.config.index_path, self.config.id_map_path, self._fallback_path):
            self._remove_file(path)

    def _upsert_faiss_vectors(self, updates: Sequence[tuple[str, List[float]]]) -> None:
        """Insert vectors into an ID-mapped FAISS index."""

        if not updates:
            return
        if self._index is None:
            bootstrap_items = list(self._fallback_vectors.items())
            bootstrap_dim = len(updates[0][1])
            if bootstrap_items:
                bootstrap_dim = len(bootstrap_items[0][1])
            self._index = self._create_index(bootstrap_dim)
            if bootstrap_items:
                bootstrap_matrix = np.asarray(
                    [vector for _symbol_id, vector in bootstrap_items],
                    dtype="float32",
                )
                bootstrap_ids = np.asarray(
                    [self._ensure_vector_id(symbol_id) for symbol_id, _vector in bootstrap_items],
                    dtype="int64",
                )
                self._index.add_with_ids(bootstrap_matrix, bootstrap_ids)
                self._fallback_vectors = {}
                self._fallback_order = []
        matrix = np.asarray([vector for _symbol_id, vector in updates], dtype="float32")
        vector_ids = np.asarray(
            [self._ensure_vector_id(symbol_id) for symbol_id, _vector in updates],
            dtype="int64",
        )
        self._index.add_with_ids(matrix, vector_ids)

    def _ensure_vector_id(self, symbol_id: str) -> int:
        """Return a stable int64 id for a symbol, allocating one if needed."""

        existing = self._symbol_to_vector_id.get(symbol_id)
        if existing is not None:
            return existing
        vector_id = self._next_vector_id
        self._next_vector_id += 1
        self._symbol_to_vector_id[symbol_id] = vector_id
        self._vector_id_to_symbol[vector_id] = symbol_id
        return vector_id

    def _persist_id_map(self) -> None:
        """Write the id map payload to disk."""

        payload = {
            "schema_version": 2,
            "next_vector_id": self._next_vector_id,
            "symbol_to_vector_id": self._symbol_to_vector_id,
            "fallback_order": self._fallback_order,
        }
        with open(self.config.id_map_path, "w", encoding="utf8") as handle:
            json.dump(payload, handle)

    def _load_id_map(self) -> bool:
        """Read id mapping from disk. Returns True when legacy format is detected."""

        self._symbol_to_vector_id = {}
        self._vector_id_to_symbol = {}
        self._next_vector_id = 1
        self._fallback_order = []
        if not os.path.exists(self.config.id_map_path):
            return False
        with open(self.config.id_map_path, "r", encoding="utf8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for idx, symbol_id in enumerate(payload, start=1):
                self._symbol_to_vector_id[str(symbol_id)] = idx
                self._vector_id_to_symbol[idx] = str(symbol_id)
            self._next_vector_id = len(payload) + 1
            return True
        if not isinstance(payload, dict):
            return False

        raw_map = payload.get("symbol_to_vector_id", {})
        if isinstance(raw_map, dict):
            for symbol_id, vector_id in raw_map.items():
                numeric_id = int(vector_id)
                self._symbol_to_vector_id[str(symbol_id)] = numeric_id
                self._vector_id_to_symbol[numeric_id] = str(symbol_id)
        raw_next = payload.get("next_vector_id")
        if raw_next is None:
            self._next_vector_id = max(self._vector_id_to_symbol.keys(), default=0) + 1
        else:
            self._next_vector_id = int(raw_next)
        raw_order = payload.get("fallback_order", [])
        if isinstance(raw_order, list):
            self._fallback_order = [str(symbol_id) for symbol_id in raw_order]
        return False

    def _load_fallback_vectors(self) -> None:
        """Load fallback vectors from .npy file."""

        self._fallback_vectors = {}
        if not os.path.exists(self._fallback_path):
            return
        try:
            matrix = np.load(self._fallback_path)
        except OSError:
            return
        if matrix.size == 0:
            return

        if self._fallback_order and len(self._fallback_order) == len(matrix):
            symbol_ids = list(self._fallback_order)
        else:
            symbol_ids = [
                symbol_id
                for symbol_id, _vector_id in sorted(
                    self._symbol_to_vector_id.items(),
                    key=lambda item: item[1],
                )
            ]
            symbol_ids = symbol_ids[: len(matrix)]
        for symbol_id, vector in zip(symbol_ids, matrix.tolist()):
            self._fallback_vectors[symbol_id] = vector

    def _migrate_legacy_vectors(self) -> None:
        """Rebuild vector artifacts from cached symbol embeddings after legacy map detection."""

        vectors = self._load_vectors_from_db()
        self._index = None
        self._fallback_vectors = {}
        self._fallback_order = []
        self._symbol_to_vector_id = {}
        self._vector_id_to_symbol = {}
        self._next_vector_id = 1
        if self._faiss_available:
            self._upsert_faiss_vectors(vectors)
        else:
            for symbol_id, vector in vectors:
                self._ensure_vector_id(symbol_id)
                self._fallback_vectors[symbol_id] = vector
        self.save()

    def _load_vectors_from_db(self) -> List[tuple[str, List[float]]]:
        """Load (symbol_id, embedding_vector) tuples from index.db."""

        db_path = os.path.join(self.config.cache_dir, "index.db")
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT id, embedding_vector FROM symbols WHERE embedding_vector IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        vectors: List[tuple[str, List[float]]] = []
        for symbol_id, raw_vector in rows:
            if not raw_vector:
                continue
            try:
                vector = json.loads(raw_vector)
            except json.JSONDecodeError:
                continue
            if not isinstance(vector, list):
                continue
            vectors.append((str(symbol_id), [float(item) for item in vector]))
        return vectors

    @staticmethod
    def _create_index(dimension: int):
        """Create a FAISS index for the given dimension."""

        import faiss

        return faiss.IndexIDMap2(faiss.IndexFlatL2(dimension))

    @staticmethod
    def _check_faiss() -> bool:
        """Return True if FAISS can be imported."""

        try:
            import faiss  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _remove_file(path: str) -> None:
        """Best-effort file deletion."""

        if os.path.exists(path):
            os.remove(path)

    def _touch_index_placeholder(self) -> None:
        """Create a placeholder index file when FAISS is unavailable."""

        if os.path.exists(self.config.index_path):
            return
        try:
            with open(self.config.index_path, "wb") as handle:
                handle.write(b"FAISS_UNAVAILABLE\n")
        except OSError:
            pass
