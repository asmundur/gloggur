from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from gloggur.models import Symbol
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig


def _symbol(symbol_id: str, vector: list[float]) -> Symbol:
    return Symbol(
        id=symbol_id,
        name="add",
        kind="function",
        file_path="/tmp/sample.py",
        start_line=1,
        end_line=2,
        body_hash="hash",
        embedding_vector=vector,
        language="python",
    )


def _create_symbol_table(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE symbols (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                signature TEXT,
                docstring TEXT,
                body_hash TEXT NOT NULL,
                embedding_vector TEXT,
                language TEXT
            );
            """
        )
    finally:
        conn.commit()
        conn.close()


def test_vector_store_upsert_and_remove_ids_without_faiss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    store = VectorStore(VectorStoreConfig(str(tmp_path)))
    first = _symbol("s1", [0.1, 0.2, 0.3])

    store.upsert_vectors([first])
    results = store.search([0.1, 0.2, 0.3], k=5)
    assert results and results[0][0] == "s1"

    updated = _symbol("s1", [0.9, 0.9, 0.9])
    store.upsert_vectors([updated])
    store.remove_ids(["s1"])

    assert store.search([0.9, 0.9, 0.9], k=5) == []


def test_vector_store_upsert_and_remove_ids_with_faiss_double(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFaissIndex:
        def __init__(self) -> None:
            self._vectors: dict[int, np.ndarray] = {}

        def add_with_ids(self, matrix: np.ndarray, ids: np.ndarray) -> None:
            for vector, vector_id in zip(matrix, ids):
                self._vectors[int(vector_id)] = np.asarray(vector, dtype="float32")

        def remove_ids(self, ids: np.ndarray) -> None:
            for vector_id in ids:
                self._vectors.pop(int(vector_id), None)

        def search(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            if not self._vectors:
                distances = np.full((1, k), np.inf, dtype="float32")
                indices = np.full((1, k), -1, dtype="int64")
                return distances, indices

            query = vector[0]
            scored = []
            for vector_id, candidate in self._vectors.items():
                distance = float(np.sum((candidate - query) ** 2))
                scored.append((distance, vector_id))
            scored.sort(key=lambda item: item[0])
            scored = scored[:k]

            distances = np.full((1, k), np.inf, dtype="float32")
            indices = np.full((1, k), -1, dtype="int64")
            for idx, (distance, vector_id) in enumerate(scored):
                distances[0, idx] = distance
                indices[0, idx] = vector_id
            return distances, indices

    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: True))
    monkeypatch.setattr(
        VectorStore,
        "_create_index",
        staticmethod(lambda _dimension: FakeFaissIndex()),
    )
    store = VectorStore(VectorStoreConfig(str(tmp_path)))
    first = _symbol("s1", [0.1, 0.2, 0.3])

    store.upsert_vectors([first])
    results = store.search([0.1, 0.2, 0.3], k=5)
    assert results and results[0][0] == "s1"

    updated = _symbol("s1", [0.9, 0.9, 0.9])
    store.upsert_vectors([updated])
    assert isinstance(store._index, FakeFaissIndex)
    assert len(store._index._vectors) == 1
    results_after_upsert = store.search([0.9, 0.9, 0.9], k=5)
    assert results_after_upsert and results_after_upsert[0][0] == "s1"

    store.remove_ids(["s1"])
    assert store.search([0.9, 0.9, 0.9], k=5) == []


def test_vector_store_migrates_legacy_id_map_without_faiss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    cache_dir = Path(tmp_path)
    db_path = cache_dir / "index.db"
    _create_symbol_table(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO symbols (
                id, name, kind, file_path, start_line, end_line,
                signature, docstring, body_hash, embedding_vector, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "s1",
                "add",
                "function",
                "sample.py",
                1,
                2,
                "def add(a, b):",
                None,
                "hash",
                json.dumps([0.2, 0.3, 0.4]),
                "python",
            ),
        )
    finally:
        conn.commit()
        conn.close()

    (cache_dir / "vectors.json").write_text(json.dumps(["s1"]), encoding="utf8")
    store = VectorStore(VectorStoreConfig(str(cache_dir)))

    payload = json.loads((cache_dir / "vectors.json").read_text(encoding="utf8"))
    assert payload["schema_version"] == 2
    assert payload["symbol_to_vector_id"]["s1"] == 1
    assert store.search([0.2, 0.3, 0.4], k=5)[0][0] == "s1"
