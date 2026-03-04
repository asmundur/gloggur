from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_vector_store_defers_id_map_persistence_until_save_without_faiss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    store = VectorStore(VectorStoreConfig(str(tmp_path)))
    symbol = _symbol("s1", [0.1, 0.2, 0.3])

    store.upsert_vectors([symbol])

    assert store._id_map_dirty is True
    assert store.search([0.1, 0.2, 0.3], k=5)[0][0] == "s1"
    assert not (tmp_path / "vectors.json").exists()

    store.save()

    assert store._id_map_dirty is False
    payload = json.loads((tmp_path / "vectors.json").read_text(encoding="utf8"))
    assert payload["symbol_to_vector_id"] == {"s1": 1}

    reloaded = VectorStore(VectorStoreConfig(str(tmp_path)))
    assert reloaded._id_map_dirty is False
    assert reloaded.search([0.1, 0.2, 0.3], k=5)[0][0] == "s1"


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


def test_vector_store_defers_id_map_persistence_until_save_with_faiss_double(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFaissIndex:
        def __init__(self) -> None:
            self._vectors: dict[int, np.ndarray] = {}

        def add_with_ids(self, matrix: np.ndarray, ids: np.ndarray) -> None:
            for vector, vector_id in zip(matrix, ids, strict=True):
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

    def _write_index(index: FakeFaissIndex, path: str) -> None:
        payload = {
            str(vector_id): vector.tolist()
            for vector_id, vector in index._vectors.items()
        }
        Path(path).write_text(json.dumps(payload), encoding="utf8")

    def _read_index(path: str) -> FakeFaissIndex:
        payload = json.loads(Path(path).read_text(encoding="utf8"))
        index = FakeFaissIndex()
        index._vectors = {
            int(vector_id): np.asarray(vector, dtype="float32")
            for vector_id, vector in payload.items()
        }
        return index

    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: True))
    monkeypatch.setattr(
        VectorStore,
        "_create_index",
        staticmethod(lambda _dimension: FakeFaissIndex()),
    )
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(write_index=_write_index, read_index=_read_index),
    )

    store = VectorStore(VectorStoreConfig(str(tmp_path)))
    symbol = _symbol("s1", [0.1, 0.2, 0.3])

    store.upsert_vectors([symbol])

    assert store._id_map_dirty is True
    assert store.search([0.1, 0.2, 0.3], k=5)[0][0] == "s1"
    assert not (tmp_path / "vectors.json").exists()

    store.save()

    assert store._id_map_dirty is False
    payload = json.loads((tmp_path / "vectors.json").read_text(encoding="utf8"))
    assert payload["symbol_to_vector_id"] == {"s1": 1}

    reloaded = VectorStore(VectorStoreConfig(str(tmp_path)))
    assert reloaded._id_map_dirty is False
    assert reloaded.search([0.1, 0.2, 0.3], k=5)[0][0] == "s1"


def test_vector_store_resets_legacy_id_map_without_faiss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    cache_dir = Path(tmp_path)
    store = VectorStore(VectorStoreConfig(str(cache_dir)))
    store.upsert_vectors([_symbol("s1", [0.2, 0.3, 0.4])])
    store.save()
    assert store.search([0.2, 0.3, 0.4], k=5)[0][0] == "s1"
    assert (cache_dir / "vectors.json").exists()

    (cache_dir / "vectors.json").write_text(json.dumps(["s1"]), encoding="utf8")
    reloaded = VectorStore(VectorStoreConfig(str(cache_dir)))

    assert reloaded.search([0.2, 0.3, 0.4], k=5) == []
    assert not (cache_dir / "vectors.json").exists()
