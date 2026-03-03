from __future__ import annotations

import gloggur.storage.vector_store as vector_store_module
from gloggur.config import GloggurConfig
from gloggur.storage.backends import create_storage_backend


def test_default_storage_backend_creates_vector_and_metadata_stores(
    tmp_path,
    monkeypatch,
) -> None:
    """Default storage backend should expose vector + metadata store factories."""
    monkeypatch.setattr(
        vector_store_module.VectorStore,
        "_check_faiss",
        staticmethod(lambda: False),
    )
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"))
    backend = create_storage_backend(config)
    vector_store = backend.create_vector_store(config.cache_dir)
    metadata_store = backend.create_metadata_store(config.cache_dir)
    assert hasattr(vector_store, "search")
    assert hasattr(metadata_store, "get_symbol")
