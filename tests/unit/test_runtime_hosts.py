from __future__ import annotations

import gloggur.storage.vector_store as vector_store_module
from gloggur.config import GloggurConfig
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.parsers.registry import ParserRegistry
from gloggur.runtime.hosts import create_runtime_host
from gloggur.storage.backends import create_storage_backend


def test_default_runtime_host_builds_watch_service(tmp_path, monkeypatch) -> None:
    """Default runtime host should build a watch service with existing dependencies."""
    monkeypatch.setattr(
        vector_store_module.VectorStore,
        "_check_faiss",
        staticmethod(lambda: False),
    )
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"))
    host = create_runtime_host(config)
    cache = CacheManager(CacheConfig(config.cache_dir))
    storage = create_storage_backend(config)
    vector_store = storage.create_vector_store(config.cache_dir)
    parser_registry = ParserRegistry()
    service = host.build_watch_service(
        config=config,
        embedding_provider=None,
        cache=cache,
        vector_store=vector_store,
        parser_registry=parser_registry,
    )
    assert service is not None
