from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.indexer import Indexer
from gloggur.parsers.registry import ParserRegistry
from gloggur.search.hybrid_search import HybridSearch
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from scripts.validation.fixtures import TestFixtures

pytest.importorskip("faiss")


def _write_fallback_marker(cache_dir: str) -> None:
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def test_hybrid_search_returns_ranked_results() -> None:
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        config = GloggurConfig(cache_dir=cache_dir, local_embedding_model="local")
        cache = CacheManager(CacheConfig(cache_dir))
        embedding = LocalEmbeddingProvider("local", cache_dir=cache_dir)
        vector_store = VectorStore(VectorStoreConfig(cache_dir))
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=embedding,
            vector_store=vector_store,
        )

        indexer.index_repository(str(repo))

        metadata_store = MetadataStore(MetadataStoreConfig(cache_dir))
        searcher = HybridSearch(embedding, vector_store, metadata_store)
        payload = searcher.search("add", top_k=5)

        assert payload["metadata"]["total_results"] > 0
        first = payload["results"][0]
        assert 0.0 <= first["similarity_score"] <= 1.0
        assert first["file"].endswith("sample.py")
