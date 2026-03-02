from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

from gloggur.config import GloggurConfig
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.indexer import Indexer
from gloggur.parsers.registry import ParserRegistry
from gloggur.search.guidance import AgentGuidance
from gloggur.search.hybrid_search import HybridSearch
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from scripts.verification.fixtures import TestFixtures
from tests.unit.test_search import _write_fallback_marker


def test_agent_guidance_generates_context() -> None:
    """AgentGuidance should generate a valid context payload."""
    source = """
def func_a(): return 1
def func_b(): return 2
"""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"source.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        config = GloggurConfig(cache_dir=cache_dir, local_embedding_model="local")
        cache = CacheManager(CacheConfig(cache_dir))
        embedding = LocalEmbeddingProvider("local", cache_dir=cache_dir)
        vector_store = VectorStore(VectorStoreConfig(cache_dir))
        metadata_store = MetadataStore(MetadataStoreConfig(cache_dir))

        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=embedding,
            vector_store=vector_store,
        )
        indexer.index_repository(str(repo))

        searcher = HybridSearch(embedding, vector_store, metadata_store)
        guidance = AgentGuidance(searcher)

        symbols = metadata_store.list_symbols()
        assert len(symbols) == 2
        func_a_id = next(s.id for s in symbols if s.name == "func_a")

        payload = guidance.generate_agent_context(func_a_id)
        assert "error" not in payload
        assert payload["symbol_id"] == func_a_id
        assert payload["name"] == "func_a"

        impact = payload["change_impact"]
        assert impact["target_symbol_id"] == func_a_id
        # func_b is a structural neighbor
        assert impact["estimated_impact_count"] == 1
