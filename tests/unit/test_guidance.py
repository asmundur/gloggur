from __future__ import annotations

import tempfile

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
from gloggur.models import Symbol

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


class MockMetadataStore:
    def __init__(self, symbols):
        self.symbols = symbols

    def get_symbol(self, symbol_id):
        for s in self.symbols:
            if s.id == symbol_id:
                return s
        return None


class MockHybridSearch:
    def __init__(self, symbols):
        self.metadata_store = MockMetadataStore(symbols)

    def get_semantic_neighborhood(self, symbol_id, **kwargs):
        return {"semantic_neighbors": [], "structural_neighbors": []}


def test_guidance_queries() -> None:
    """AgentGuidance queries should correlate calls and coverage."""
    sym_target = Symbol(
        id="test.py:1:target",
        name="target",
        kind="function",
        file_path="test.py",
        start_line=1,
        end_line=5,
        body_hash="hash",
        covered_by=["test_id_1", "test_id_2"],
        invariants=["x > 0"]
    )
    
    sym_test_1 = Symbol(
        id="test_id_1",
        name="test_foo",
        kind="function",
        file_path="test_test.py",
        start_line=1,
        end_line=5,
        body_hash="hash_t1",
        calls=["target", "other"]
    )
    
    sym_test_2 = Symbol(
        id="test_id_2",
        name="test_bar",
        kind="function",
        file_path="test_test.py",
        start_line=10,
        end_line=15,
        body_hash="hash_t2",
        calls=["other_only"]
    )
    
    sym_untested = Symbol(
        id="untested:1:func",
        name="func",
        kind="function",
        file_path="untested.py",
        start_line=1,
        end_line=5,
        body_hash="hash_un",
        covered_by=[],
        invariants=["y != None"]
    )

    searcher = MockHybridSearch([sym_target, sym_test_1, sym_test_2, sym_untested])
    guidance = AgentGuidance(searcher) # type: ignore

    # Test constraining tests
    res_constraining = guidance.get_constraining_tests("test.py:1:target")
    assert res_constraining["total_constraining_tests"] == 2
    
    tests = res_constraining["constraining_tests"]
    assert isinstance(tests, list)
    t1 = next(t for t in tests if t["test_symbol_id"] == "test_id_1")
    t2 = next(t for t in tests if t["test_symbol_id"] == "test_id_2")
    
    assert t1["constraint_strength"] == "strong"  # calls target
    assert t2["constraint_strength"] == "moderate" # dynamically covers but doesn't call
    
    # Test untested behaviors
    res_untested_target = guidance.get_untested_behaviors("test.py:1:target")
    assert res_untested_target["risk_level"] == "low"
    untested_behaviors_target = res_untested_target["untested_behaviors"]
    assert isinstance(untested_behaviors_target, list)
    assert len(untested_behaviors_target) == 0
    
    res_untested_func = guidance.get_untested_behaviors("untested:1:func")
    assert res_untested_func["risk_level"] == "high"
    untested_behaviors_func = res_untested_func["untested_behaviors"]
    assert isinstance(untested_behaviors_func, list)
    assert len(untested_behaviors_func) == 2 # no coverage + strict invariants
