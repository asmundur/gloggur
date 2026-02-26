from __future__ import annotations

import tempfile

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.indexer import Indexer
from gloggur.parsers.registry import ParserRegistry
from scripts.verification.fixtures import TestFixtures


def test_indexer_indexes_repo_and_sets_metadata() -> None:
    """Indexer should index repo and set metadata."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        result = indexer.index_repository(str(repo))

        assert result.indexed_files == 1
        assert result.files_changed == 1
        assert result.files_removed == 0
        assert result.symbols_added > 0
        assert result.symbols_updated == 0
        assert result.symbols_removed == 0
        assert result.indexed_symbols > 0
        assert result.skipped_files == 0
        payload = result.as_payload()
        for field in (
            "files_scanned",
            "files_changed",
            "files_removed",
            "symbols_added",
            "symbols_updated",
            "symbols_removed",
        ):
            assert field in payload
        metadata = cache.get_index_metadata()
        assert metadata is not None
        assert metadata.indexed_files == 1
        assert metadata.total_symbols == len(cache.list_symbols())


def test_indexer_skips_unchanged_files() -> None:
    """Indexer should skip unchanged files on reindex."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        first = indexer.index_file(str(repo / "sample.py"))
        second = indexer.index_file(str(repo / "sample.py"))

        assert first is not None
        assert second is None
        metadata = cache.get_file_metadata(str(repo / "sample.py"))
        assert metadata is not None
        assert metadata.content_hash


def test_indexer_prunes_deleted_files_and_reports_symbol_removals() -> None:
    """Reindex should remove stale file symbols when files are deleted from disk."""
    keep_source = (
        "def keep_me(value: int) -> int:\n"
        "    return value + 1\n"
    )
    drop_source = (
        "def drop_me(value: int) -> int:\n"
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": keep_source, "drop.py": drop_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        first = indexer.index_repository(str(repo))
        assert first.indexed_files == 2
        assert first.files_removed == 0

        drop_path = repo / "drop.py"
        drop_metadata = cache.get_file_metadata(str(drop_path))
        assert drop_metadata is not None
        removed_symbol_count = len(drop_metadata.symbols)
        assert removed_symbol_count > 0

        drop_path.unlink()
        second = indexer.index_repository(str(repo))

        assert second.indexed_files == 0
        assert second.skipped_files == 1
        assert second.files_removed == 1
        assert second.symbols_removed == removed_symbol_count
        assert cache.get_file_metadata(str(drop_path)) is None
        assert cache.list_symbols_for_file(str(drop_path)) == []


def test_indexer_surfaces_stale_cleanup_failures_with_deterministic_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup failure for stale files should not report success and should include remediation."""
    keep_source = (
        "def keep_me(value: int) -> int:\n"
        "    return value + 1\n"
    )
    drop_source = (
        "def drop_me(value: int) -> int:\n"
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": keep_source, "drop.py": drop_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        first = indexer.index_repository(str(repo))
        assert first.failed == 0

        stale_path = str(repo / "drop.py")
        (repo / "drop.py").unlink()
        original_delete = cache.delete_file_metadata

        def _raise_on_stale_delete(path: str) -> None:
            if path == stale_path:
                raise OSError("simulated stale cleanup failure")
            original_delete(path)

        monkeypatch.setattr(cache, "delete_file_metadata", _raise_on_stale_delete)
        second = indexer.index_repository(str(repo))

        assert second.failed == 1
        assert second.failed_reasons == {"stale_cleanup_error": 1}
        assert any(stale_path in sample for sample in second.failed_samples)
        payload = second.as_payload()
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert payload["failure_codes"] == ["stale_cleanup_error"]
        assert "stale_cleanup_error" in guidance
        assert isinstance(guidance["stale_cleanup_error"], list)
        assert guidance["stale_cleanup_error"]


def test_indexer_detects_vector_metadata_mismatch_under_symbol_removal() -> None:
    """Vector/cache divergence should fail closed with a stable mismatch reason code."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    class FaultyVectorStore:
        def __init__(self) -> None:
            self._ids: set[str] = set()

        def remove_ids(self, _symbol_ids: list[str]) -> None:
            # Intentional no-op to emulate stale vectors surviving symbol removals.
            return

        def upsert_vectors(self, symbols: list[object]) -> None:
            for symbol in symbols:
                symbol_id = getattr(symbol, "id", None)
                if isinstance(symbol_id, str):
                    self._ids.add(symbol_id)

        def list_symbol_ids(self) -> list[str]:
            return sorted(self._ids)

        def save(self) -> None:
            return

    source_with_two = (
        "def alpha(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def beta(value: int) -> int:\n"
        "    return value + 2\n"
    )
    source_with_one = (
        "def alpha(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source_with_two})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        vector_store = FaultyVectorStore()
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=vector_store,
        )

        first = indexer.index_repository(str(repo))
        assert first.failed == 0

        (repo / "sample.py").write_text(source_with_one, encoding="utf8")
        second = indexer.index_repository(str(repo))

        assert second.failed == 1
        assert second.failed_reasons == {"vector_metadata_mismatch": 1}
        assert any("vector/cache mismatch" in sample for sample in second.failed_samples)
        payload = second.as_payload()
        guidance = payload["failure_guidance"]
        assert payload["failure_codes"] == ["vector_metadata_mismatch"]
        assert isinstance(guidance, dict)
        assert "vector_metadata_mismatch" in guidance
        assert guidance["vector_metadata_mismatch"]


def test_indexer_fails_closed_when_vector_consistency_is_unverifiable() -> None:
    """Indexing should not report success when vector/cache consistency cannot be verified."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    class OpaqueVectorStore:
        def remove_ids(self, _symbol_ids: list[str]) -> None:
            return

        def upsert_vectors(self, _symbols: list[object]) -> None:
            return

        def save(self) -> None:
            return

    source = (
        "def alpha(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=OpaqueVectorStore(),
        )

        result = indexer.index_repository(str(repo))

        assert result.failed == 1
        assert result.failed_reasons == {"vector_consistency_unverifiable": 1}
        assert any("list_symbol_ids" in sample for sample in result.failed_samples)
        payload = result.as_payload()
        assert payload["failure_codes"] == ["vector_consistency_unverifiable"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "vector_consistency_unverifiable" in guidance
        assert isinstance(guidance["vector_consistency_unverifiable"], list)
        assert guidance["vector_consistency_unverifiable"]
