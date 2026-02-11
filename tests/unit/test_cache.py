from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing

from gloggur.indexer.cache import CACHE_SCHEMA_VERSION, CacheConfig, CacheManager
from gloggur.models import IndexMetadata, Symbol


def _sample_symbol(symbol_id: str = "sample:1:add") -> Symbol:
    """Create a sample symbol for cache tests."""
    return Symbol(
        id=symbol_id,
        name="add",
        kind="function",
        file_path="sample.py",
        start_line=1,
        end_line=2,
        signature="def add(a, b):",
        docstring="Add numbers.",
        body_hash="abc123",
        embedding_vector=[0.1, 0.2],
        language="python",
    )


def test_cache_round_trip_symbols_metadata_and_warnings() -> None:
    """Ensure cache round-trips symbols, metadata, and warnings."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))

    symbol = _sample_symbol()
    cache.upsert_symbols([symbol])
    symbols = cache.list_symbols()
    assert len(symbols) == 1
    assert symbols[0].id == symbol.id

    metadata = IndexMetadata(version="1", total_symbols=1, indexed_files=1)
    cache.set_index_metadata(metadata)
    loaded = cache.get_index_metadata()
    assert loaded is not None
    assert loaded.version == "1"
    assert loaded.indexed_files == 1

    cache.set_audit_warnings(symbol.id, ["Missing docstring"])
    warnings = cache.get_audit_warnings(symbol.id)
    assert warnings == ["Missing docstring"]


def test_cache_clear_removes_entries() -> None:
    """Ensure cache clear removes all entries."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    cache.upsert_symbols([_sample_symbol()])
    cache.clear()
    assert cache.list_symbols() == []
    assert cache.get_index_metadata() is None


def test_cache_schema_version_persists_across_clear() -> None:
    """Schema version marker should remain after clear for future compatibility checks."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    cache.clear()
    assert cache.get_schema_version() == CACHE_SCHEMA_VERSION


def test_cache_auto_resets_legacy_tables() -> None:
    """Legacy table layouts should trigger automatic cache recreation."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    db_path = os.path.join(cache_dir, "index.db")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE validations (
                symbol_id TEXT PRIMARY KEY,
                warnings TEXT NOT NULL
            );
            """
        )
    cache = CacheManager(CacheConfig(cache_dir))
    assert cache.last_reset_reason is not None
    assert "legacy tables present" in cache.last_reset_reason
    assert cache.get_schema_version() == CACHE_SCHEMA_VERSION


def test_cache_index_profile_round_trip_and_clear() -> None:
    """Index profile should round-trip and clear should remove stale profile markers."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    cache.set_index_profile("local:model-a")
    assert cache.get_index_profile() == "local:model-a"
    cache.clear()
    assert cache.get_index_profile() is None
