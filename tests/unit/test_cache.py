from __future__ import annotations

import tempfile

from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.models import IndexMetadata, Symbol


def _sample_symbol(symbol_id: str = "sample:1:add") -> Symbol:
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

    cache.set_validation_warnings(symbol.id, ["Missing docstring"])
    warnings = cache.get_validation_warnings(symbol.id)
    assert warnings == ["Missing docstring"]


def test_cache_clear_removes_entries() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    cache.upsert_symbols([_sample_symbol()])
    cache.clear()
    assert cache.list_symbols() == []
    assert cache.get_index_metadata() is None
