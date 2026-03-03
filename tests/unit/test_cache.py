from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing

import pytest

import gloggur.indexer.cache as cache_module
from gloggur.indexer.cache import CACHE_SCHEMA_VERSION, CacheConfig, CacheManager
from gloggur.models import FileMetadata, IndexMetadata, Signal, Symbol


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
        signals=[
            Signal(
                type="code.call",
                payload={"target": "helper"},
                source="test",
            )
        ],
        attributes={"origin": "unit-test"},
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
    assert symbols[0].signals[0].type == "code.call"
    assert symbols[0].attributes == {"origin": "unit-test"}

    metadata = IndexMetadata(version="1", total_symbols=1, indexed_files=1)
    cache.set_index_metadata(metadata)
    loaded = cache.get_index_metadata()
    assert loaded is not None
    assert loaded.version == "1"
    assert loaded.indexed_files == 1

    cache.set_audit_warnings(symbol.id, ["Missing docstring"])
    warnings = cache.get_audit_warnings(symbol.id)
    assert warnings == ["Missing docstring"]


def test_cache_replace_file_index_replaces_symbol_rows_and_metadata() -> None:
    """replace_file_index should atomically swap one file's metadata and symbol rows."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))

    first = _sample_symbol("sample.py:1:add")
    second = _sample_symbol("sample.py:5:subtract")
    second.name = "subtract"
    second.signature = "def subtract(a, b):"
    second.start_line = 5
    second.end_line = 6
    second.embedding_vector = [0.3, 0.4]

    cache.replace_file_index(
        "sample.py",
        FileMetadata(
            path="sample.py",
            language="python",
            content_hash="hash-a",
            symbols=[first.id],
        ),
        [first],
    )
    cache.replace_file_index(
        "sample.py",
        FileMetadata(
            path="sample.py",
            language="python",
            content_hash="hash-b",
            symbols=[second.id],
        ),
        [second],
    )

    metadata = cache.get_file_metadata("sample.py")
    assert metadata is not None
    assert metadata.content_hash == "hash-b"
    assert metadata.symbols == [second.id]
    symbols = cache.list_symbols_for_file("sample.py")
    assert [symbol.id for symbol in symbols] == [second.id]
    assert cache.count_symbols() == 1


def test_cache_replace_file_index_keeps_metadata_for_empty_symbol_files() -> None:
    """replace_file_index should preserve file metadata even when no symbols remain."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))

    cache.replace_file_index(
        "sample.py",
        FileMetadata(path="sample.py", language="python", content_hash="hash-empty", symbols=[]),
        [],
    )

    metadata = cache.get_file_metadata("sample.py")
    assert metadata is not None
    assert metadata.content_hash == "hash-empty"
    assert metadata.symbols == []
    assert cache.list_symbols_for_file("sample.py") == []
    assert cache.count_symbols() == 0


def test_cache_round_trip_structured_audit_reports_and_legacy_warning_reads() -> None:
    """Structured audit payloads should preserve score metadata without breaking warning reads."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))

    symbol = _sample_symbol("sample.py:1:add")
    cache.set_audit_report(
        symbol.id,
        warnings=[],
        semantic_score=0.88,
        score_metadata={"scored": True, "threshold_applied": 0.2},
    )

    warnings = cache.get_audit_warnings(symbol.id)
    reports = cache.list_audit_reports_for_file("sample.py")

    assert warnings == []
    assert reports == [
        (
            symbol.id,
            [],
            0.88,
            {"scored": True, "threshold_applied": 0.2},
        )
    ]


def test_cache_clear_removes_entries() -> None:
    """Ensure cache clear removes all entries."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    cache.upsert_symbols([_sample_symbol()])
    cache.clear()
    assert cache.list_symbols() == []
    assert cache.count_symbols() == 0
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


def test_cache_last_success_resume_markers_round_trip_and_clear() -> None:
    """Last-success resume markers should round-trip and be removed by clear."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    cache.set_last_success_resume_fingerprint("fingerprint-a")
    cache.set_last_success_resume_at("2026-02-26T00:00:00+00:00")
    cache.set_last_success_tool_version("0.1.0")
    assert cache.get_last_success_resume_fingerprint() == "fingerprint-a"
    assert cache.get_last_success_resume_at() == "2026-02-26T00:00:00+00:00"
    assert cache.get_last_success_tool_version() == "0.1.0"
    cache.clear()
    assert cache.get_last_success_resume_fingerprint() is None
    assert cache.get_last_success_resume_at() is None
    assert cache.get_last_success_tool_version() is None


def test_cache_recovers_from_corrupted_db_and_sidecars() -> None:
    """Corrupted DB artifacts should be quarantined/removed and replaced with a healthy DB."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    db_path = os.path.join(cache_dir, "index.db")
    with open(db_path, "wb") as handle:
        handle.write(b"not-a-sqlite-db")
    with open(f"{db_path}-wal", "wb") as handle:
        handle.write(b"wal-garbage")
    with open(f"{db_path}-shm", "wb") as handle:
        handle.write(b"shm-garbage")

    cache = CacheManager(CacheConfig(cache_dir))
    assert cache.last_reset_reason is not None
    assert "cache corruption detected" in cache.last_reset_reason
    assert cache.get_schema_version() == CACHE_SCHEMA_VERSION
    assert not os.path.exists(f"{db_path}-wal")
    assert not os.path.exists(f"{db_path}-shm")
    quarantined = [name for name in os.listdir(cache_dir) if ".corrupt." in name]
    assert any(name.startswith("index.db.corrupt.") for name in quarantined)


def test_cache_recovers_when_integrity_check_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """DatabaseError during integrity probing should force deterministic corruption recovery."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    db_path = os.path.join(cache_dir, "index.db")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE placeholder (id INTEGER PRIMARY KEY)")
        conn.commit()

    def _raise_integrity_error(_self: CacheManager, _conn: sqlite3.Connection) -> str | None:
        raise sqlite3.DatabaseError("integrity probe failed")

    monkeypatch.setattr(CacheManager, "_integrity_issue", _raise_integrity_error)
    cache = CacheManager(CacheConfig(cache_dir))
    assert cache.last_reset_reason is not None
    assert "cache corruption detected" in cache.last_reset_reason
    assert "integrity probe failed" in cache.last_reset_reason
    assert cache.get_schema_version() == CACHE_SCHEMA_VERSION


def test_cache_corruption_recovery_fails_loudly_when_quarantine_and_delete_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If both quarantine and delete fail, cache initialization should raise a clear error."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    db_path = os.path.join(cache_dir, "index.db")
    with open(db_path, "wb") as handle:
        handle.write(b"not-a-sqlite-db")

    def _always_fail_replace(_src: str, _dst: str) -> None:
        raise OSError("replace denied")

    def _always_fail_remove(_path: str) -> None:
        raise OSError("remove denied")

    monkeypatch.setattr(cache_module.os, "replace", _always_fail_replace)
    monkeypatch.setattr(cache_module.os, "remove", _always_fail_remove)
    with pytest.raises(RuntimeError, match="Cache corruption detected but recovery failed"):
        CacheManager(CacheConfig(cache_dir))


def test_cache_healthy_db_does_not_trigger_recovery_or_quarantine() -> None:
    """Inverse case: healthy DB should not be reset or emit quarantine artifacts."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    first = CacheManager(CacheConfig(cache_dir))
    first.set_index_profile("local:model-a")

    second = CacheManager(CacheConfig(cache_dir))
    assert second.last_reset_reason is None
    assert second.get_index_profile() == "local:model-a"
    assert [name for name in os.listdir(cache_dir) if ".corrupt." in name] == []
