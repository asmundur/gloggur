from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing

import pytest

import gloggur.indexer.cache as cache_module
from gloggur.indexer.cache import (
    BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE,
    CACHE_SCHEMA_VERSION,
    CacheConfig,
    CacheManager,
)
from gloggur.models import EdgeRecord, FileMetadata, IndexMetadata, Signal, Symbol, SymbolChunk


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
        [],
        [],
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
        [],
        [],
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
        [],
        [],
    )

    metadata = cache.get_file_metadata("sample.py")
    assert metadata is not None
    assert metadata.content_hash == "hash-empty"
    assert metadata.symbols == []
    assert cache.list_symbols_for_file("sample.py") == []
    assert cache.count_symbols() == 0


def test_cache_chunk_round_trip_preserves_byte_spans() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))

    chunk = SymbolChunk(
        chunk_id="chunk-1",
        symbol_id="symbol-1",
        chunk_part_index=1,
        chunk_part_total=1,
        text="def add(a, b):\n    return a + b\n",
        file_path="sample.py",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=32,
        language="python",
    )
    cache.upsert_chunks([chunk])

    stored = cache.list_chunks()
    assert len(stored) == 1
    assert stored[0].start_byte == 0
    assert stored[0].end_byte == 32


def test_cache_build_file_checkpoints_round_trip_and_delete_individual_rows() -> None:
    """Checkpoint summaries should round-trip without disturbing canonical staged rows."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    source = "def add(a, b):\n    return a + b\n"

    with closing(sqlite3.connect(cache.config.db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "build_file_checkpoints" in tables
    assert "build_checkpoint_files" not in tables
    assert "build_checkpoint_symbols" not in tables
    assert "build_checkpoint_chunks" not in tables
    assert "build_checkpoint_edges" not in tables

    first_symbol = _sample_symbol("sample.py:1:add")
    first_symbol.file_path = "sample.py"
    first_chunk = SymbolChunk(
        chunk_id="sample.py:chunk:1",
        symbol_id=first_symbol.id,
        chunk_part_index=1,
        chunk_part_total=1,
        text=source,
        file_path="sample.py",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=len(source.encode("utf8")),
        language="python",
        embedding_vector=[0.3, 0.4],
    )
    first_edge = EdgeRecord(
        edge_id="sample.py:edge:1",
        edge_type="CALLS",
        from_id=first_symbol.id,
        to_id="helper",
        from_kind="function",
        to_kind="function",
        file_path="sample.py",
        line=2,
        confidence=1.0,
        embedding_vector=[0.5, 0.6],
    )
    cache.replace_file_index(
        "sample.py",
        FileMetadata(
            path="sample.py",
            language="python",
            content_hash="hash-a",
            symbols=[first_symbol.id],
        ),
        [first_symbol],
        [first_chunk],
        [first_edge],
    )

    cache.upsert_build_file_checkpoint(
        path="sample.py",
        state=BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE,
        content_hash="hash-a",
        mtime_ns=11,
        size_bytes=len(source.encode("utf8")),
        language="python",
        symbol_count=1,
        chunk_count=1,
        edge_count=1,
        symbols_added=1,
        symbols_updated=0,
        symbols_removed=0,
    )

    checkpoint = cache.get_build_file_checkpoint("sample.py")
    assert checkpoint is not None
    assert checkpoint.path == "sample.py"
    assert checkpoint.symbol_count == 1
    assert checkpoint.chunk_count == 1
    assert checkpoint.edge_count == 1
    assert cache.get_build_checkpoint_stats() == {
        "extract_completed_files": 1,
        "embedded_completed_files": 0,
        "pending_embed_files": 1,
    }

    second_symbol = _sample_symbol("other.py:1:add")
    second_symbol.file_path = "other.py"
    second_chunk = SymbolChunk(
        chunk_id="other.py:chunk:1",
        symbol_id=second_symbol.id,
        chunk_part_index=1,
        chunk_part_total=1,
        text=source,
        file_path="other.py",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=len(source.encode("utf8")),
        language="python",
    )
    second_edge = EdgeRecord(
        edge_id="other.py:edge:1",
        edge_type="CALLS",
        from_id=second_symbol.id,
        to_id="helper",
        from_kind="function",
        to_kind="function",
        file_path="other.py",
        line=2,
        confidence=1.0,
    )
    cache.replace_file_index(
        "other.py",
        FileMetadata(
            path="other.py",
            language="python",
            content_hash="hash-b",
            symbols=[second_symbol.id],
        ),
        [second_symbol],
        [second_chunk],
        [second_edge],
    )
    cache.upsert_build_file_checkpoint(
        path="other.py",
        state=BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE,
        content_hash="hash-b",
        mtime_ns=22,
        size_bytes=len(source.encode("utf8")),
        language="python",
        symbol_count=1,
        chunk_count=1,
        edge_count=1,
        symbols_added=1,
        symbols_updated=0,
        symbols_removed=0,
    )

    cache.mark_build_file_checkpoint_embedded("sample.py")
    assert cache.get_build_checkpoint_stats() == {
        "extract_completed_files": 2,
        "embedded_completed_files": 1,
        "pending_embed_files": 1,
    }

    cache.delete_build_file_checkpoint("other.py")
    assert cache.get_build_file_checkpoint("sample.py") is not None
    assert cache.get_build_file_checkpoint("other.py") is None
    assert cache.get_file_metadata("other.py") is not None
    assert [symbol.id for symbol in cache.list_symbols_for_file("other.py")] == [second_symbol.id]
    assert cache.get_build_checkpoint_stats() == {
        "extract_completed_files": 1,
        "embedded_completed_files": 1,
        "pending_embed_files": 0,
    }

    cache.clear_build_file_checkpoints()
    assert cache.get_build_checkpoint_stats() == {
        "extract_completed_files": 0,
        "embedded_completed_files": 0,
        "pending_embed_files": 0,
    }
    assert cache.count_files() == 2
    assert cache.get_file_metadata("sample.py") is not None
    assert cache.get_file_metadata("other.py") is not None


def test_cache_round_trip_structured_audit_reports_and_legacy_warning_reads() -> None:
    """Structured audit payloads should preserve score metadata without breaking warning reads."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))

    symbol = _sample_symbol("sample.py:1:add")
    cache.upsert_symbols([symbol])
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


def test_cache_build_state_round_trips_and_flags_stale_cleanup() -> None:
    """Build-state sidecar should round-trip and expose pending cleanup when stale stages exist."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    os.makedirs(cache.build_cache_dir("stale-build"), exist_ok=True)

    cache.write_build_state(
        {
            "state": "building",
            "build_id": "active-build",
            "pid": 123,
            "started_at": "2026-03-07T00:00:00+00:00",
            "updated_at": "2026-03-07T00:00:01+00:00",
            "stage": "embed_chunks",
            "cleanup_pending": False,
        }
    )

    assert cache.get_build_state() == {
        "state": "building",
        "build_id": "active-build",
        "pid": 123,
        "started_at": "2026-03-07T00:00:00+00:00",
        "updated_at": "2026-03-07T00:00:01+00:00",
        "stage": "embed_chunks",
        "cleanup_pending": True,
    }


def test_cache_get_build_state_infers_interrupted_from_staged_build_dir() -> None:
    """A leftover staged build without sidecar metadata should be treated as interrupted."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    os.makedirs(cache.build_cache_dir("orphaned-build"), exist_ok=True)

    assert cache.get_build_state() == {
        "state": "interrupted",
        "build_id": "orphaned-build",
        "pid": None,
        "started_at": None,
        "updated_at": None,
        "stage": None,
        "cleanup_pending": True,
    }


def test_cache_build_state_and_resume_manifest_preserve_extract_progress() -> None:
    """Build-state and staged resume manifests should round-trip extract progress unchanged."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-progress"
    stage_dir = cache.prepare_staged_build(build_id)
    stage_cache = CacheManager(CacheConfig(stage_dir))
    progress = {
        "current_file": "sample.py",
        "subphase": "prepare_file",
        "files_done": 1,
        "files_total": 3,
        "started_at": "2026-03-14T10:00:00+00:00",
        "updated_at": "2026-03-14T10:00:05+00:00",
    }

    cache.write_build_state(
        {
            "state": "interrupted",
            "build_id": build_id,
            "pid": 123,
            "started_at": "2026-03-14T10:00:00+00:00",
            "updated_at": "2026-03-14T10:00:05+00:00",
            "stage": "extract_symbols",
            "cleanup_pending": True,
            "progress": progress,
        }
    )
    manifest = cache.write_staged_build_resume_manifest(
        build_id,
        {
            "build_id": build_id,
            "source": "manifest",
            "workspace_path_hash": "workspace-hash",
            "index_target_path": "/tmp/repo",
            "embedding_profile": "test:test|embed_graph_edges=0",
            "schema_version": stage_cache.get_schema_version(),
            "tool_version": "test-version",
            "stage_cache_dir": stage_dir,
            "state": "interrupted",
            "started_at": "2026-03-14T10:00:00+00:00",
            "updated_at": "2026-03-14T10:00:05+00:00",
            "stage": "extract_symbols",
            "counts": {"files": 1, "symbols": 2, "chunks": 2, "embedded_chunks": 0},
            "progress": progress,
        },
    )

    build_state = cache.get_build_state()
    assert isinstance(build_state, dict)
    assert build_state["progress"] == progress
    assert manifest["progress"] == progress
    assert cache.get_staged_build_resume_manifest(build_id)["progress"] == progress


def test_cache_prepare_and_publish_staged_build_replaces_active_metadata() -> None:
    """Publishing a staged build should atomically swap the active cache artifacts."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    cache.set_index_metadata(IndexMetadata(version="1", total_symbols=1, indexed_files=1))

    stage_dir = cache.prepare_staged_build("build-1")
    staged_cache = CacheManager(CacheConfig(stage_dir))
    staged_cache.set_index_metadata(IndexMetadata(version="2", total_symbols=3, indexed_files=2))

    cache.publish_staged_build("build-1")
    reloaded = CacheManager(CacheConfig(cache_dir))
    metadata = reloaded.get_index_metadata()

    assert metadata is not None
    assert metadata.version == "2"
    assert metadata.total_symbols == 3
    assert not os.path.exists(stage_dir)


def test_cache_auto_resets_legacy_tables() -> None:
    """Legacy table layouts should trigger automatic cache recreation."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    db_path = os.path.join(cache_dir, "index.db")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript("""
            CREATE TABLE validations (
                symbol_id TEXT PRIMARY KEY,
                warnings TEXT NOT NULL
            );
            """)
    cache = CacheManager(CacheConfig(cache_dir))
    assert cache.last_reset_reason is not None
    assert "legacy tables present" in cache.last_reset_reason
    assert cache.get_schema_version() == CACHE_SCHEMA_VERSION


def test_cache_auto_resets_when_chunks_table_missing_byte_columns() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    db_path = os.path.join(cache_dir, "index.db")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript("""
            CREATE TABLE files (
                path TEXT PRIMARY KEY,
                language TEXT,
                content_hash TEXT NOT NULL,
                last_indexed TEXT NOT NULL
            );
            CREATE TABLE symbols (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                fqname TEXT,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                container_id TEXT,
                container_fqname TEXT,
                signature TEXT,
                docstring TEXT,
                body_hash TEXT NOT NULL,
                embedding_vector TEXT,
                language TEXT,
                repo_id TEXT,
                commit_hash TEXT,
                visibility TEXT,
                exported INTEGER,
                tokens_estimate INTEGER,
                invariants TEXT,
                calls TEXT,
                covered_by TEXT,
                is_serialization_boundary INTEGER NOT NULL DEFAULT 0,
                implicit_contract TEXT,
                signals TEXT,
                attributes TEXT
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                symbol_id TEXT NOT NULL,
                chunk_part_index INTEGER NOT NULL,
                chunk_part_total INTEGER NOT NULL,
                text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                tokens_estimate INTEGER,
                language TEXT,
                repo_id TEXT,
                commit_hash TEXT,
                embedding_vector TEXT
            );
            CREATE TABLE edges (
                edge_id TEXT PRIMARY KEY,
                edge_type TEXT NOT NULL,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                from_kind TEXT NOT NULL,
                to_kind TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                confidence REAL NOT NULL,
                repo_id TEXT,
                commit_hash TEXT,
                text TEXT,
                embedding_vector TEXT
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE audits (symbol_id TEXT PRIMARY KEY, warnings TEXT NOT NULL, file_path TEXT);
            CREATE TABLE audit_files (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                last_audited TEXT NOT NULL
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta (key, value) VALUES ('schema_version', '6');
            """)

    cache = CacheManager(CacheConfig(cache_dir))

    assert cache.last_reset_reason is not None
    assert (
        "missing columns" in cache.last_reset_reason
        or "required tables missing" in cache.last_reset_reason
    )
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
