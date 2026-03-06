from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from gloggur.io_failures import wrap_io_error
from gloggur.models import (
    AuditFileMetadata,
    EdgeRecord,
    FileMetadata,
    IndexMetadata,
    Symbol,
    SymbolChunk,
)

SCHEMA_VERSION_KEY = "schema_version"
INDEX_PROFILE_KEY = "index_profile"
LAST_SUCCESS_RESUME_FINGERPRINT_KEY = "last_success_resume_fingerprint"
LAST_SUCCESS_RESUME_AT_KEY = "last_success_resume_at"
LAST_SUCCESS_TOOL_VERSION_KEY = "last_success_tool_version"
CACHE_SCHEMA_VERSION = "7"
SQLITE_BUSY_TIMEOUT_MS = 5_000
SQLITE_CONNECT_TIMEOUT_SECONDS = SQLITE_BUSY_TIMEOUT_MS / 1000
SQLITE_JOURNAL_MODE = "WAL"
SQLITE_SYNCHRONOUS = "NORMAL"

REQUIRED_TABLES = {
    "files",
    "symbols",
    "chunks",
    "edges",
    "metadata",
    "audits",
    "audit_files",
    "meta",
}
LEGACY_TABLES = {"validations", "validation_files"}
REQUIRED_COLUMNS = {
    "files": {"path", "language", "content_hash", "last_indexed"},
    "symbols": {
        "id",
        "name",
        "kind",
        "fqname",
        "file_path",
        "start_line",
        "end_line",
        "container_id",
        "container_fqname",
        "signature",
        "docstring",
        "body_hash",
        "embedding_vector",
        "language",
        "repo_id",
        "commit_hash",
        "visibility",
        "exported",
        "tokens_estimate",
        "invariants",
        "calls",
        "covered_by",
        "is_serialization_boundary",
        "implicit_contract",
        "signals",
        "attributes",
    },
    "chunks": {
        "chunk_id",
        "symbol_id",
        "chunk_part_index",
        "chunk_part_total",
        "text",
        "file_path",
        "start_line",
        "end_line",
        "start_byte",
        "end_byte",
        "tokens_estimate",
        "language",
        "repo_id",
        "commit_hash",
        "embedding_vector",
    },
    "edges": {
        "edge_id",
        "edge_type",
        "from_id",
        "to_id",
        "from_kind",
        "to_kind",
        "file_path",
        "line",
        "confidence",
        "repo_id",
        "commit_hash",
        "text",
        "embedding_vector",
    },
    "metadata": {"key", "value"},
    "audits": {"symbol_id", "warnings", "file_path"},
    "audit_files": {"path", "content_hash", "last_audited"},
    "meta": {"key", "value"},
}


@dataclass(frozen=True)
class _ResetPlan:
    """Plan describing why and how the cache database should be reset."""

    reason: str
    corruption_detected: bool = False


@dataclass
class CacheConfig:
    """Configuration for the on-disk cache (cache dir and db path)."""

    cache_dir: str

    @property
    def db_path(self) -> str:
        """Return the path to the SQLite index database."""
        return os.path.join(self.cache_dir, "index.db")


class CacheManager:
    """SQLite-backed cache for symbols, files, and audits."""

    def __init__(self, config: CacheConfig) -> None:
        """Initialize the cache and ensure the database exists."""
        self.config = config
        self.last_reset_reason: str | None = None
        try:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="create cache directory",
                path=self.config.cache_dir,
            ) from exc
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for a transactional database connection."""
        try:
            conn = sqlite3.connect(
                self.config.db_path,
                timeout=SQLITE_CONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, sqlite3.DatabaseError) as exc:
            raise wrap_io_error(
                exc,
                operation="open cache database connection",
                path=self.config.db_path,
            ) from exc
        conn.row_factory = sqlite3.Row
        try:
            self._configure_connection(conn)
        except Exception:
            conn.close()
            raise
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            except sqlite3.OperationalError:
                pass
            if isinstance(exc, (OSError, sqlite3.DatabaseError)):
                raise wrap_io_error(
                    exc,
                    operation="execute cache database transaction",
                    path=self.config.db_path,
                ) from exc
            raise
        finally:
            conn.close()

    def _configure_connection(
        self,
        conn: sqlite3.Connection,
        *,
        wrap_errors: bool = True,
    ) -> None:
        """Apply SQLite pragmas for safer concurrent reader/writer behavior."""
        try:
            conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
            conn.execute(f"PRAGMA journal_mode = {SQLITE_JOURNAL_MODE}")
            conn.execute(f"PRAGMA synchronous = {SQLITE_SYNCHRONOUS}")
        except sqlite3.DatabaseError as exc:
            if not wrap_errors:
                raise
            raise wrap_io_error(
                exc,
                operation="configure cache database pragmas",
                path=self.config.db_path,
            ) from exc

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        reset_plan = self._schema_reset_plan()
        if reset_plan:
            self._reset_database(reset_plan)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    language TEXT,
                    content_hash TEXT NOT NULL,
                    last_indexed TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS symbols (
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
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    symbol_id TEXT NOT NULL,
                    chunk_part_index INTEGER NOT NULL,
                    chunk_part_total INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    start_byte INTEGER NOT NULL,
                    end_byte INTEGER NOT NULL,
                    tokens_estimate INTEGER,
                    language TEXT,
                    repo_id TEXT,
                    commit_hash TEXT,
                    embedding_vector TEXT
                );
                CREATE TABLE IF NOT EXISTS edges (
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
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audits (
                    symbol_id TEXT PRIMARY KEY,
                    warnings TEXT NOT NULL,
                    file_path TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_files (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    last_audited TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_symbols_fqname ON symbols (fqname);
                CREATE INDEX IF NOT EXISTS idx_symbols_file_path ON symbols (file_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_symbol_id ON chunks (symbol_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks (file_path);
                CREATE INDEX IF NOT EXISTS idx_edges_from_id ON edges (from_id);
                CREATE INDEX IF NOT EXISTS idx_edges_to_id ON edges (to_id);
                CREATE INDEX IF NOT EXISTS idx_edges_type ON edges (edge_type);
                CREATE INDEX IF NOT EXISTS idx_edges_file_path ON edges (file_path);
                """)
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION_KEY, CACHE_SCHEMA_VERSION),
            )

    def get_file_metadata(self, path: str) -> FileMetadata | None:
        """Return cached metadata for a file."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
            if not row:
                return None
            symbols = self._list_symbol_ids(conn, path)
            return FileMetadata(
                path=row["path"],
                language=row["language"],
                content_hash=row["content_hash"],
                last_indexed=datetime.fromisoformat(row["last_indexed"]),
                symbols=symbols,
            )

    def upsert_file_metadata(self, metadata: FileMetadata) -> None:
        """Insert or update file metadata."""
        with self._connect() as conn:
            self._upsert_file_metadata_row(conn, metadata)

    def delete_symbols_for_file(self, path: str) -> None:
        """Delete all cached symbols for a file."""
        with self._connect() as conn:
            conn.execute("DELETE FROM symbols WHERE file_path = ?", (path,))

    def delete_chunks_for_file(self, path: str) -> None:
        """Delete all cached chunk rows for a file."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (path,))

    def delete_file_metadata(self, path: str) -> None:
        """Delete cached file metadata for a path."""
        with self._connect() as conn:
            conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def upsert_symbols(self, symbols: Iterable[Symbol]) -> None:
        """Insert or update a batch of symbols."""
        symbol_rows = [self._symbol_row(symbol) for symbol in symbols]
        if not symbol_rows:
            return
        with self._connect() as conn:
            self._upsert_symbol_rows(conn, symbol_rows)

    def upsert_chunks(self, chunks: Iterable[SymbolChunk]) -> None:
        """Insert or update a batch of symbol chunks."""
        chunk_rows = [self._chunk_row(chunk) for chunk in chunks]
        if not chunk_rows:
            return
        with self._connect() as conn:
            self._upsert_chunk_rows(conn, chunk_rows)

    def replace_file_index(
        self,
        path: str,
        metadata: FileMetadata,
        symbols: Iterable[Symbol],
        chunks: Iterable[SymbolChunk],
        edges: Iterable[EdgeRecord],
    ) -> None:
        """Replace one file's symbols/chunks/edges and metadata in a single transaction."""
        symbol_rows = [self._symbol_row(symbol) for symbol in symbols]
        chunk_rows = [self._chunk_row(chunk) for chunk in chunks]
        edge_rows = [self._edge_row(edge) for edge in edges]
        with self._connect() as conn:
            conn.execute("DELETE FROM symbols WHERE file_path = ?", (path,))
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (path,))
            conn.execute("DELETE FROM edges WHERE file_path = ?", (path,))
            if symbol_rows:
                self._upsert_symbol_rows(conn, symbol_rows)
            if chunk_rows:
                self._upsert_chunk_rows(conn, chunk_rows)
            if edge_rows:
                self._upsert_edge_rows(conn, edge_rows)
            self._upsert_file_metadata_row(conn, metadata)

    def list_symbols(self) -> list[Symbol]:
        """Return all cached symbols."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM symbols").fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def count_files(self) -> int:
        """Return the number of indexed files."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM files").fetchone()
            return int(row["count"] if row else 0)

    def count_symbols(self) -> int:
        """Return the number of cached symbols."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM symbols").fetchone()
            return int(row["count"] if row else 0)

    def list_file_paths(self) -> list[str]:
        """Return all indexed file paths in deterministic order."""
        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM files ORDER BY path").fetchall()
            return [str(row["path"]) for row in rows]

    def list_symbols_for_file(self, path: str) -> list[Symbol]:
        """Return cached symbols for a file path."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE file_path = ? ORDER BY start_line", (path,)
            ).fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def list_chunks_for_file(self, path: str) -> list[SymbolChunk]:
        """Return cached symbol chunks for a file path."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE file_path = ?
                ORDER BY start_line, chunk_part_index, chunk_id
                """,
                (path,),
            ).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def list_chunks(self) -> list[SymbolChunk]:
        """Return all cached symbol chunks."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks ORDER BY file_path, start_line, chunk_part_index, chunk_id"
            ).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def list_chunks_for_symbol(self, symbol_id: str) -> list[SymbolChunk]:
        """Return chunk rows for a symbol id."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE symbol_id = ?
                ORDER BY chunk_part_index, chunk_id
                """,
                (symbol_id,),
            ).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def upsert_edges(self, edges: Iterable[EdgeRecord]) -> None:
        """Insert or update a batch of edge records."""
        edge_rows = [self._edge_row(edge) for edge in edges]
        if not edge_rows:
            return
        with self._connect() as conn:
            self._upsert_edge_rows(conn, edge_rows)

    def replace_edges_for_file(self, path: str, edges: Iterable[EdgeRecord]) -> None:
        """Replace all edges observed in one file atomically."""
        edge_rows = [self._edge_row(edge) for edge in edges]
        with self._connect() as conn:
            conn.execute("DELETE FROM edges WHERE file_path = ?", (path,))
            if edge_rows:
                self._upsert_edge_rows(conn, edge_rows)

    def delete_edges_for_file(self, path: str) -> None:
        """Delete edge rows observed in a file."""
        with self._connect() as conn:
            conn.execute("DELETE FROM edges WHERE file_path = ?", (path,))

    def list_edges(self) -> list[EdgeRecord]:
        """Return all edge records in deterministic order."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges ORDER BY file_path, line, edge_type, edge_id"
            ).fetchall()
            return [self._row_to_edge(row) for row in rows]

    def list_edges_for_file(self, path: str) -> list[EdgeRecord]:
        """Return edge records emitted for one file path."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM edges
                WHERE file_path = ?
                ORDER BY line, edge_type, edge_id
                """,
                (path,),
            ).fetchall()
            return [self._row_to_edge(row) for row in rows]

    def list_edges_for_symbol(
        self,
        symbol_id: str,
        *,
        direction: str = "both",
        edge_type: str | None = None,
        limit: int | None = None,
    ) -> list[EdgeRecord]:
        """Return incoming/outgoing/bidirectional edges for a symbol id."""
        clauses = []
        params: list[object] = []
        if direction == "incoming":
            clauses.append("to_id = ?")
            params.append(symbol_id)
        elif direction == "outgoing":
            clauses.append("from_id = ?")
            params.append(symbol_id)
        else:
            clauses.append("(from_id = ? OR to_id = ?)")
            params.extend([symbol_id, symbol_id])
        if edge_type:
            clauses.append("edge_type = ?")
            params.append(edge_type)
        where = " AND ".join(clauses) if clauses else "1=1"
        query = (
            "SELECT * FROM edges "
            f"WHERE {where} "
            "ORDER BY confidence DESC, file_path, line, edge_id"
        )
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_edge(row) for row in rows]

    def get_index_metadata(self) -> IndexMetadata | None:
        """Return index-level metadata, if present."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", ("index",)).fetchone()
            if not row:
                return None
            payload = json.loads(row["value"])
            return IndexMetadata(**payload)

    def set_index_metadata(self, metadata: IndexMetadata) -> None:
        """Persist index-level metadata."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metadata (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("index", metadata.model_dump_json()),
            )

    def delete_index_metadata(self) -> None:
        """Remove index-level metadata (used while a rebuild is in progress)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM metadata WHERE key = ?", ("index",))

    def get_schema_version(self) -> str | None:
        """Return the cache schema version marker."""
        with self._connect() as conn:
            return self._read_meta_value(conn, SCHEMA_VERSION_KEY)

    def get_index_profile(self) -> str | None:
        """Return the cached index profile marker."""
        with self._connect() as conn:
            return self._read_meta_value(conn, INDEX_PROFILE_KEY)

    def set_index_profile(self, profile: str) -> None:
        """Persist the active index profile marker."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (INDEX_PROFILE_KEY, profile),
            )

    def get_last_success_resume_fingerprint(self) -> str | None:
        """Return the cached last-success resume fingerprint marker."""
        with self._connect() as conn:
            return self._read_meta_value(conn, LAST_SUCCESS_RESUME_FINGERPRINT_KEY)

    def set_last_success_resume_fingerprint(self, fingerprint: str) -> None:
        """Persist the last-success resume fingerprint marker."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (LAST_SUCCESS_RESUME_FINGERPRINT_KEY, fingerprint),
            )

    def get_last_success_resume_at(self) -> str | None:
        """Return the last-success resume timestamp marker."""
        with self._connect() as conn:
            return self._read_meta_value(conn, LAST_SUCCESS_RESUME_AT_KEY)

    def set_last_success_resume_at(self, timestamp: str) -> None:
        """Persist the last-success resume timestamp marker."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (LAST_SUCCESS_RESUME_AT_KEY, timestamp),
            )

    def get_last_success_tool_version(self) -> str | None:
        """Return the last-success tool-version marker."""
        with self._connect() as conn:
            return self._read_meta_value(conn, LAST_SUCCESS_TOOL_VERSION_KEY)

    def set_last_success_tool_version(self, version: str) -> None:
        """Persist the last-success tool-version marker."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (LAST_SUCCESS_TOOL_VERSION_KEY, version),
            )

    def set_audit_warnings(
        self,
        symbol_id: str,
        warnings: list[str],
        *,
        file_path: str | None = None,
    ) -> None:
        """Store audit warnings for a symbol."""
        payload = self._serialize_audit_payload(warnings)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audits (symbol_id, warnings, file_path) VALUES (?, ?, ?)
                ON CONFLICT(symbol_id) DO UPDATE SET
                    warnings = excluded.warnings,
                    file_path = COALESCE(excluded.file_path, audits.file_path)
                """,
                (symbol_id, payload, file_path),
            )

    def set_audit_report(
        self,
        symbol_id: str,
        *,
        warnings: list[str],
        file_path: str | None = None,
        semantic_score: float | None = None,
        score_metadata: dict[str, object] | None = None,
    ) -> None:
        """Store a structured audit report for a symbol."""
        payload = self._serialize_audit_payload(
            warnings,
            semantic_score=semantic_score,
            score_metadata=score_metadata,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audits (symbol_id, warnings, file_path) VALUES (?, ?, ?)
                ON CONFLICT(symbol_id) DO UPDATE SET
                    warnings = excluded.warnings,
                    file_path = COALESCE(excluded.file_path, audits.file_path)
                """,
                (symbol_id, payload, file_path),
            )

    def get_audit_warnings(self, symbol_id: str) -> list[str]:
        """Fetch audit warnings for a symbol."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT warnings FROM audits WHERE symbol_id = ?", (symbol_id,)
            ).fetchone()
            if not row:
                return []
            warnings, _, _ = self._deserialize_audit_payload(row["warnings"])
            return warnings

    def list_audit_reports_for_file(
        self,
        path: str,
    ) -> list[tuple[str, list[str], float | None, dict[str, object] | None]]:
        """Fetch cached audit report payloads for one file path."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT audits.symbol_id, audits.warnings
                FROM audits
                WHERE audits.file_path = ?
                UNION
                SELECT audits.symbol_id, audits.warnings
                FROM audits
                JOIN symbols ON symbols.id = audits.symbol_id
                WHERE symbols.file_path = ?
                ORDER BY symbol_id
                """,
                (path, path),
            ).fetchall()
            reports: list[tuple[str, list[str], float | None, dict[str, object] | None]] = []
            for row in rows:
                warnings, semantic_score, score_metadata = self._deserialize_audit_payload(
                    row["warnings"]
                )
                reports.append((row["symbol_id"], warnings, semantic_score, score_metadata))
            return reports

    def list_audit_warnings_for_file(self, path: str) -> list[tuple[str, list[str]]]:
        """Fetch cached audit warnings for all symbols belonging to one file path."""
        return [
            (symbol_id, warnings)
            for symbol_id, warnings, _, _ in self.list_audit_reports_for_file(path)
        ]

    def delete_audit_reports_for_file(self, path: str) -> None:
        """Delete cached audit report rows for one file path."""
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM audits
                WHERE file_path = ?
                    OR symbol_id IN (
                    SELECT id FROM symbols WHERE file_path = ?
                )
                """,
                (path, path),
            )

    def get_audit_file_metadata(self, path: str) -> AuditFileMetadata | None:
        """Return cached audit metadata for a file."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM audit_files WHERE path = ?", (path,)).fetchone()
            if not row:
                return None
            return AuditFileMetadata(
                path=row["path"],
                content_hash=row["content_hash"],
                last_audited=datetime.fromisoformat(row["last_audited"]),
            )

    def upsert_audit_file_metadata(self, metadata: AuditFileMetadata) -> None:
        """Insert or update audit metadata for a file."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_files (path, content_hash, last_audited)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    last_audited = excluded.last_audited
                """,
                (
                    metadata.path,
                    metadata.content_hash,
                    metadata.last_audited.isoformat(),
                ),
            )

    def clear(self) -> None:
        """Clear all cached data."""
        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM audits;
                DELETE FROM audit_files;
                DELETE FROM edges;
                DELETE FROM chunks;
                DELETE FROM symbols;
                DELETE FROM files;
                DELETE FROM metadata;
                """)
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION_KEY, CACHE_SCHEMA_VERSION),
            )
            conn.execute("DELETE FROM meta WHERE key = ?", (INDEX_PROFILE_KEY,))
            conn.execute(
                "DELETE FROM meta WHERE key = ?",
                (LAST_SUCCESS_RESUME_FINGERPRINT_KEY,),
            )
            conn.execute(
                "DELETE FROM meta WHERE key = ?",
                (LAST_SUCCESS_RESUME_AT_KEY,),
            )
            conn.execute(
                "DELETE FROM meta WHERE key = ?",
                (LAST_SUCCESS_TOOL_VERSION_KEY,),
            )

    def _schema_reset_plan(self) -> _ResetPlan | None:
        """Return a reset plan if schema is incompatible or DB corruption is detected."""
        if not os.path.exists(self.config.db_path):
            return None

        try:
            conn = sqlite3.connect(
                self.config.db_path,
                timeout=SQLITE_CONNECT_TIMEOUT_SECONDS,
            )
            try:
                self._configure_connection(conn, wrap_errors=False)
                integrity_issue = self._integrity_issue(conn)
                if integrity_issue:
                    return _ResetPlan(
                        reason=f"cache corruption detected ({integrity_issue})",
                        corruption_detected=True,
                    )

                existing_tables = self._list_tables(conn)
                legacy_tables = sorted(existing_tables & LEGACY_TABLES)
                if legacy_tables:
                    return _ResetPlan(reason=f"legacy tables present ({', '.join(legacy_tables)})")

                missing_tables = sorted(REQUIRED_TABLES - existing_tables)
                if missing_tables:
                    return _ResetPlan(
                        reason=f"required tables missing ({', '.join(missing_tables)})"
                    )

                for table, expected_columns in REQUIRED_COLUMNS.items():
                    existing_columns = self._list_columns(conn, table)
                    missing_columns = sorted(expected_columns - existing_columns)
                    if missing_columns:
                        return _ResetPlan(
                            reason=(
                                f"table '{table}' missing columns ({', '.join(missing_columns)})"
                            )
                        )

                version = self._read_schema_version(conn)
                if version != CACHE_SCHEMA_VERSION:
                    found = version if version is not None else "none"
                    return _ResetPlan(
                        reason=(
                            "schema version mismatch "
                            f"(found {found}, expected {CACHE_SCHEMA_VERSION})"
                        )
                    )
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            raise wrap_io_error(
                exc,
                operation="open cache database for schema validation",
                path=self.config.db_path,
            ) from exc
        except sqlite3.DatabaseError as exc:
            return _ResetPlan(
                reason=f"cache corruption detected (sqlite open/integrity error: {exc})",
                corruption_detected=True,
            )

        return None

    def _reset_database(self, reset_plan: _ResetPlan) -> None:
        """Reset incompatible or corrupted SQLite artifacts."""
        self.last_reset_reason = reset_plan.reason
        if reset_plan.corruption_detected:
            self._recover_from_corruption(reset_plan.reason)
            return
        self._delete_database_artifacts()
        rebuild_notice = (
            "Cache schema changed; rebuilding cache at "
            f"{self.config.db_path} ({reset_plan.reason}).\n"
        )
        sys.stderr.write(rebuild_notice)

    def _integrity_issue(self, conn: sqlite3.Connection) -> str | None:
        """Return an integrity issue detail if `PRAGMA integrity_check` reports corruption."""
        rows = conn.execute("PRAGMA integrity_check(1)").fetchall()
        messages = [str(row[0]) for row in rows if row and row[0] is not None]
        if not messages:
            return "integrity_check returned no rows"
        if len(messages) == 1 and messages[0].strip().lower() == "ok":
            return None
        preview = ", ".join(messages[:3])
        if len(messages) > 3:
            preview += ", ..."
        return f"integrity_check failed: {preview}"

    def _recover_from_corruption(self, reason: str) -> None:
        """Quarantine corrupted artifacts when possible, then remove sidecars and rebuild."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        db_path = self.config.db_path
        quarantined_db = self._quarantine_or_remove(db_path, timestamp)
        self._quarantine_or_remove(f"{db_path}-wal", timestamp)
        self._quarantine_or_remove(f"{db_path}-shm", timestamp)
        if quarantined_db:
            action = f"quarantined to {quarantined_db}"
        else:
            action = "removed corrupted artifact"
        sys.stderr.write(
            f"Cache corruption detected at {db_path}; {action}; rebuilding cache ({reason}).\n"
        )

    def _delete_database_artifacts(self) -> None:
        """Delete primary SQLite database and sidecars."""
        db_path = self.config.db_path
        for path in (db_path, f"{db_path}-wal", f"{db_path}-shm"):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    raise wrap_io_error(
                        exc,
                        operation="delete cache database artifact",
                        path=path,
                    ) from exc

    def _quarantine_or_remove(self, path: str, timestamp: str) -> str | None:
        """Move an artifact aside with a .corrupt suffix; fall back to deletion if rename fails."""
        if not os.path.exists(path):
            return None

        suffix = f".corrupt.{timestamp}"
        target = f"{path}{suffix}"
        attempt = 1
        while os.path.exists(target):
            attempt += 1
            target = f"{path}{suffix}.{attempt}"

        try:
            os.replace(path, target)
            return target
        except OSError as replace_error:
            try:
                os.remove(path)
                return None
            except OSError as remove_error:
                raise CacheRecoveryError(
                    "Cache corruption detected but recovery failed for "
                    f"{path}: rename failed ({replace_error}); delete failed ({remove_error}). "
                    "Fix permissions and remove corrupted cache files manually."
                ) from remove_error

    def _list_tables(self, conn: sqlite3.Connection) -> set[str]:
        """Return all non-internal table names in the SQLite database."""
        rows = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """).fetchall()
        return {str(row[0]) for row in rows}

    def _list_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        """Return column names for a table."""
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}

    def _read_schema_version(self, conn: sqlite3.Connection) -> str | None:
        """Read the schema version from the meta table."""
        return self._read_meta_value(conn, SCHEMA_VERSION_KEY)

    def _read_meta_value(self, conn: sqlite3.Connection, key: str) -> str | None:
        """Read a metadata value from the meta table."""
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return str(row[0])

    @staticmethod
    def _serialize_audit_payload(
        warnings: list[str],
        *,
        semantic_score: float | None = None,
        score_metadata: dict[str, object] | None = None,
    ) -> str:
        """Serialize legacy warning-only or structured audit report payloads."""
        if semantic_score is None and score_metadata is None:
            return json.dumps(warnings)
        return json.dumps(
            {
                "warnings": warnings,
                "semantic_score": semantic_score,
                "score_metadata": score_metadata,
            }
        )

    @staticmethod
    def _deserialize_audit_payload(
        raw_payload: str,
    ) -> tuple[list[str], float | None, dict[str, object] | None]:
        """Deserialize audit payload rows with backward compatibility for legacy lists."""
        payload = json.loads(raw_payload)
        if isinstance(payload, list):
            return [str(item) for item in payload], None, None
        if isinstance(payload, dict):
            raw_warnings = payload.get("warnings", [])
            warnings = (
                [str(item) for item in raw_warnings] if isinstance(raw_warnings, list) else []
            )
            raw_score = payload.get("semantic_score")
            semantic_score = float(raw_score) if isinstance(raw_score, (int, float)) else None
            raw_metadata = payload.get("score_metadata")
            score_metadata = raw_metadata if isinstance(raw_metadata, dict) else None
            return warnings, semantic_score, score_metadata
        raise ValueError("audit payload must be a list or object")

    def _list_symbol_ids(self, conn: sqlite3.Connection, path: str) -> list[str]:
        """Return symbol ids for a file (internal helper)."""
        rows = conn.execute(
            "SELECT id FROM symbols WHERE file_path = ? ORDER BY start_line, end_line, id",
            (path,),
        ).fetchall()
        return [row["id"] for row in rows]

    def _row_to_symbol(self, row: sqlite3.Row) -> Symbol:
        """Convert a database row into a Symbol."""
        vector = json.loads(row["embedding_vector"]) if row["embedding_vector"] else None
        invariants = json.loads(row["invariants"]) if row["invariants"] else []
        calls = json.loads(row["calls"]) if row["calls"] else []
        covered_by = json.loads(row["covered_by"]) if row["covered_by"] else []
        signals = json.loads(row["signals"]) if row["signals"] else []
        attributes = json.loads(row["attributes"]) if row["attributes"] else {}
        if not isinstance(attributes, dict):
            attributes = {}

        return Symbol(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            fqname=row["fqname"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            container_id=row["container_id"],
            container_fqname=row["container_fqname"],
            signature=row["signature"],
            docstring=row["docstring"],
            body_hash=row["body_hash"],
            embedding_vector=vector,
            language=row["language"],
            repo_id=row["repo_id"],
            commit=row["commit_hash"],
            visibility=row["visibility"],
            exported=bool(row["exported"]) if row["exported"] is not None else None,
            tokens_estimate=(
                int(row["tokens_estimate"]) if row["tokens_estimate"] is not None else None
            ),
            invariants=invariants,
            calls=calls,
            covered_by=covered_by,
            is_serialization_boundary=bool(row["is_serialization_boundary"]),
            implicit_contract=row["implicit_contract"],
            signals=signals,
            attributes=attributes,
        )

    @staticmethod
    def _symbol_row(symbol: Symbol) -> tuple[object, ...]:
        """Normalize one symbol into the row payload used by upsert statements."""
        return (
            symbol.id,
            symbol.name,
            symbol.kind,
            symbol.fqname,
            symbol.file_path,
            symbol.start_line,
            symbol.end_line,
            symbol.container_id,
            symbol.container_fqname,
            symbol.signature,
            symbol.docstring,
            symbol.body_hash,
            json.dumps(symbol.embedding_vector) if symbol.embedding_vector is not None else None,
            symbol.language,
            symbol.repo_id,
            symbol.commit,
            symbol.visibility,
            int(symbol.exported) if symbol.exported is not None else None,
            symbol.tokens_estimate,
            json.dumps(symbol.invariants),
            json.dumps(symbol.calls),
            json.dumps(symbol.covered_by),
            int(symbol.is_serialization_boundary),
            symbol.implicit_contract,
            json.dumps([signal.model_dump() for signal in symbol.signals]),
            json.dumps(symbol.attributes),
        )

    @staticmethod
    def _upsert_symbol_rows(
        conn: sqlite3.Connection,
        symbol_rows: list[tuple[object, ...]],
    ) -> None:
        """Insert or update symbol rows with one executemany call."""
        conn.executemany(
            """
            INSERT INTO symbols (
                id, name, kind, fqname, file_path, start_line, end_line,
                container_id, container_fqname,
                signature, docstring, body_hash, embedding_vector, language,
                repo_id, commit_hash, visibility, exported, tokens_estimate,
                invariants, calls, covered_by, is_serialization_boundary, implicit_contract,
                signals, attributes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                kind = excluded.kind,
                fqname = excluded.fqname,
                file_path = excluded.file_path,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                container_id = excluded.container_id,
                container_fqname = excluded.container_fqname,
                signature = excluded.signature,
                docstring = excluded.docstring,
                body_hash = excluded.body_hash,
                embedding_vector = excluded.embedding_vector,
                language = excluded.language,
                repo_id = excluded.repo_id,
                commit_hash = excluded.commit_hash,
                visibility = excluded.visibility,
                exported = excluded.exported,
                tokens_estimate = excluded.tokens_estimate,
                invariants = excluded.invariants,
                calls = excluded.calls,
                covered_by = excluded.covered_by,
                is_serialization_boundary = excluded.is_serialization_boundary,
                implicit_contract = excluded.implicit_contract,
                signals = excluded.signals,
                attributes = excluded.attributes
            """,
            symbol_rows,
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> SymbolChunk:
        """Convert a database row into a SymbolChunk."""
        vector = json.loads(row["embedding_vector"]) if row["embedding_vector"] else None
        return SymbolChunk(
            chunk_id=row["chunk_id"],
            symbol_id=row["symbol_id"],
            chunk_part_index=int(row["chunk_part_index"]),
            chunk_part_total=int(row["chunk_part_total"]),
            text=row["text"],
            file_path=row["file_path"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            start_byte=int(row["start_byte"]) if row["start_byte"] is not None else None,
            end_byte=int(row["end_byte"]) if row["end_byte"] is not None else None,
            tokens_estimate=(
                int(row["tokens_estimate"]) if row["tokens_estimate"] is not None else None
            ),
            language=row["language"],
            repo_id=row["repo_id"],
            commit=row["commit_hash"],
            embedding_vector=vector,
        )

    @staticmethod
    def _chunk_row(chunk: SymbolChunk) -> tuple[object, ...]:
        """Normalize one chunk into the row payload used by upsert statements."""
        return (
            chunk.chunk_id,
            chunk.symbol_id,
            chunk.chunk_part_index,
            chunk.chunk_part_total,
            chunk.text,
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
            chunk.start_byte,
            chunk.end_byte,
            chunk.tokens_estimate,
            chunk.language,
            chunk.repo_id,
            chunk.commit,
            json.dumps(chunk.embedding_vector) if chunk.embedding_vector is not None else None,
        )

    @staticmethod
    def _upsert_chunk_rows(
        conn: sqlite3.Connection,
        chunk_rows: list[tuple[object, ...]],
    ) -> None:
        """Insert or update chunk rows with one executemany call."""
        conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id, symbol_id, chunk_part_index, chunk_part_total, text,
                file_path, start_line, end_line, start_byte, end_byte, tokens_estimate,
                language, repo_id, commit_hash, embedding_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                symbol_id = excluded.symbol_id,
                chunk_part_index = excluded.chunk_part_index,
                chunk_part_total = excluded.chunk_part_total,
                text = excluded.text,
                file_path = excluded.file_path,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                start_byte = excluded.start_byte,
                end_byte = excluded.end_byte,
                tokens_estimate = excluded.tokens_estimate,
                language = excluded.language,
                repo_id = excluded.repo_id,
                commit_hash = excluded.commit_hash,
                embedding_vector = excluded.embedding_vector
            """,
            chunk_rows,
        )

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> EdgeRecord:
        """Convert a database row into an EdgeRecord."""
        vector = json.loads(row["embedding_vector"]) if row["embedding_vector"] else None
        return EdgeRecord(
            edge_id=row["edge_id"],
            edge_type=row["edge_type"],
            from_id=row["from_id"],
            to_id=row["to_id"],
            from_kind=row["from_kind"],
            to_kind=row["to_kind"],
            file_path=row["file_path"],
            line=int(row["line"]),
            confidence=float(row["confidence"]),
            repo_id=row["repo_id"],
            commit=row["commit_hash"],
            text=row["text"],
            embedding_vector=vector,
        )

    @staticmethod
    def _edge_row(edge: EdgeRecord) -> tuple[object, ...]:
        """Normalize one edge record into the row payload used by upsert statements."""
        return (
            edge.edge_id,
            edge.edge_type,
            edge.from_id,
            edge.to_id,
            edge.from_kind,
            edge.to_kind,
            edge.file_path,
            edge.line,
            edge.confidence,
            edge.repo_id,
            edge.commit,
            edge.text,
            json.dumps(edge.embedding_vector) if edge.embedding_vector is not None else None,
        )

    @staticmethod
    def _upsert_edge_rows(
        conn: sqlite3.Connection,
        edge_rows: list[tuple[object, ...]],
    ) -> None:
        """Insert or update edge rows with one executemany call."""
        conn.executemany(
            """
            INSERT INTO edges (
                edge_id, edge_type, from_id, to_id, from_kind, to_kind,
                file_path, line, confidence, repo_id, commit_hash, text, embedding_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                edge_type = excluded.edge_type,
                from_id = excluded.from_id,
                to_id = excluded.to_id,
                from_kind = excluded.from_kind,
                to_kind = excluded.to_kind,
                file_path = excluded.file_path,
                line = excluded.line,
                confidence = excluded.confidence,
                repo_id = excluded.repo_id,
                commit_hash = excluded.commit_hash,
                text = excluded.text,
                embedding_vector = excluded.embedding_vector
            """,
            edge_rows,
        )

    @staticmethod
    def _upsert_file_metadata_row(conn: sqlite3.Connection, metadata: FileMetadata) -> None:
        """Insert or update one file metadata row."""
        conn.execute(
            """
            INSERT INTO files (path, language, content_hash, last_indexed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                language = excluded.language,
                content_hash = excluded.content_hash,
                last_indexed = excluded.last_indexed
            """,
            (
                metadata.path,
                metadata.language,
                metadata.content_hash,
                metadata.last_indexed.isoformat(),
            ),
        )


class CacheRecoveryError(RuntimeError):
    """Raised when automatic corruption recovery cannot repair cache artifacts."""
