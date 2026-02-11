from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, List, Optional

from gloggur.models import AuditFileMetadata, FileMetadata, IndexMetadata, Symbol

SCHEMA_VERSION_KEY = "schema_version"
INDEX_PROFILE_KEY = "index_profile"
CACHE_SCHEMA_VERSION = "2"

REQUIRED_TABLES = {"files", "symbols", "metadata", "audits", "audit_files", "meta"}
LEGACY_TABLES = {"validations", "validation_files"}
REQUIRED_COLUMNS = {
    "files": {"path", "language", "content_hash", "last_indexed"},
    "symbols": {
        "id",
        "name",
        "kind",
        "file_path",
        "start_line",
        "end_line",
        "signature",
        "docstring",
        "body_hash",
        "embedding_vector",
        "language",
    },
    "metadata": {"key", "value"},
    "audits": {"symbol_id", "warnings"},
    "audit_files": {"path", "content_hash", "last_audited"},
    "meta": {"key", "value"},
}


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
        self.last_reset_reason: Optional[str] = None
        os.makedirs(self.config.cache_dir, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for a transactional database connection."""
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        reset_reason = self._schema_reset_reason()
        if reset_reason:
            self._reset_database(reset_reason)
        with self._connect() as conn:
            conn.executescript(
                """
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
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    signature TEXT,
                    docstring TEXT,
                    body_hash TEXT NOT NULL,
                    embedding_vector TEXT,
                    language TEXT
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
                    warnings TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_files (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    last_audited TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION_KEY, CACHE_SCHEMA_VERSION),
            )

    def get_file_metadata(self, path: str) -> Optional[FileMetadata]:
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

    def delete_symbols_for_file(self, path: str) -> None:
        """Delete all cached symbols for a file."""
        with self._connect() as conn:
            conn.execute("DELETE FROM symbols WHERE file_path = ?", (path,))

    def upsert_symbols(self, symbols: Iterable[Symbol]) -> None:
        """Insert or update a batch of symbols."""
        with self._connect() as conn:
            for symbol in symbols:
                conn.execute(
                    """
                    INSERT INTO symbols (
                        id, name, kind, file_path, start_line, end_line,
                        signature, docstring, body_hash, embedding_vector, language
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        kind = excluded.kind,
                        file_path = excluded.file_path,
                        start_line = excluded.start_line,
                        end_line = excluded.end_line,
                        signature = excluded.signature,
                        docstring = excluded.docstring,
                        body_hash = excluded.body_hash,
                        embedding_vector = excluded.embedding_vector,
                        language = excluded.language
                    """,
                    (
                        symbol.id,
                        symbol.name,
                        symbol.kind,
                        symbol.file_path,
                        symbol.start_line,
                        symbol.end_line,
                        symbol.signature,
                        symbol.docstring,
                        symbol.body_hash,
                        json.dumps(symbol.embedding_vector)
                        if symbol.embedding_vector is not None
                        else None,
                        symbol.language,
                    ),
                )

    def list_symbols(self) -> List[Symbol]:
        """Return all cached symbols."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM symbols").fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def list_symbols_for_file(self, path: str) -> List[Symbol]:
        """Return cached symbols for a file path."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE file_path = ? ORDER BY start_line", (path,)
            ).fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def get_index_metadata(self) -> Optional[IndexMetadata]:
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

    def get_schema_version(self) -> Optional[str]:
        """Return the cache schema version marker."""
        with self._connect() as conn:
            return self._read_meta_value(conn, SCHEMA_VERSION_KEY)

    def get_index_profile(self) -> Optional[str]:
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

    def set_audit_warnings(self, symbol_id: str, warnings: List[str]) -> None:
        """Store audit warnings for a symbol."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audits (symbol_id, warnings) VALUES (?, ?)
                ON CONFLICT(symbol_id) DO UPDATE SET warnings = excluded.warnings
                """,
                (symbol_id, json.dumps(warnings)),
            )

    def get_audit_warnings(self, symbol_id: str) -> List[str]:
        """Fetch audit warnings for a symbol."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT warnings FROM audits WHERE symbol_id = ?", (symbol_id,)
            ).fetchone()
            if not row:
                return []
            return json.loads(row["warnings"])

    def get_audit_file_metadata(self, path: str) -> Optional[AuditFileMetadata]:
        """Return cached audit metadata for a file."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM audit_files WHERE path = ?", (path,)
            ).fetchone()
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
            conn.executescript(
                """
                DELETE FROM audits;
                DELETE FROM audit_files;
                DELETE FROM symbols;
                DELETE FROM files;
                DELETE FROM metadata;
                """
            )
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION_KEY, CACHE_SCHEMA_VERSION),
            )
            conn.execute("DELETE FROM meta WHERE key = ?", (INDEX_PROFILE_KEY,))

    def _schema_reset_reason(self) -> Optional[str]:
        """Return a reason to reset the cache if current schema is incompatible."""
        if not os.path.exists(self.config.db_path):
            return None

        try:
            with sqlite3.connect(self.config.db_path) as conn:
                existing_tables = self._list_tables(conn)
                legacy_tables = sorted(existing_tables & LEGACY_TABLES)
                if legacy_tables:
                    return f"legacy tables present ({', '.join(legacy_tables)})"

                missing_tables = sorted(REQUIRED_TABLES - existing_tables)
                if missing_tables:
                    return f"required tables missing ({', '.join(missing_tables)})"

                for table, expected_columns in REQUIRED_COLUMNS.items():
                    existing_columns = self._list_columns(conn, table)
                    missing_columns = sorted(expected_columns - existing_columns)
                    if missing_columns:
                        return (
                            f"table '{table}' missing columns "
                            f"({', '.join(missing_columns)})"
                        )

                version = self._read_schema_version(conn)
                if version != CACHE_SCHEMA_VERSION:
                    found = version if version is not None else "none"
                    return (
                        "schema version mismatch "
                        f"(found {found}, expected {CACHE_SCHEMA_VERSION})"
                    )
        except sqlite3.DatabaseError as exc:
            return f"invalid sqlite database ({exc})"

        return None

    def _reset_database(self, reason: str) -> None:
        """Remove incompatible SQLite database files so schema can be recreated."""
        self.last_reset_reason = reason
        db_path = self.config.db_path
        for path in (db_path, f"{db_path}-wal", f"{db_path}-shm"):
            if os.path.exists(path):
                os.remove(path)
        sys.stderr.write(
            f"Cache schema changed; rebuilding cache at {db_path} ({reason}).\n"
        )

    def _list_tables(self, conn: sqlite3.Connection) -> set[str]:
        """Return all non-internal table names in the SQLite database."""
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        return {str(row[0]) for row in rows}

    def _list_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        """Return column names for a table."""
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}

    def _read_schema_version(self, conn: sqlite3.Connection) -> Optional[str]:
        """Read the schema version from the meta table."""
        return self._read_meta_value(conn, SCHEMA_VERSION_KEY)

    def _read_meta_value(self, conn: sqlite3.Connection, key: str) -> Optional[str]:
        """Read a metadata value from the meta table."""
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return str(row[0])

    def _list_symbol_ids(self, conn: sqlite3.Connection, path: str) -> List[str]:
        """Return symbol ids for a file (internal helper)."""
        rows = conn.execute("SELECT id FROM symbols WHERE file_path = ?", (path,)).fetchall()
        return [row["id"] for row in rows]

    def _row_to_symbol(self, row: sqlite3.Row) -> Symbol:
        """Convert a database row into a Symbol."""
        vector = json.loads(row["embedding_vector"]) if row["embedding_vector"] else None
        return Symbol(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            signature=row["signature"],
            docstring=row["docstring"],
            body_hash=row["body_hash"],
            embedding_vector=vector,
            language=row["language"],
        )
