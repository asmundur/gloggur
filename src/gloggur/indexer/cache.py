from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, List, Optional

from gloggur.models import AuditFileMetadata, FileMetadata, IndexMetadata, Symbol
from gloggur.io_failures import wrap_io_error

SCHEMA_VERSION_KEY = "schema_version"
INDEX_PROFILE_KEY = "index_profile"
CACHE_SCHEMA_VERSION = "2"
SQLITE_BUSY_TIMEOUT_MS = 5_000
SQLITE_CONNECT_TIMEOUT_SECONDS = SQLITE_BUSY_TIMEOUT_MS / 1000
SQLITE_JOURNAL_MODE = "WAL"
SQLITE_SYNCHRONOUS = "NORMAL"

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
        self.last_reset_reason: Optional[str] = None
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

    def delete_file_metadata(self, path: str) -> None:
        """Delete cached file metadata for a path."""
        with self._connect() as conn:
            conn.execute("DELETE FROM files WHERE path = ?", (path,))

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

    def count_files(self) -> int:
        """Return the number of indexed files."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM files").fetchone()
            return int(row["count"] if row else 0)

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

    def delete_index_metadata(self) -> None:
        """Remove index-level metadata (used while a rebuild is in progress)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM metadata WHERE key = ?", ("index",))

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

    def _schema_reset_plan(self) -> Optional[_ResetPlan]:
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
                    return _ResetPlan(
                        reason=f"legacy tables present ({', '.join(legacy_tables)})"
                    )

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
                                f"table '{table}' missing columns "
                                f"({', '.join(missing_columns)})"
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
        sys.stderr.write(
            rebuild_notice
        )

    def _integrity_issue(self, conn: sqlite3.Connection) -> Optional[str]:
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

    def _quarantine_or_remove(self, path: str, timestamp: str) -> Optional[str]:
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


class CacheRecoveryError(RuntimeError):
    """Raised when automatic corruption recovery cannot repair cache artifacts."""
