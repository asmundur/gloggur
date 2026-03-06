from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from gloggur.symbol_index.models import IndexedFile, SymbolOccurrence

SQLITE_BUSY_TIMEOUT_MS = 5_000
SQLITE_CONNECT_TIMEOUT_SECONDS = SQLITE_BUSY_TIMEOUT_MS / 1000
SQLITE_JOURNAL_MODE = "WAL"
SQLITE_SYNCHRONOUS = "NORMAL"
SCHEMA_VERSION_KEY = "schema_version"
SYMBOL_INDEX_SCHEMA_VERSION = "2"
REQUIRED_TABLES = {"occurrences", "files", "meta"}
REQUIRED_COLUMNS = {
    "occurrences": {
        "symbol",
        "kind",
        "path",
        "start_line",
        "end_line",
        "start_byte",
        "end_byte",
        "language",
        "container",
        "signature",
    },
    "files": {"path", "content_hash", "mtime_ns", "language", "last_indexed"},
    "meta": {"key", "value"},
}


@dataclass(frozen=True)
class SymbolIndexStoreConfig:
    repo_root: Path

    @property
    def db_path(self) -> Path:
        return self.repo_root / ".gloggur" / "index" / "symbols.db"


class SymbolIndexStore:
    """SQLite-backed symbol occurrence index at .gloggur/index/symbols.db."""

    def __init__(self, config: SymbolIndexStoreConfig, *, create_if_missing: bool = True) -> None:
        self.config = config
        self._create_if_missing = create_if_missing
        self._available = True
        self._unavailability_reason: str | None = None
        self.last_reset_reason: str | None = None
        db_path = self.config.db_path
        if not create_if_missing and not db_path.exists():
            self._available = False
            self._unavailability_reason = "missing or unreadable symbols.db"
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_problem = self._schema_problem() if db_path.exists() else None
        if schema_problem:
            if create_if_missing:
                self._reset_database(schema_problem)
            else:
                self._available = False
                self._unavailability_reason = schema_problem
                return
        if create_if_missing or db_path.exists():
            self._init_db()

    @property
    def db_path(self) -> str:
        return str(self.config.db_path)

    @property
    def available(self) -> bool:
        return self._available and self.config.db_path.exists()

    @property
    def unavailability_reason(self) -> str | None:
        return self._unavailability_reason

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if not self.available and not self._create_if_missing:
            raise RuntimeError("symbol index database is unavailable")
        conn = sqlite3.connect(str(self.config.db_path), timeout=SQLITE_CONNECT_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
            conn.execute(f"PRAGMA journal_mode = {SQLITE_JOURNAL_MODE}")
            conn.execute(f"PRAGMA synchronous = {SQLITE_SYNCHRONOUS}")
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS occurrences (
                    symbol TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    start_byte INTEGER NOT NULL,
                    end_byte INTEGER NOT NULL,
                    language TEXT,
                    container TEXT,
                    signature TEXT
                );
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    language TEXT,
                    last_indexed TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_occ_symbol_kind ON occurrences (symbol, kind);
                CREATE INDEX IF NOT EXISTS idx_occ_path ON occurrences (path);
                CREATE INDEX IF NOT EXISTS idx_occ_path_line ON occurrences (path, start_line);
                CREATE INDEX IF NOT EXISTS idx_occ_kind ON occurrences (kind);
                """)
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION_KEY, SYMBOL_INDEX_SCHEMA_VERSION),
            )

    def _schema_problem(self) -> str | None:
        try:
            with self._connect() as conn:
                existing_tables = self._list_tables(conn)
                missing_tables = sorted(REQUIRED_TABLES - existing_tables)
                if missing_tables:
                    return f"required tables missing ({', '.join(missing_tables)})"
                for table, expected_columns in REQUIRED_COLUMNS.items():
                    existing_columns = self._list_columns(conn, table)
                    missing_columns = sorted(expected_columns - existing_columns)
                    if missing_columns:
                        return f"table '{table}' missing columns ({', '.join(missing_columns)})"
                version = self._read_meta_value(conn, SCHEMA_VERSION_KEY)
                if version != SYMBOL_INDEX_SCHEMA_VERSION:
                    found = version if version is not None else "none"
                    return (
                        "schema version mismatch "
                        f"(found {found}, expected {SYMBOL_INDEX_SCHEMA_VERSION})"
                    )
        except sqlite3.DatabaseError as exc:
            return f"sqlite open/integrity error: {exc}"
        return None

    def _reset_database(self, reason: str) -> None:
        self.last_reset_reason = reason
        self._delete_database_artifacts()

    def _delete_database_artifacts(self) -> None:
        db_path = self.config.db_path
        for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
            if path.exists():
                path.unlink()

    def get_file(self, path: str) -> IndexedFile | None:
        if not self.available:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT path, content_hash, mtime_ns, language, last_indexed
                FROM files
                WHERE path = ?
                """,
                (path,),
            ).fetchone()
        if row is None:
            return None
        return IndexedFile(
            path=str(row["path"]),
            content_hash=str(row["content_hash"]),
            mtime_ns=int(row["mtime_ns"]),
            language=str(row["language"]) if row["language"] is not None else None,
            last_indexed=datetime.fromisoformat(str(row["last_indexed"])),
        )

    def list_indexed_paths(self) -> list[str]:
        if not self.available:
            return []
        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM files ORDER BY path").fetchall()
        return [str(row["path"]) for row in rows]

    def upsert_file(self, indexed_file: IndexedFile) -> None:
        if not self.available:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (path, content_hash, mtime_ns, language, last_indexed)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    mtime_ns = excluded.mtime_ns,
                    language = excluded.language,
                    last_indexed = excluded.last_indexed
                """,
                (
                    indexed_file.path,
                    indexed_file.content_hash,
                    indexed_file.mtime_ns,
                    indexed_file.language,
                    indexed_file.last_indexed.isoformat(),
                ),
            )

    def replace_file_occurrences(
        self,
        *,
        indexed_file: IndexedFile,
        occurrences: list[SymbolOccurrence],
    ) -> None:
        if not self.available:
            return
        rows = [
            (
                item.symbol,
                item.kind,
                item.path,
                item.start_line,
                item.end_line,
                item.start_byte,
                item.end_byte,
                item.language,
                item.container,
                item.signature,
            )
            for item in occurrences
        ]
        with self._connect() as conn:
            conn.execute("DELETE FROM occurrences WHERE path = ?", (indexed_file.path,))
            if rows:
                conn.executemany(
                    """
                    INSERT INTO occurrences (
                        symbol, kind, path, start_line, end_line, start_byte, end_byte,
                        language, container, signature
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.execute(
                """
                INSERT INTO files (path, content_hash, mtime_ns, language, last_indexed)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    mtime_ns = excluded.mtime_ns,
                    language = excluded.language,
                    last_indexed = excluded.last_indexed
                """,
                (
                    indexed_file.path,
                    indexed_file.content_hash,
                    indexed_file.mtime_ns,
                    indexed_file.language,
                    indexed_file.last_indexed.isoformat(),
                ),
            )

    def delete_file(self, path: str) -> None:
        if not self.available:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM occurrences WHERE path = ?", (path,))
            conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def prune_missing_files(self, *, seen_paths: set[str], scope_prefix: str | None = None) -> int:
        if not self.available:
            return 0
        normalized_prefix = os.path.normpath(scope_prefix) if scope_prefix else None
        removed = 0
        for indexed_path in self.list_indexed_paths():
            if normalized_prefix and not os.path.normpath(indexed_path).startswith(
                normalized_prefix.rstrip(os.sep) + os.sep
            ):
                if os.path.normpath(indexed_path) != normalized_prefix:
                    continue
            if indexed_path in seen_paths:
                continue
            self.delete_file(indexed_path)
            removed += 1
        return removed

    def list_occurrences(
        self,
        *,
        path_prefixes: tuple[str, ...] = (),
        language: str | None = None,
    ) -> list[SymbolOccurrence]:
        if not self.available:
            return []
        query = (
            "SELECT symbol, kind, path, start_line, end_line, start_byte, end_byte, "
            "language, container, signature "
            "FROM occurrences WHERE 1=1"
        )
        params: list[object] = []
        if language:
            query += " AND language = ?"
            params.append(language)
        if path_prefixes:
            query += " AND ("
            path_clauses: list[str] = []
            for prefix in path_prefixes:
                normalized = prefix.rstrip("/\\")
                if not normalized:
                    continue
                path_clauses.append("path = ?")
                params.append(normalized)
                path_clauses.append("path LIKE ? ESCAPE '\\'")
                escaped = self._escape_like(normalized)
                params.append(f"{escaped}/%")
                params.append(f"{escaped}\\\\%")
                path_clauses.append("path LIKE ? ESCAPE '\\'")
            if not path_clauses:
                query += "1=1"
            else:
                query += " OR ".join(path_clauses)
            query += ")"
        query += " ORDER BY path, start_line, end_line, symbol, kind"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            SymbolOccurrence(
                symbol=str(row["symbol"]),
                kind=str(row["kind"]),
                path=str(row["path"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                start_byte=int(row["start_byte"]) if row["start_byte"] is not None else None,
                end_byte=int(row["end_byte"]) if row["end_byte"] is not None else None,
                language=str(row["language"]) if row["language"] is not None else None,
                container=str(row["container"]) if row["container"] is not None else None,
                signature=str(row["signature"]) if row["signature"] is not None else None,
            )
            for row in rows
        ]

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _list_tables(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """).fetchall()
        return {str(row[0]) for row in rows}

    @staticmethod
    def _list_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}

    @staticmethod
    def _read_meta_value(conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row["value"])
