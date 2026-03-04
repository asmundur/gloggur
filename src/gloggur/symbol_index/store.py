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
        db_path = self.config.db_path
        if not create_if_missing and not db_path.exists():
            self._available = False
            return
        if create_if_missing:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    @property
    def db_path(self) -> str:
        return str(self.config.db_path)

    @property
    def available(self) -> bool:
        return self._available and self.config.db_path.exists()

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
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS occurrences (
                    symbol TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    line INTEGER NOT NULL,
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
                CREATE INDEX IF NOT EXISTS idx_occ_symbol_kind ON occurrences (symbol, kind);
                CREATE INDEX IF NOT EXISTS idx_occ_path ON occurrences (path);
                CREATE INDEX IF NOT EXISTS idx_occ_path_line ON occurrences (path, line);
                CREATE INDEX IF NOT EXISTS idx_occ_kind ON occurrences (kind);
                """
            )

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
                item.line,
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
                    INSERT INTO occurrences (symbol, kind, path, line, language, container, signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
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
            "SELECT symbol, kind, path, line, language, container, signature "
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
        query += " ORDER BY path, line, symbol, kind"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            SymbolOccurrence(
                symbol=str(row["symbol"]),
                kind=str(row["kind"]),
                path=str(row["path"]),
                line=int(row["line"]),
                language=str(row["language"]) if row["language"] is not None else None,
                container=str(row["container"]) if row["container"] is not None else None,
                signature=str(row["signature"]) if row["signature"] is not None else None,
            )
            for row in rows
        ]

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
