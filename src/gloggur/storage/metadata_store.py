from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from gloggur.io_failures import wrap_io_error
from gloggur.models import Symbol

SQLITE_BUSY_TIMEOUT_MS = 5_000
SQLITE_CONNECT_TIMEOUT_SECONDS = SQLITE_BUSY_TIMEOUT_MS / 1000


@dataclass
class MetadataStoreConfig:
    """Configuration for the metadata store (SQLite db path)."""

    cache_dir: str

    @property
    def db_path(self) -> str:
        """Return the SQLite database path."""
        return os.path.join(self.cache_dir, "index.db")


class MetadataStore:
    """Read-only access to indexed symbol metadata in SQLite."""

    def __init__(self, config: MetadataStoreConfig) -> None:
        """Initialize the metadata store."""
        self.config = config

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        """Fetch a symbol by its id from the symbols table."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM symbols WHERE id = ?", (symbol_id,)).fetchone()
            if not row:
                return None
            return self._row_to_symbol(row)

    def filter_symbols(
        self,
        kinds: list[str] | None = None,
        file_path: str | None = None,
        language: str | None = None,
    ) -> list[Symbol]:
        """Filter symbols by kind, file path, and/or language."""
        query = "SELECT * FROM symbols WHERE 1=1"
        params: list[str] = []
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            query += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        if file_path:
            query += " AND file_path = ?"
            params.append(file_path)
        if language:
            query += " AND language = ?"
            params.append(language)
        query += " ORDER BY start_line"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def list_symbols(self) -> list[Symbol]:
        """List all symbols ordered by file and start line."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM symbols ORDER BY file_path, start_line").fetchall()
            return [self._row_to_symbol(row) for row in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database access."""
        try:
            conn = sqlite3.connect(
                self.config.db_path,
                timeout=SQLITE_CONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, sqlite3.DatabaseError) as exc:
            raise wrap_io_error(
                exc,
                operation="open metadata database connection",
                path=self.config.db_path,
            ) from exc
        try:
            conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise wrap_io_error(
                exc,
                operation="configure metadata database pragmas",
                path=self.config.db_path,
            ) from exc
        conn.row_factory = sqlite3.Row
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
                    operation="execute metadata database transaction",
                    path=self.config.db_path,
                ) from exc
            raise
        finally:
            conn.close()

    @staticmethod
    def _row_to_symbol(row: sqlite3.Row) -> Symbol:
        """Convert a symbols table row into a Symbol."""
        import json

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
