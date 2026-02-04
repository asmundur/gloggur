from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, List, Optional

from gloggur.models import FileMetadata, IndexMetadata, Symbol, ValidationFileMetadata


@dataclass
class CacheConfig:
    """Configuration for the on-disk cache."""
    cache_dir: str

    @property
    def db_path(self) -> str:
        """Return the path to the SQLite index database."""
        return os.path.join(self.cache_dir, "index.db")


class CacheManager:
    """SQLite-backed cache for symbols, files, and validations."""
    def __init__(self, config: CacheConfig) -> None:
        """Initialize the cache and ensure the database exists."""
        self.config = config
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
                CREATE TABLE IF NOT EXISTS validations (
                    symbol_id TEXT PRIMARY KEY,
                    warnings TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS validation_files (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    last_validated TEXT NOT NULL
                );
                """
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

    def set_validation_warnings(self, symbol_id: str, warnings: List[str]) -> None:
        """Store validation warnings for a symbol."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO validations (symbol_id, warnings) VALUES (?, ?)
                ON CONFLICT(symbol_id) DO UPDATE SET warnings = excluded.warnings
                """,
                (symbol_id, json.dumps(warnings)),
            )

    def get_validation_warnings(self, symbol_id: str) -> List[str]:
        """Fetch validation warnings for a symbol."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT warnings FROM validations WHERE symbol_id = ?", (symbol_id,)
            ).fetchone()
            if not row:
                return []
            return json.loads(row["warnings"])

    def get_validation_file_metadata(self, path: str) -> Optional[ValidationFileMetadata]:
        """Return cached validation metadata for a file."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM validation_files WHERE path = ?", (path,)
            ).fetchone()
            if not row:
                return None
            return ValidationFileMetadata(
                path=row["path"],
                content_hash=row["content_hash"],
                last_validated=datetime.fromisoformat(row["last_validated"]),
            )

    def upsert_validation_file_metadata(self, metadata: ValidationFileMetadata) -> None:
        """Insert or update validation metadata for a file."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO validation_files (path, content_hash, last_validated)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    last_validated = excluded.last_validated
                """,
                (
                    metadata.path,
                    metadata.content_hash,
                    metadata.last_validated.isoformat(),
                ),
            )

    def clear(self) -> None:
        """Clear all cached data."""
        with self._connect() as conn:
            conn.executescript(
                """
                DELETE FROM validations;
                DELETE FROM validation_files;
                DELETE FROM symbols;
                DELETE FROM files;
                DELETE FROM metadata;
                """
            )

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
