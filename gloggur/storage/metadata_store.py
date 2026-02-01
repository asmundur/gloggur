from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional

from gloggur.models import Symbol


@dataclass
class MetadataStoreConfig:
    cache_dir: str

    @property
    def db_path(self) -> str:
        return os.path.join(self.cache_dir, "index.db")


class MetadataStore:
    def __init__(self, config: MetadataStoreConfig) -> None:
        self.config = config

    def get_symbol(self, symbol_id: str) -> Optional[Symbol]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM symbols WHERE id = ?", (symbol_id,)).fetchone()
            if not row:
                return None
            return self._row_to_symbol(row)

    def filter_symbols(
        self,
        kinds: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[Symbol]:
        query = "SELECT * FROM symbols WHERE 1=1"
        params: List[str] = []
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

    def list_symbols(self) -> List[Symbol]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM symbols ORDER BY file_path, start_line").fetchall()
            return [self._row_to_symbol(row) for row in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
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

    @staticmethod
    def _row_to_symbol(row: sqlite3.Row) -> Symbol:
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
