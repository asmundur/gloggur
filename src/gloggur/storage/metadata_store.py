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

    def filter_symbols_by_file_match(
        self,
        *,
        file_path: str,
        kinds: list[str] | None = None,
        language: str | None = None,
    ) -> list[Symbol]:
        """Filter symbols by exact file path or boundary-safe directory prefix matching."""
        candidates = self._file_match_candidates(file_path)
        if not candidates:
            return []
        symbols_by_id: dict[str, Symbol] = {}
        for candidate in candidates:
            for symbol in self._filter_symbols_exact_or_prefix(
                candidate,
                kinds=kinds,
                language=language,
            ):
                if symbol.id not in symbols_by_id:
                    symbols_by_id[symbol.id] = symbol
        symbols = list(symbols_by_id.values())
        symbols.sort(key=lambda item: (item.file_path, item.start_line, item.id))
        return symbols

    def list_symbols(self) -> list[Symbol]:
        """List all symbols ordered by file and start line."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM symbols ORDER BY file_path, start_line").fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def _filter_symbols_exact_or_prefix(
        self,
        file_path: str,
        *,
        kinds: list[str] | None = None,
        language: str | None = None,
    ) -> list[Symbol]:
        """Filter symbols using exact-path equality plus escaped LIKE prefix clauses."""
        prefix_root = file_path.rstrip("/\\")
        if not prefix_root:
            return []
        query = "SELECT * FROM symbols WHERE 1=1"
        params: list[str] = []
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            query += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        if language:
            query += " AND language = ?"
            params.append(language)
        escaped_prefix = self._escape_sql_like(prefix_root)
        prefix_patterns = [f"{escaped_prefix}/%"]
        windows_pattern = f"{escaped_prefix}\\\\%"
        if windows_pattern not in prefix_patterns:
            prefix_patterns.append(windows_pattern)
        query += " AND (file_path = ?"
        params.append(prefix_root)
        for pattern in prefix_patterns:
            query += " OR file_path LIKE ? ESCAPE '\\'"
            params.append(pattern)
        query += ") ORDER BY file_path, start_line"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_symbol(row) for row in rows]

    @staticmethod
    def _escape_sql_like(value: str) -> str:
        """Escape wildcard metacharacters for SQL LIKE clauses."""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _file_match_candidates(file_path: str) -> list[str]:
        """Return de-duplicated file-match candidates across relative path forms."""
        raw = file_path.strip()
        if not raw:
            return []
        candidates: list[str] = []
        for candidate in (raw, os.path.normpath(raw)):
            normalized = candidate.rstrip("/\\")
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        if not os.path.isabs(raw):
            dot_candidate = os.path.join(".", raw)
            for candidate in (dot_candidate, os.path.normpath(dot_candidate)):
                normalized = candidate.rstrip("/\\")
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        abs_candidate = os.path.abspath(raw).rstrip("/\\")
        if abs_candidate and abs_candidate not in candidates:
            candidates.append(abs_candidate)
        return candidates

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
        invariants = json.loads(row["invariants"]) if row["invariants"] else []
        calls = json.loads(row["calls"]) if "calls" in row.keys() and row["calls"] else []
        covered_by = (
            json.loads(row["covered_by"])
            if "covered_by" in row.keys() and row["covered_by"]
            else []
        )
        signals = json.loads(row["signals"]) if "signals" in row.keys() and row["signals"] else []
        if not signals:
            for expression in invariants:
                signals.append(
                    {
                        "type": "code.invariant",
                        "payload": {"expression": expression},
                        "source": "legacy_projection",
                    }
                )
            for target in calls:
                signals.append(
                    {
                        "type": "code.call",
                        "payload": {"target": target},
                        "source": "legacy_projection",
                    }
                )
            if bool(row["is_serialization_boundary"]):
                signals.append(
                    {
                        "type": "boundary.serialization",
                        "payload": {"detector": "legacy_projection"},
                        "source": "legacy_projection",
                    }
                )
            implicit_contract = row["implicit_contract"]
            if implicit_contract:
                signals.append(
                    {
                        "type": "test.implicit_contract",
                        "payload": {"text": implicit_contract},
                        "source": "legacy_projection",
                    }
                )
        attributes = (
            json.loads(row["attributes"])
            if "attributes" in row.keys() and row["attributes"]
            else {}
        )
        if not isinstance(attributes, dict):
            attributes = {}

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
            invariants=invariants,
            calls=calls,
            covered_by=covered_by,
            is_serialization_boundary=bool(row["is_serialization_boundary"]),
            implicit_contract=row["implicit_contract"],
            signals=signals,
            attributes=attributes,
        )
