from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from gloggur.io_failures import wrap_io_error
from gloggur.models import EdgeRecord, Symbol, SymbolChunk

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

    def sample_symbol_file_paths(self, *, limit: int = 64) -> list[str]:
        """Return a bounded deterministic set of symbol file paths."""
        safe_limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT file_path
                FROM symbols
                WHERE file_path IS NOT NULL AND file_path != ''
                ORDER BY file_path
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        paths: list[str] = []
        for row in rows:
            value = row["file_path"]
            if isinstance(value, str) and value.strip():
                paths.append(value)
        return paths

    def list_chunks(self) -> list[SymbolChunk]:
        """List all symbol chunks in deterministic order."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM chunks
                ORDER BY file_path, start_line, chunk_part_index, chunk_id
                """).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> SymbolChunk | None:
        """Return one chunk row by id."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if not row:
                return None
            return self._row_to_chunk(row)

    def list_chunks_for_symbol(self, symbol_id: str) -> list[SymbolChunk]:
        """Return chunk rows for one symbol."""
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

    def filter_chunks_by_path(self, file_path: str) -> list[SymbolChunk]:
        """Return chunk rows for one file path."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE file_path = ?
                ORDER BY start_line, chunk_part_index, chunk_id
                """,
                (file_path,),
            ).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def list_edges(self) -> list[EdgeRecord]:
        """Return all edge records."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges ORDER BY file_path, line, edge_type, edge_id"
            ).fetchall()
            return [self._row_to_edge(row) for row in rows]

    def list_edges_for_file(self, file_path: str) -> list[EdgeRecord]:
        """Return edge rows emitted for one file path."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM edges
                WHERE file_path = ?
                ORDER BY line, edge_type, edge_id
                """,
                (file_path,),
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
        """Return incoming/outgoing/bidirectional edges for a symbol."""
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
    def _row_to_chunk(row: sqlite3.Row) -> SymbolChunk:
        """Convert a chunks table row into a SymbolChunk."""
        import json

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
    def _row_to_edge(row: sqlite3.Row) -> EdgeRecord:
        """Convert an edges table row into an EdgeRecord."""
        import json

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
