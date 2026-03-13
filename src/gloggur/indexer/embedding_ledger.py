from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from gloggur.io_failures import wrap_io_error

EMBEDDING_LEDGER_DB_NAME = "embedding-ledger.db"
_SQLITE_BUSY_TIMEOUT_MS = 5_000
_SQLITE_CONNECT_TIMEOUT_SECONDS = _SQLITE_BUSY_TIMEOUT_MS / 1000


def embedding_text_hash(text: str) -> str:
    """Return a stable hash for one embedding input payload."""
    return hashlib.sha256(text.encode("utf8")).hexdigest()


class EmbeddingLedger:
    """Persistent embedding cache keyed by profile, record kind, and text hash."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="create embedding ledger directory",
                path=self.cache_dir,
            ) from exc
        self.db_path = os.path.join(self.cache_dir, EMBEDDING_LEDGER_DB_NAME)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=_SQLITE_CONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, sqlite3.DatabaseError) as exc:
            raise wrap_io_error(
                exc,
                operation="open embedding ledger database connection",
                path=self.db_path,
            ) from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
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
                    operation="execute embedding ledger transaction",
                    path=self.db_path,
                ) from exc
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS embedding_ledger (
                    embedding_profile TEXT NOT NULL,
                    record_kind TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (embedding_profile, record_kind, text_hash)
                );
                """)

    def get_vectors(
        self,
        *,
        embedding_profile: str,
        record_kind: str,
        text_hashes: Iterable[str],
    ) -> dict[str, list[float]]:
        if not os.path.exists(self.db_path):
            return {}
        keys = sorted({text_hash for text_hash in text_hashes if text_hash})
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        params = [embedding_profile, record_kind, *keys]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT text_hash, vector
                FROM embedding_ledger
                WHERE embedding_profile = ?
                  AND record_kind = ?
                  AND text_hash IN ({placeholders})
                """,
                params,
            ).fetchall()
        payload: dict[str, list[float]] = {}
        for row in rows:
            raw_vector = row["vector"]
            try:
                vector = json.loads(raw_vector)
            except json.JSONDecodeError:
                continue
            is_numeric_vector = isinstance(vector, list) and all(
                isinstance(value, (int, float)) for value in vector
            )
            if is_numeric_vector:
                payload[str(row["text_hash"])] = [float(value) for value in vector]
        return payload

    def upsert_vectors(
        self,
        *,
        embedding_profile: str,
        record_kind: str,
        entries: Iterable[tuple[str, list[float]]],
    ) -> None:
        if not os.path.exists(self.db_path):
            self._init_db()
        rows: list[tuple[str, str, str, str, int, str, str]] = []
        now = datetime.now(timezone.utc).isoformat()
        for text_hash, vector in entries:
            if not text_hash or not vector:
                continue
            rows.append(
                (
                    embedding_profile,
                    record_kind,
                    text_hash,
                    json.dumps(vector, separators=(",", ":"), ensure_ascii=True),
                    len(vector),
                    now,
                    now,
                )
            )
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO embedding_ledger (
                    embedding_profile,
                    record_kind,
                    text_hash,
                    vector,
                    dimension,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(embedding_profile, record_kind, text_hash) DO UPDATE SET
                    vector = excluded.vector,
                    dimension = excluded.dimension,
                    updated_at = excluded.updated_at
                """,
                rows,
            )

    def clear(self) -> None:
        for path in (self.db_path, f"{self.db_path}-wal", f"{self.db_path}-shm"):
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
            except OSError as exc:
                raise wrap_io_error(
                    exc,
                    operation="delete embedding ledger artifact",
                    path=path,
                ) from exc
