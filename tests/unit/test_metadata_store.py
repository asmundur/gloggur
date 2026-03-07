from __future__ import annotations

import json
import sqlite3

import pytest

import gloggur.storage.metadata_store as metadata_store_module
from gloggur.io_failures import StorageIOError
from gloggur.models import Symbol
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig


def _insert_symbol(
    conn: sqlite3.Connection,
    *,
    symbol_id: str,
    name: str,
    file_path: str,
    start_line: int = 1,
    kind: str = "function",
) -> None:
    """Insert one symbol row into the metadata test table."""
    conn.execute(
        """
        INSERT INTO symbols (
            id, name, kind, fqname, file_path, start_line, end_line,
            container_id, container_fqname, signature, docstring, body_hash, embedding_vector,
            language, repo_id, commit_hash, visibility, exported, tokens_estimate, invariants,
            calls, covered_by, is_serialization_boundary, implicit_contract, signals, attributes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol_id,
            name,
            kind,
            name,
            file_path,
            start_line,
            start_line + 1,
            None,
            None,
            f"def {name}():",
            None,
            f"hash-{symbol_id}",
            None,
            "python",
            "repo-test",
            "commit-test",
            "public",
            1,
            16,
            "[]",
            "[]",
            "[]",
            0,
            None,
            "[]",
            "{}",
        ),
    )


def _seed_symbols(db_path: str) -> None:
    """Seed a metadata store database with sample symbols."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE symbols (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            fqname TEXT,
            file_path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            container_id TEXT,
            container_fqname TEXT,
            signature TEXT,
            docstring TEXT,
            body_hash TEXT NOT NULL,
            embedding_vector TEXT,
            language TEXT,
            repo_id TEXT,
            commit_hash TEXT,
            visibility TEXT,
            exported INTEGER,
            tokens_estimate INTEGER,
            invariants TEXT,
            calls TEXT,
            covered_by TEXT,
            is_serialization_boundary INTEGER NOT NULL DEFAULT 0,
            implicit_contract TEXT,
            signals TEXT,
            attributes TEXT
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            symbol_id TEXT NOT NULL,
            chunk_part_index INTEGER NOT NULL,
            chunk_part_total INTEGER NOT NULL,
            text TEXT,
            file_path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            start_byte INTEGER,
            end_byte INTEGER,
            tokens_estimate INTEGER,
            language TEXT,
            repo_id TEXT,
            commit_hash TEXT,
            embedding_vector TEXT
        );
        CREATE TABLE edges (
            edge_id TEXT PRIMARY KEY,
            edge_type TEXT NOT NULL,
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            from_kind TEXT NOT NULL,
            to_kind TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line INTEGER NOT NULL,
            confidence REAL NOT NULL,
            repo_id TEXT,
            commit_hash TEXT,
            text TEXT,
            embedding_vector TEXT
        );
        """)
    conn.execute(
        """
        INSERT INTO symbols (
            id, name, kind, fqname, file_path, start_line, end_line,
            container_id, container_fqname, signature, docstring, body_hash, embedding_vector,
            language, repo_id, commit_hash, visibility, exported, tokens_estimate, invariants,
            calls, covered_by, is_serialization_boundary, implicit_contract, signals, attributes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sym-1",
            "alpha",
            "function",
            "alpha",
            "alpha.py",
            1,
            2,
            None,
            None,
            "def alpha():",
            "Alpha doc",
            "hash-1",
            json.dumps([0.1, 0.2]),
            "python",
            "repo-test",
            "commit-test",
            "public",
            1,
            16,
            "[]",
            "[]",
            "[]",
            0,
            None,
            "[]",
            "{}",
        ),
    )
    conn.execute(
        """
        INSERT INTO symbols (
            id, name, kind, fqname, file_path, start_line, end_line,
            container_id, container_fqname, signature, docstring, body_hash, embedding_vector,
            language, repo_id, commit_hash, visibility, exported, tokens_estimate, invariants,
            calls, covered_by, is_serialization_boundary, implicit_contract, signals, attributes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sym-2",
            "beta",
            "class",
            "beta",
            "beta.py",
            10,
            20,
            None,
            None,
            "class Beta:",
            None,
            "hash-2",
            None,
            "python",
            "repo-test",
            "commit-test",
            "public",
            1,
            16,
            "[]",
            "[]",
            "[]",
            0,
            None,
            "[]",
            "{}",
        ),
    )
    conn.commit()
    conn.close()


def _insert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    symbol_id: str,
    file_path: str,
    start_line: int,
    text: str,
    embedding_vector: list[float] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, symbol_id, chunk_part_index, chunk_part_total, text, file_path,
            start_line, end_line, start_byte, end_byte, tokens_estimate, language,
            repo_id, commit_hash, embedding_vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            symbol_id,
            0,
            1,
            text,
            file_path,
            start_line,
            start_line + 1,
            None,
            None,
            8,
            "python",
            "repo-test",
            "commit-test",
            json.dumps(embedding_vector) if embedding_vector is not None else None,
        ),
    )


def _insert_edge(
    conn: sqlite3.Connection,
    *,
    edge_id: str,
    from_id: str,
    to_id: str,
    file_path: str,
    line: int,
    edge_type: str = "calls",
) -> None:
    conn.execute(
        """
        INSERT INTO edges (
            edge_id, edge_type, from_id, to_id, from_kind, to_kind, file_path, line,
            confidence, repo_id, commit_hash, text, embedding_vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            edge_id,
            edge_type,
            from_id,
            to_id,
            "function",
            "function",
            file_path,
            line,
            0.8,
            "repo-test",
            "commit-test",
            f"{from_id}->{to_id}",
            None,
        ),
    )


def test_metadata_store_get_list_and_filter(tmp_path) -> None:
    """Metadata store should get, list, and filter symbols."""
    db_path = tmp_path / "index.db"
    _seed_symbols(str(db_path))

    store = MetadataStore(MetadataStoreConfig(cache_dir=str(tmp_path)))

    symbol = store.get_symbol("sym-1")
    assert symbol is not None
    assert symbol.name == "alpha"
    assert symbol.embedding_vector == [0.1, 0.2]

    symbols = store.list_symbols()
    assert [item.id for item in symbols] == ["sym-1", "sym-2"]

    filtered = store.filter_symbols(kinds=["class"])
    assert [item.id for item in filtered] == ["sym-2"]

    filtered = store.filter_symbols(file_path="alpha.py", language="python")
    assert [item.id for item in filtered] == ["sym-1"]


def test_metadata_store_file_match_supports_exact_prefix_and_like_escaping(tmp_path) -> None:
    """File-match filtering should support boundary-safe prefixes with escaped SQL LIKE."""
    db_path = tmp_path / "index.db"
    _seed_symbols(str(db_path))
    conn = sqlite3.connect(db_path)
    try:
        _insert_symbol(
            conn,
            symbol_id="sym-3",
            name="src_target",
            file_path="./src/requests/models.py",
        )
        _insert_symbol(
            conn,
            symbol_id="sym-4",
            name="src2_target",
            file_path="./src2/requests/models.py",
        )
        _insert_symbol(
            conn,
            symbol_id="sym-5",
            name="pct_literal",
            file_path="./src/%tmp/example.py",
        )
        _insert_symbol(
            conn,
            symbol_id="sym-6",
            name="pct_nonliteral",
            file_path="./src/xtmp/example.py",
        )
        conn.commit()
    finally:
        conn.close()

    store = MetadataStore(MetadataStoreConfig(cache_dir=str(tmp_path)))

    src_results = store.filter_symbols_by_file_match(file_path="src")
    src_ids = [item.id for item in src_results]
    assert "sym-3" in src_ids
    assert "sym-4" not in src_ids

    src_dot_results = store.filter_symbols_by_file_match(file_path="./src/")
    src_dot_ids = [item.id for item in src_dot_results]
    assert "sym-3" in src_dot_ids
    assert "sym-4" not in src_dot_ids

    escaped_results = store.filter_symbols_by_file_match(file_path="./src/%tmp")
    escaped_ids = [item.id for item in escaped_results]
    assert "sym-5" in escaped_ids
    assert "sym-6" not in escaped_ids


def test_metadata_store_chunk_and_edge_queries_are_filtered_and_deterministic(tmp_path) -> None:
    db_path = tmp_path / "index.db"
    _seed_symbols(str(db_path))
    conn = sqlite3.connect(db_path)
    try:
        _insert_chunk(
            conn,
            chunk_id="chunk-1",
            symbol_id="sym-1",
            file_path="alpha.py",
            start_line=1,
            text="def alpha():\n    return 1",
            embedding_vector=[0.1, 0.2],
        )
        _insert_chunk(
            conn,
            chunk_id="chunk-2",
            symbol_id="sym-2",
            file_path="beta.py",
            start_line=10,
            text="class Beta:\n    pass",
        )
        _insert_edge(
            conn,
            edge_id="edge-1",
            from_id="sym-1",
            to_id="sym-2",
            file_path="alpha.py",
            line=3,
        )
        _insert_edge(
            conn,
            edge_id="edge-2",
            from_id="sym-2",
            to_id="sym-1",
            file_path="beta.py",
            line=11,
            edge_type="references",
        )
        conn.commit()
    finally:
        conn.close()

    store = MetadataStore(MetadataStoreConfig(cache_dir=str(tmp_path)))

    chunks = store.list_chunks()
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert store.get_chunk("chunk-1") is not None
    assert store.get_chunk("missing") is None
    assert [chunk.chunk_id for chunk in store.list_chunks_for_symbol("sym-1")] == ["chunk-1"]
    assert [chunk.chunk_id for chunk in store.filter_chunks_by_path("beta.py")] == ["chunk-2"]

    edges = store.list_edges()
    assert [edge.edge_id for edge in edges] == ["edge-1", "edge-2"]
    assert [edge.edge_id for edge in store.list_edges_for_file("beta.py")] == ["edge-2"]
    assert [
        edge.edge_id for edge in store.list_edges_for_symbol("sym-1", direction="incoming")
    ] == ["edge-2"]
    assert [
        edge.edge_id for edge in store.list_edges_for_symbol("sym-1", direction="outgoing")
    ] == ["edge-1"]
    assert [
        edge.edge_id
        for edge in store.list_edges_for_symbol("sym-1", edge_type="references", limit=1)
    ] == ["edge-2"]


def test_metadata_store_upsert_symbol_insert_update_and_delete(tmp_path) -> None:
    db_path = tmp_path / "index.db"
    _seed_symbols(str(db_path))
    store = MetadataStore(MetadataStoreConfig(cache_dir=str(tmp_path)))
    symbol = Symbol(
        id="sym-upsert",
        name="gamma",
        kind="function",
        fqname="pkg.gamma",
        file_path="gamma.py",
        start_line=4,
        end_line=8,
        body_hash="hash-gamma",
        signature="def gamma() -> int:",
        language="python",
    )

    store.upsert_symbol(symbol)
    inserted = store.get_symbol("sym-upsert")

    assert inserted is not None
    assert inserted.docstring is None
    assert inserted.calls == []

    store.upsert_symbol(
        symbol.model_copy(
            update={
                "docstring": "Updated doc",
                "body_hash": "hash-gamma-v2",
                "exported": True,
            }
        )
    )
    updated = store.get_symbol("sym-upsert")

    assert updated is not None
    assert updated.docstring == "Updated doc"
    assert updated.body_hash == "hash-gamma-v2"
    assert updated.exported is True
    assert store.delete_symbol("sym-upsert") is True
    assert store.get_symbol("sym-upsert") is None
    assert store.delete_symbol("sym-upsert") is False


def test_metadata_store_upsert_symbol_rejects_non_symbol_payload(tmp_path) -> None:
    db_path = tmp_path / "index.db"
    _seed_symbols(str(db_path))
    store = MetadataStore(MetadataStoreConfig(cache_dir=str(tmp_path)))

    with pytest.raises(TypeError, match="Symbol instance"):
        store.upsert_symbol({"id": "sym-bad"})  # type: ignore[arg-type]


def test_metadata_store_row_normalization_handles_optional_fields_and_invalid_attributes(
    tmp_path,
) -> None:
    db_path = tmp_path / "index.db"
    _seed_symbols(str(db_path))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO symbols (
                id, name, kind, fqname, file_path, start_line, end_line,
                container_id, container_fqname, signature, docstring, body_hash, embedding_vector,
                language, repo_id, commit_hash, visibility, exported, tokens_estimate, invariants,
                calls, covered_by, is_serialization_boundary, implicit_contract, signals, attributes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sym-optional",
                "optional",
                "function",
                None,
                "optional.py",
                2,
                4,
                None,
                None,
                None,
                None,
                "hash-optional",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
                "[]",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    store = MetadataStore(MetadataStoreConfig(cache_dir=str(tmp_path)))
    symbol = store.get_symbol("sym-optional")

    assert symbol is not None
    assert symbol.exported is None
    assert symbol.tokens_estimate is None
    assert symbol.invariants == []
    assert symbol.attributes == {}


def test_metadata_store_wraps_connect_and_pragma_failures(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MetadataStore(MetadataStoreConfig(cache_dir=str(tmp_path)))

    def raise_on_connect(*args, **kwargs):
        raise sqlite3.DatabaseError("open failed")

    monkeypatch.setattr(metadata_store_module.sqlite3, "connect", raise_on_connect)

    with pytest.raises(StorageIOError) as connect_error:
        store.get_symbol("sym-1")

    assert connect_error.value.operation == "open metadata database connection"

    class FakePragmaConnection:
        def __init__(self) -> None:
            self.closed = False
            self.row_factory = None

        def execute(self, query: str, params: object = ()) -> object:
            _ = params
            if query.startswith("PRAGMA"):
                raise sqlite3.DatabaseError("pragma failed")
            return []

        def close(self) -> None:
            self.closed = True

    fake_conn = FakePragmaConnection()
    monkeypatch.setattr(metadata_store_module.sqlite3, "connect", lambda *args, **kwargs: fake_conn)

    with pytest.raises(StorageIOError) as pragma_error:
        store.get_symbol("sym-1")

    assert pragma_error.value.operation == "configure metadata database pragmas"
    assert fake_conn.closed is True


def test_metadata_store_wraps_transaction_failures_and_rolls_back(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MetadataStore(MetadataStoreConfig(cache_dir=str(tmp_path)))

    class FakeConnection:
        def __init__(self) -> None:
            self.rollback_calls = 0
            self.row_factory = None

        def execute(self, query: str, params: object = ()) -> object:
            _ = params
            if query.startswith("PRAGMA"):
                return []
            raise sqlite3.DatabaseError("write failed")

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            self.rollback_calls += 1

        def close(self) -> None:
            return None

    fake_conn = FakeConnection()
    monkeypatch.setattr(metadata_store_module.sqlite3, "connect", lambda *args, **kwargs: fake_conn)

    with pytest.raises(StorageIOError) as error:
        store.upsert_symbol(
            Symbol(
                id="sym-write",
                name="write",
                kind="function",
                file_path="write.py",
                start_line=1,
                end_line=2,
                body_hash="hash-write",
            )
        )

    assert error.value.operation == "execute metadata database transaction"
    assert fake_conn.rollback_calls == 1
