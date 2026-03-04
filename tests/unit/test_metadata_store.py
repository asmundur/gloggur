from __future__ import annotations

import json
import sqlite3

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
    conn.executescript(
        """
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
        """
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
