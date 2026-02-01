from __future__ import annotations

import json
import sqlite3

from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig


def _seed_symbols(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE symbols (
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
        """
    )
    conn.execute(
        """
        INSERT INTO symbols (
            id, name, kind, file_path, start_line, end_line,
            signature, docstring, body_hash, embedding_vector, language
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sym-1",
            "alpha",
            "function",
            "alpha.py",
            1,
            2,
            "def alpha():",
            "Alpha doc",
            "hash-1",
            json.dumps([0.1, 0.2]),
            "python",
        ),
    )
    conn.execute(
        """
        INSERT INTO symbols (
            id, name, kind, file_path, start_line, end_line,
            signature, docstring, body_hash, embedding_vector, language
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sym-2",
            "beta",
            "class",
            "beta.py",
            10,
            20,
            "class Beta:",
            None,
            "hash-2",
            None,
            "python",
        ),
    )
    conn.commit()
    conn.close()


def test_metadata_store_get_list_and_filter(tmp_path) -> None:
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
