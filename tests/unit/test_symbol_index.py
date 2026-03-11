from __future__ import annotations

import os
import sqlite3
from contextlib import closing

from gloggur.byte_spans import LineByteSpanIndex
from gloggur.config import GloggurConfig
from gloggur.parsers.registry import ParserRegistry
from gloggur.symbol_index.indexer import SymbolIndexer
from gloggur.symbol_index.store import SymbolIndexStore, SymbolIndexStoreConfig


def _config(tmp_path, *, include_minified_js: bool = False) -> GloggurConfig:
    return GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        include_minified_js=include_minified_js,
    )


def test_symbol_index_creates_expected_db_path(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def alpha(x: int) -> int:\n    return x + 1\n", encoding="utf8")
    indexer = SymbolIndexer(
        repo_root=repo,
        config=_config(tmp_path),
        parser_registry=ParserRegistry(),
    )

    result = indexer.index_path(str(repo))

    expected = repo / ".gloggur" / "index" / "symbols.db"
    assert result.db_path == str(expected)
    assert expected.exists()
    assert result.files_changed == 1


def test_symbol_index_skips_unchanged_files_incrementally(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def alpha(x: int) -> int:\n    return x + 1\n", encoding="utf8")
    indexer = SymbolIndexer(
        repo_root=repo,
        config=_config(tmp_path),
        parser_registry=ParserRegistry(),
    )

    first = indexer.index_path(str(repo))
    second = indexer.index_path(str(repo))

    assert first.files_changed == 1
    assert second.files_changed == 0
    assert second.files_unchanged == 1


def test_symbol_index_skips_minified_js_by_default_and_can_include(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.js").write_text("function appMain() { return 1; }\n", encoding="utf8")
    vendor_dir = repo / "vendor"
    vendor_dir.mkdir()
    (vendor_dir / "jquery.min.js").write_text(
        "function minifiedVendor(){return 1};",
        encoding="utf8",
    )

    default_indexer = SymbolIndexer(
        repo_root=repo,
        config=_config(tmp_path),
        parser_registry=ParserRegistry(),
    )
    default_result = default_indexer.index_path(str(repo))
    assert default_result.files_considered == 1

    include_indexer = SymbolIndexer(
        repo_root=repo,
        config=_config(tmp_path, include_minified_js=True),
        parser_registry=ParserRegistry(),
    )
    include_result = include_indexer.index_path(str(repo))
    assert include_result.files_considered == 2


def test_symbol_index_skips_structural_virtualenv_roots_and_keeps_sibling_project_code(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("def app() -> int:\n    return 1\n", encoding="utf8")
    (repo / "tools").mkdir()
    (repo / "tools" / "helper.py").write_text(
        "def helper() -> int:\n    return 2\n",
        encoding="utf8",
    )
    (repo / "_venv" / "Scripts").mkdir(parents=True)
    (repo / "_venv" / "Scripts" / "python.exe").write_text("", encoding="utf8")
    (repo / "_venv" / "Lib" / "site-packages").mkdir(parents=True)
    (repo / "_venv" / "Lib" / "site-packages" / "vendor_win.py").write_text(
        "def vendor_win() -> int:\n    return 3\n",
        encoding="utf8",
    )
    (repo / "python-runtime" / "bin").mkdir(parents=True)
    (repo / "python-runtime" / "bin" / "python").write_text("", encoding="utf8")
    (repo / "python-runtime" / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    (repo / "python-runtime" / "lib" / "python3.13" / "site-packages" / "vendor.py").write_text(
        "def vendor() -> int:\n    return 4\n",
        encoding="utf8",
    )

    indexer = SymbolIndexer(
        repo_root=repo,
        config=_config(tmp_path),
        parser_registry=ParserRegistry(),
    )
    store = SymbolIndexStore(SymbolIndexStoreConfig(repo_root=repo))

    result = indexer.index_path(str(repo))

    assert result.files_considered == 2
    paths = sorted(item.path for item in store.list_occurrences() if item.kind == "def")
    assert paths == [str(repo / "src" / "app.py"), str(repo / "tools" / "helper.py")]


def test_symbol_index_replaces_changed_file_occurrences(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "sample.py"
    source.write_text("def alpha(x: int) -> int:\n    return x + 1\n", encoding="utf8")
    indexer = SymbolIndexer(
        repo_root=repo,
        config=_config(tmp_path),
        parser_registry=ParserRegistry(),
    )
    store = SymbolIndexStore(SymbolIndexStoreConfig(repo_root=repo))
    indexer.index_path(str(repo))
    before = store.list_occurrences()
    assert any(item.kind == "def" and item.symbol == "alpha" for item in before)

    source.write_text("def beta(x: int) -> int:\n    return x + 2\n", encoding="utf8")
    indexer.index_path(str(repo))
    after = store.list_occurrences()
    assert any(item.kind == "def" and item.symbol == "beta" for item in after)
    assert not any(item.kind == "def" and item.symbol == "alpha" for item in after)


def test_symbol_index_prunes_deleted_files(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    keep_path = repo / "keep.py"
    drop_path = repo / "drop.py"
    keep_path.write_text("def keep() -> int:\n    return 1\n", encoding="utf8")
    drop_path.write_text("def drop() -> int:\n    return 2\n", encoding="utf8")
    indexer = SymbolIndexer(
        repo_root=repo,
        config=_config(tmp_path),
        parser_registry=ParserRegistry(),
    )
    store = SymbolIndexStore(SymbolIndexStoreConfig(repo_root=repo))
    indexer.index_path(str(repo))
    assert any(item.path == str(drop_path) for item in store.list_occurrences())

    os.remove(drop_path)
    result = indexer.index_path(str(repo))

    assert result.files_removed == 1
    assert not any(item.path == str(drop_path) for item in store.list_occurrences())


def test_symbol_index_defs_refs_shape_and_dedup(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "sample.py"
    path.write_text(
        "def alpha(value: int) -> int:\n"
        "    return value + value\n"
        "def beta(value: int) -> int:\n"
        "    return alpha(value)\n",
        encoding="utf8",
    )
    indexer = SymbolIndexer(
        repo_root=repo,
        config=_config(tmp_path),
        parser_registry=ParserRegistry(),
    )
    store = SymbolIndexStore(SymbolIndexStoreConfig(repo_root=repo))
    indexer.index_path(str(repo))
    occurrences = store.list_occurrences()

    defs = [item for item in occurrences if item.kind == "def"]
    refs = [item for item in occurrences if item.kind == "ref"]
    assert defs
    assert refs
    assert any(item.symbol == "alpha" for item in defs)
    tuples = {(item.symbol, item.path, item.line) for item in refs}
    assert len(tuples) == len(refs)

    span_index = LineByteSpanIndex.from_bytes(path.read_bytes())
    alpha_def = next(item for item in defs if item.symbol == "alpha")
    assert (alpha_def.start_byte, alpha_def.end_byte) == span_index.span_for_lines(1, 2)
    alpha_ref = next(item for item in refs if item.symbol == "alpha")
    assert (alpha_ref.start_byte, alpha_ref.end_byte) == span_index.span_for_lines(4, 4)


def test_symbol_index_store_rebuilds_old_schema_when_writable(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    db_path = repo / ".gloggur" / "index" / "symbols.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE occurrences (
                symbol TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                line INTEGER NOT NULL
            );
            CREATE TABLE files (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                language TEXT,
                last_indexed TEXT NOT NULL
            );
            """
        )

    store = SymbolIndexStore(SymbolIndexStoreConfig(repo_root=repo))

    assert store.available is True
    assert store.last_reset_reason is not None
    assert (
        "required tables missing" in store.last_reset_reason
        or "missing columns" in store.last_reset_reason
    )
    assert store.list_occurrences() == []


def test_symbol_index_store_marks_old_schema_unavailable_without_rebuild(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    db_path = repo / ".gloggur" / "index" / "symbols.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE occurrences (
                symbol TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                line INTEGER NOT NULL
            );
            CREATE TABLE files (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                language TEXT,
                last_indexed TEXT NOT NULL
            );
            """
        )

    store = SymbolIndexStore(
        SymbolIndexStoreConfig(repo_root=repo),
        create_if_missing=False,
    )

    assert store.available is False
    assert store.unavailability_reason is not None
    assert (
        "required tables missing" in store.unavailability_reason
        or "missing columns" in store.unavailability_reason
    )
