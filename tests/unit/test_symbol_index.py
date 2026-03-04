from __future__ import annotations

import os

from gloggur.config import GloggurConfig
from gloggur.parsers.registry import ParserRegistry
from gloggur.symbol_index.indexer import SymbolIndexer
from gloggur.symbol_index.store import SymbolIndexStore, SymbolIndexStoreConfig


def _config(tmp_path) -> GloggurConfig:
    return GloggurConfig(cache_dir=str(tmp_path / "cache"))


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
