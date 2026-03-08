from __future__ import annotations

from gloggur.byte_spans import LineByteSpanIndex
from gloggur.search.router.backends import run_symbol_backend
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.hints import extract_query_hints
from gloggur.search.router.types import ExecutionHints, SearchIntent
from gloggur.symbol_index.models import IndexedFile, SymbolOccurrence
from gloggur.symbol_index.store import SymbolIndexStore, SymbolIndexStoreConfig


def _build_symbol_store(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / "sample.py"
    file_path.write_text(
        "def Foo() -> int:\n"
        "    return 1\n\n"
        "def caller() -> int:\n"
        "    return Foo() + Foo()\n",
        encoding="utf8",
    )
    span_index = LineByteSpanIndex.from_bytes(file_path.read_bytes())
    store = SymbolIndexStore(SymbolIndexStoreConfig(repo_root=repo))
    store.replace_file_occurrences(
        indexed_file=IndexedFile(
            path=str(file_path),
            content_hash="hash-1",
            mtime_ns=1,
            language="python",
        ),
        occurrences=[
            SymbolOccurrence(
                symbol="Foo",
                kind="def",
                path=str(file_path),
                start_line=1,
                end_line=2,
                start_byte=span_index.span_for_lines(1, 2)[0],
                end_byte=span_index.span_for_lines(1, 2)[1],
                language="python",
                signature="def Foo() -> int:",
            ),
            SymbolOccurrence(
                symbol="Foo",
                kind="ref",
                path=str(file_path),
                start_line=5,
                end_line=5,
                start_byte=span_index.span_for_lines(5, 5)[0],
                end_byte=span_index.span_for_lines(5, 5)[1],
                language="python",
            ),
            SymbolOccurrence(
                symbol="Foo",
                kind="ref",
                path=str(file_path),
                start_line=5,
                end_line=5,
                start_byte=span_index.span_for_lines(5, 5)[0],
                end_byte=span_index.span_for_lines(5, 5)[1],
                language="python",
            ),
        ],
    )
    return repo, store


def test_symbol_backend_prefers_definition_for_non_usage_queries(tmp_path) -> None:
    repo, store = _build_symbol_store(tmp_path)
    query = "where is Foo defined"
    result = run_symbol_backend(
        symbol_store=store,
        hints=extract_query_hints(query),
        query=query,
        repo_root=repo,
        intent=SearchIntent(),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert result.hits
    assert "symbol_def" in result.hits[0].tags


def test_symbol_backend_prefers_references_for_usage_queries(tmp_path) -> None:
    repo, store = _build_symbol_store(tmp_path)
    query = "who calls Foo"
    result = run_symbol_backend(
        symbol_store=store,
        hints=extract_query_hints(query),
        query=query,
        repo_root=repo,
        intent=SearchIntent(),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert result.hits
    assert "symbol_ref" in result.hits[0].tags


def test_symbol_backend_emits_def_and_ref_tags(tmp_path) -> None:
    repo, store = _build_symbol_store(tmp_path)
    query = "Foo"
    result = run_symbol_backend(
        symbol_store=store,
        hints=extract_query_hints(query),
        query=query,
        repo_root=repo,
        intent=SearchIntent(max_snippets=10),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    tag_set = {tag for hit in result.hits for tag in hit.tags}
    assert "symbol_def" in tag_set
    assert "symbol_ref" in tag_set
    assert all(hit.start_byte is not None for hit in result.hits)
    assert all(hit.end_byte is not None for hit in result.hits)


def test_symbol_backend_reports_error_when_symbol_index_missing(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = SymbolIndexStore(
        SymbolIndexStoreConfig(repo_root=repo),
        create_if_missing=False,
    )
    query = "where is Foo defined"
    result = run_symbol_backend(
        symbol_store=store,
        hints=extract_query_hints(query),
        query=query,
        repo_root=repo,
        intent=SearchIntent(),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert result.hits == ()
    assert result.error is not None
    assert "symbol index unavailable" in result.error


def test_symbol_backend_reports_error_when_symbol_index_is_corrupt(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    db_path = repo / ".gloggur" / "index" / "symbols.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"not-a-sqlite-db")
    store = SymbolIndexStore(
        SymbolIndexStoreConfig(repo_root=repo),
        create_if_missing=False,
    )
    query = "where is Foo defined"
    result = run_symbol_backend(
        symbol_store=store,
        hints=extract_query_hints(query),
        query=query,
        repo_root=repo,
        intent=SearchIntent(),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert result.hits == ()
    assert result.error is not None
    assert "symbol index unavailable" in result.error
