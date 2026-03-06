from __future__ import annotations

import time
from pathlib import Path

import pytest

from gloggur.search.router import SearchConstraints, SearchRouter, SearchRouterConfig
from gloggur.search.router.hints import extract_query_hints
from gloggur.search.router.types import BackendHit, BackendResult


def test_extract_query_hints_parses_symbols_literals_paths_and_stack_locations() -> None:
    query = 'Fix "timeout exceeded" in ServiceRunner.run for src/app.py:42 and handler.parse()'
    hints = extract_query_hints(query)

    assert "timeout exceeded" in hints.literals
    assert "ServiceRunner.run" in hints.symbols
    assert "handler.parse" in hints.symbols
    assert "src/app.py" in hints.path_hints
    assert ("src/app.py", 42) in hints.stack_locations


def test_extract_query_hints_keeps_short_quoted_identifiers() -> None:
    hints = extract_query_hints('find uses of "id" in parser')

    assert "id" in hints.symbols
    assert "id" in hints.identifier_tokens


def test_search_router_auto_prefers_exact_on_quality_tie(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact_result = BackendResult(
        name="exact",
        hits=(
            BackendHit(
                backend="exact",
                path="src/exact.py",
                start_line=10,
                end_line=10,
                snippet="def exact_hit():\n    pass",
                raw_score=0.8,
                tags=("literal_match",),
            ),
        ),
        timing_ms=20,
    )
    semantic_result = BackendResult(
        name="semantic",
        hits=(
            BackendHit(
                backend="semantic",
                path="src/semantic.py",
                start_line=20,
                end_line=20,
                snippet="def semantic_hit():\n    pass",
                raw_score=0.8,
                tags=("semantic_match",),
            ),
        ),
        timing_ms=10,
    )

    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: exact_result,
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_semantic_backend",
        lambda **_kwargs: semantic_result,
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda *_args, **_kwargs: 0.9,
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=object(),  # semantic backend enabled via monkeypatch
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact", "semantic")),
    )
    pack = router.search(query="find exact symbol", mode="auto")

    assert pack.summary["strategy"] == "exact"
    assert pack.hits
    assert pack.hits[0].path == "src/exact.py"


def test_search_router_hybrid_merges_when_threshold_not_met(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact_result = BackendResult(
        name="exact",
        hits=(
            BackendHit(
                backend="exact",
                path="src/a.py",
                start_line=1,
                end_line=1,
                snippet="def a(): pass",
                raw_score=0.6,
                tags=("literal_match",),
            ),
        ),
        timing_ms=12,
    )
    semantic_result = BackendResult(
        name="semantic",
        hits=(
            BackendHit(
                backend="semantic",
                path="src/b.py",
                start_line=2,
                end_line=2,
                snippet="def b(): pass",
                raw_score=0.6,
                tags=("semantic_match",),
            ),
        ),
        timing_ms=13,
    )

    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: exact_result,
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_semantic_backend",
        lambda **_kwargs: semantic_result,
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda *_args, **_kwargs: 0.1,
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=object(),
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact", "semantic"), max_snippets=4),
    )
    pack = router.search(query="ambiguous query", mode="auto")

    assert pack.summary["strategy"] == "hybrid"
    paths = {hit.path for hit in pack.hits}
    assert "src/a.py" in paths
    assert "src/b.py" in paths


def test_search_router_output_is_deterministic_for_same_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact_result = BackendResult(
        name="exact",
        hits=(
            BackendHit(
                backend="exact",
                path="src/z.py",
                start_line=3,
                end_line=3,
                snippet="def z(): pass",
                raw_score=0.8,
                tags=("literal_match",),
            ),
            BackendHit(
                backend="exact",
                path="src/a.py",
                start_line=1,
                end_line=1,
                snippet="def a(): pass",
                raw_score=0.8,
                tags=("literal_match",),
            ),
        ),
        timing_ms=10,
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: exact_result,
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda *_args, **_kwargs: 0.95,
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=None,
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact",)),
    )
    constraints = SearchConstraints(max_snippets=2, max_files=2)
    pack_a = router.search(query="stable ordering", constraints=constraints, mode="auto")
    pack_b = router.search(query="stable ordering", constraints=constraints, mode="auto")

    assert [hit.to_dict() for hit in pack_a.hits] == [hit.to_dict() for hit in pack_b.hits]


def test_search_router_forced_exact_bypasses_threadpool_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact_result = BackendResult(
        name="exact",
        hits=(
            BackendHit(
                backend="exact",
                path="src/slow.py",
                start_line=4,
                end_line=4,
                snippet="def caller(): pass",
                raw_score=0.9,
                tags=("literal_match",),
            ),
        ),
        timing_ms=200,
    )

    def _slow_exact_backend(**_kwargs):
        time.sleep(0.2)
        return exact_result

    monkeypatch.setattr("gloggur.search.router.engine.run_exact_backend", _slow_exact_backend)

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=None,
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact",), default_time_budget_ms=50),
    )
    pack = router.search(query="caller", mode="exact")

    assert pack.summary["strategy"] == "exact"
    assert pack.hits
    assert pack.hits[0].path == "src/slow.py"


def test_search_router_keeps_quoted_short_grep_pattern_for_symbol_hints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _symbol_backend(**kwargs):
        hints = kwargs["hints"]
        assert "id" in hints.symbols
        return BackendResult(
            name="symbol",
            hits=(
                BackendHit(
                    backend="symbol",
                    path="src/symbols.py",
                    start_line=7,
                    end_line=7,
                    snippet="id = get_id()",
                    raw_score=0.9,
                    tags=("symbol_ref",),
                ),
            ),
            timing_ms=5,
        )

    monkeypatch.setattr("gloggur.search.router.engine.run_symbol_backend", _symbol_backend)
    router = SearchRouter(
        repo_root=tmp_path,
        searcher=None,
        metadata_store=None,
        symbol_store=object(),
        config=SearchRouterConfig(enabled_backends=("symbol",)),
    )
    pack = router.search(query='rg "id"', mode="auto", include_debug=True)

    assert pack.hits
    assert pack.hits[0].path == "src/symbols.py"
    assert isinstance(pack.debug, dict)
    parsed = pack.debug.get("parsed_query")
    assert isinstance(parsed, dict)
    assert parsed.get("pattern_quoted") is True
