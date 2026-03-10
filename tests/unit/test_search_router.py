from __future__ import annotations

import time
from pathlib import Path

import pytest

from gloggur.search.router import SearchConstraints, SearchIntent, SearchRouter, SearchRouterConfig
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


def test_extract_query_hints_strips_declaration_prefixes() -> None:
    hints = extract_query_hints("def escape_leading_slashes")

    assert hints.declaration_terms == ("def",)
    assert hints.query_kind == "declaration"
    assert "escape_leading_slashes" in hints.symbols
    assert "def" not in hints.symbols


@pytest.mark.parametrize(
    ("query", "expected_kind"),
    [
        ("Session.mount", "identifier"),
        ("url_has_allowed_host_and_scheme", "identifier"),
        ("after_request order", "mixed"),
    ],
)
def test_extract_query_hints_classifies_query_kinds(query: str, expected_kind: str) -> None:
    hints = extract_query_hints(query)

    assert hints.query_kind == expected_kind


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


def test_search_router_identifier_queries_stay_on_non_semantic_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_order: list[str] = []

    def _exact_backend(**kwargs):
        call_order.append("exact")
        execution_hints = kwargs["execution_hints"]
        assert execution_hints.fixed_string is True
        return BackendResult(
            name="exact",
            hits=(
                BackendHit(
                    backend="exact",
                    path="src/session.py",
                    start_line=12,
                    end_line=12,
                    snippet="def mount(self, prefix, adapter):",
                    raw_score=0.97,
                    tags=("literal_match",),
                ),
            ),
            timing_ms=4,
        )

    def _symbol_backend(**_kwargs):
        call_order.append("symbol")
        return BackendResult(name="symbol", hits=(), timing_ms=3)

    def _semantic_backend(**_kwargs):
        raise AssertionError("semantic fallback should not run for decisive identifier queries")

    monkeypatch.setattr("gloggur.search.router.engine.run_exact_backend", _exact_backend)
    monkeypatch.setattr("gloggur.search.router.engine.run_symbol_backend", _symbol_backend)
    monkeypatch.setattr("gloggur.search.router.engine.run_semantic_backend", _semantic_backend)
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda result, **_kwargs: 0.95 if result.name == "exact" else 0.0,
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=object(),
        metadata_store=None,
        symbol_store=object(),
        config=SearchRouterConfig(enabled_backends=("exact", "symbol", "semantic")),
    )
    pack = router.search(query="Session.mount", mode="auto")

    assert call_order == ["exact", "symbol"]
    assert pack.summary["query_kind"] == "identifier"
    assert pack.summary["strategy"] == "exact"
    assert pack.summary["decisive"] is True
    assert pack.summary["next_action"] == "open_hit_1"


def test_search_router_declaration_queries_can_choose_symbol_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: BackendResult(name="exact", hits=(), timing_ms=3),
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_symbol_backend",
        lambda **_kwargs: BackendResult(
            name="symbol",
            hits=(
                BackendHit(
                    backend="symbol",
                    path="src/django/utils/http.py",
                    start_line=42,
                    end_line=42,
                    snippet="def escape_leading_slashes(url):",
                    raw_score=0.99,
                    tags=("symbol_def",),
                ),
            ),
            timing_ms=4,
        ),
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda result, **_kwargs: 0.97 if result.name == "symbol" else 0.0,
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=object(),
        metadata_store=None,
        symbol_store=object(),
        config=SearchRouterConfig(enabled_backends=("exact", "symbol", "semantic")),
    )
    pack = router.search(query="def escape_leading_slashes", mode="auto")

    assert pack.summary["query_kind"] == "declaration"
    assert pack.summary["strategy"] == "symbol"
    assert pack.summary["decisive"] is True
    assert pack.summary["next_action"] == "open_hit_1"


def test_search_router_mixed_query_suggests_path_narrowing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: BackendResult(
            name="exact",
            hits=(
                BackendHit(
                    backend="exact",
                    path="src/flask/app.py",
                    start_line=110,
                    end_line=110,
                    snippet="ctx._after_request_functions.append(func)",
                    raw_score=0.90,
                    tags=("literal_match",),
                ),
                BackendHit(
                    backend="exact",
                    path="src/flask/app.py",
                    start_line=144,
                    end_line=144,
                    snippet="for func in ctx._after_request_functions:",
                    raw_score=0.88,
                    tags=("literal_match",),
                ),
            ),
            timing_ms=5,
        ),
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_symbol_backend",
        lambda **_kwargs: BackendResult(name="symbol", hits=(), timing_ms=2),
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda result, **_kwargs: 0.92 if result.name == "exact" else 0.0,
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=object(),
        metadata_store=None,
        symbol_store=object(),
        config=SearchRouterConfig(enabled_backends=("exact", "symbol", "semantic")),
    )
    pack = router.search(query="after_request order", mode="auto")

    assert pack.summary["query_kind"] == "mixed"
    assert pack.summary["strategy"] == "exact"
    assert pack.summary["decisive"] is False
    assert pack.summary["next_action"] == "narrow_by_path"
    assert pack.summary["suggested_path_prefix"] == "src/flask/app.py"


def test_search_router_skips_semantic_factory_for_decisive_identifier_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    semantic_factory_calls: list[str] = []
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: BackendResult(
            name="exact",
            hits=(
                BackendHit(
                    backend="exact",
                    path="src/django/http.py",
                    start_line=21,
                    end_line=21,
                    snippet="def url_has_allowed_host_and_scheme(url, allowed_hosts):",
                    raw_score=0.97,
                    tags=("literal_match",),
                ),
            ),
            timing_ms=4,
        ),
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_symbol_backend",
        lambda **_kwargs: BackendResult(name="symbol", hits=(), timing_ms=3),
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda result, **_kwargs: 0.96 if result.name == "exact" else 0.0,
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=None,
        metadata_store=None,
        symbol_store=object(),
        config=SearchRouterConfig(enabled_backends=("exact", "symbol", "semantic")),
        searcher_factory=lambda: (semantic_factory_calls.append("semantic") or object(), None),
    )
    pack = router.search(query="url_has_allowed_host_and_scheme", mode="auto")

    assert semantic_factory_calls == []
    assert pack.summary["query_kind"] == "identifier"
    assert pack.summary["next_action"] == "open_hit_1"


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


def test_search_router_hybrid_uses_rank_fusion_for_final_ordering(
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
                snippet="def alpha(): pass",
                raw_score=0.9,
                tags=("literal_match",),
            ),
            BackendHit(
                backend="exact",
                path="src/b.py",
                start_line=2,
                end_line=2,
                snippet="def beta(): pass",
                raw_score=0.8,
                tags=("literal_match",),
            ),
        ),
        timing_ms=9,
    )
    semantic_result = BackendResult(
        name="semantic",
        hits=(
            BackendHit(
                backend="semantic",
                path="src/b.py",
                start_line=2,
                end_line=2,
                snippet="def beta(): pass",
                raw_score=0.95,
                tags=("semantic_match", "semantic_high_conf"),
            ),
            BackendHit(
                backend="semantic",
                path="src/c.py",
                start_line=3,
                end_line=3,
                snippet="def gamma(): pass",
                raw_score=0.9,
                tags=("semantic_match",),
            ),
        ),
        timing_ms=11,
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
        config=SearchRouterConfig(enabled_backends=("exact", "semantic"), max_snippets=3),
    )
    pack_a = router.search(query="ambiguous query", mode="auto")
    pack_b = router.search(query="ambiguous query", mode="auto")

    assert pack_a.summary["strategy"] == "hybrid"
    assert pack_a.hits
    assert pack_a.hits[0].path == "src/b.py"
    assert isinstance(pack_a.debug, dict)
    assert "backend_weights" in pack_a.debug
    assert "fusion_budget" in pack_a.debug
    assert "eligibility" in pack_a.debug
    assert [hit.to_dict() for hit in pack_a.hits] == [hit.to_dict() for hit in pack_b.hits]


def test_search_router_suppresses_ungrounded_semantic_only_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Semantic-only below-threshold hits should be suppressed instead of merged into junk."""
    semantic_result = BackendResult(
        name="semantic",
        hits=(
            BackendHit(
                backend="semantic",
                path="src/engine/oauth_integration_engine.py",
                start_line=91,
                end_line=92,
                snippet="class OAuthStartResult:\n    auth_url: str",
                raw_score=0.91,
                tags=("semantic_match", "semantic_high_conf"),
            ),
        ),
        timing_ms=8,
    )

    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: BackendResult(name="exact", hits=(), timing_ms=3),
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_semantic_backend",
        lambda **_kwargs: semantic_result,
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda result, **_kwargs: 0.2 if result.name == "semantic" else 0.0,
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=object(),
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact", "semantic")),
    )
    pack = router.search(query="Database setup", mode="auto")

    assert pack.summary["strategy"] == "suppressed"
    assert pack.summary["warning_codes"] == ["ungrounded_results_suppressed"]
    assert pack.hits == ()


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


def test_search_router_forced_hybrid_strategy_remains_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: BackendResult(
            name="exact",
            hits=(
                BackendHit(
                    backend="exact",
                    path="src/exact.py",
                    start_line=5,
                    end_line=5,
                    snippet="def exact(): pass",
                    raw_score=0.8,
                    tags=("literal_match",),
                ),
            ),
            timing_ms=4,
        ),
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_semantic_backend",
        lambda **_kwargs: BackendResult(
            name="semantic",
            hits=(
                BackendHit(
                    backend="semantic",
                    path="src/semantic.py",
                    start_line=8,
                    end_line=8,
                    snippet="def semantic(): pass",
                    raw_score=0.9,
                    tags=("semantic_match",),
                ),
            ),
            timing_ms=5,
        ),
    )

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=object(),
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact", "semantic")),
    )
    pack = router.search(query="forced hybrid", mode="hybrid")

    assert pack.summary["strategy"] == "hybrid"
    assert pack.summary["reason"] == "forced_mode"
    assert {hit.path for hit in pack.hits} == {"src/exact.py", "src/semantic.py"}


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


def test_search_router_search_intent_path_is_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _exact_backend(**kwargs):
        captured.update(kwargs)
        return BackendResult(
            name="exact",
            hits=(
                BackendHit(
                    backend="exact",
                    path="src/intent.py",
                    start_line=11,
                    end_line=11,
                    snippet="def intent_path(): pass",
                    raw_score=0.9,
                    tags=("literal_match",),
                ),
            ),
            timing_ms=5,
        )

    monkeypatch.setattr("gloggur.search.router.engine.run_exact_backend", _exact_backend)
    router = SearchRouter(
        repo_root=tmp_path,
        searcher=None,
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact",)),
    )
    intent = SearchIntent(path_prefix="src", max_snippets=1, time_budget_ms=250)
    pack = router.search(query="intent path", intent=intent, mode="exact")

    assert pack.hits
    assert isinstance(captured.get("intent"), SearchIntent)
    captured_intent = captured["intent"]
    assert isinstance(captured_intent, SearchIntent)
    assert captured_intent.path_prefix == "src"
    assert not hasattr(captured_intent, "case_mode")
    assert not hasattr(captured_intent, "include_globs")
    captured_hints = captured.get("execution_hints")
    assert captured_hints is not None
    assert getattr(captured_hints, "include_globs", ()) == ()


def test_search_router_search_constraints_adapter_preserves_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = BackendResult(
        name="exact",
        hits=(
            BackendHit(
                backend="exact",
                path="src/parity.py",
                start_line=1,
                end_line=1,
                snippet="def parity(): pass",
                raw_score=0.8,
                tags=("literal_match",),
            ),
        ),
        timing_ms=3,
    )
    monkeypatch.setattr(
        "gloggur.search.router.engine.run_exact_backend",
        lambda **_kwargs: result,
    )
    router = SearchRouter(
        repo_root=tmp_path,
        searcher=None,
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact",)),
    )
    intent = SearchIntent(path_prefix="src", max_snippets=1, max_files=1, time_budget_ms=200)
    constraints = SearchConstraints(path_prefix="src", max_snippets=1, max_files=1, time_budget_ms=200)

    pack_with_intent = router.search(query="parity", intent=intent, mode="exact")
    pack_with_constraints = router.search(query="parity", constraints=constraints, mode="exact")

    assert [hit.to_dict() for hit in pack_with_intent.hits] == [
        hit.to_dict() for hit in pack_with_constraints.hits
    ]
    assert pack_with_intent.summary == pack_with_constraints.summary


def test_query_compat_flags_map_to_execution_hints_without_leaking_into_search_intent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _exact_backend(**kwargs):
        captured.update(kwargs)
        return BackendResult(
            name="exact",
            hits=(
                BackendHit(
                    backend="exact",
                    path="src/parser.py",
                    start_line=3,
                    end_line=3,
                    snippet="id = find_id()",
                    raw_score=0.9,
                    tags=("literal_match",),
                ),
            ),
            timing_ms=4,
        )

    monkeypatch.setattr("gloggur.search.router.engine.run_exact_backend", _exact_backend)
    router = SearchRouter(
        repo_root=tmp_path,
        searcher=None,
        metadata_store=None,
        config=SearchRouterConfig(enabled_backends=("exact",)),
    )
    pack = router.search(
        query='rg -i -w -F -g "*.py" id src/',
        intent=SearchIntent(),
        mode="exact",
        include_debug=True,
    )

    assert pack.hits
    captured_intent = captured.get("intent")
    assert isinstance(captured_intent, SearchIntent)
    assert not hasattr(captured_intent, "case_mode")
    assert not hasattr(captured_intent, "fixed_string")
    captured_hints = captured.get("execution_hints")
    assert captured_hints is not None
    assert getattr(captured_hints, "case_mode", None) == "ignore"
    assert getattr(captured_hints, "word_match", False) is True
    assert getattr(captured_hints, "fixed_string", False) is True
    assert "*.py" in getattr(captured_hints, "include_globs", ())


def test_search_router_semantic_query_uses_fusion_and_keeps_lexical_parsing_separate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _exact_backend(**kwargs):
        captured["exact_query"] = kwargs["query"]
        return BackendResult(
            name="exact",
            hits=(
                BackendHit(
                    backend="exact",
                    path=str(tmp_path / "src/a_auth_token.py"),
                    start_line=5,
                    end_line=5,
                    snippet="token = refresh_token()",
                    raw_score=0.99,
                    tags=("literal_match",),
                ),
                BackendHit(
                    backend="exact",
                    path=str(tmp_path / "src/z_cache_token.py"),
                    start_line=8,
                    end_line=8,
                    snippet="token = warm_cache_token()",
                    raw_score=0.98,
                    tags=("literal_match",),
                ),
            ),
            timing_ms=4,
        )

    def _symbol_backend(**kwargs):
        captured["symbol_query"] = kwargs["query"]
        return BackendResult(name="symbol", hits=(), timing_ms=2)

    monkeypatch.setattr("gloggur.search.router.engine.run_exact_backend", _exact_backend)
    monkeypatch.setattr("gloggur.search.router.engine.run_symbol_backend", _symbol_backend)
    monkeypatch.setattr(
        "gloggur.search.router.engine._compute_quality",
        lambda result, **_kwargs: 0.99 if result.name == "exact" else 0.90,
    )

    class FakeSearcher:
        def search(
            self,
            query: str,
            *,
            filters: dict[str, str] | None = None,
            top_k: int = 10,
            context_radius: int = 8,
        ) -> dict[str, object]:
            captured["semantic_query"] = query
            captured["semantic_filters"] = filters
            captured["semantic_top_k"] = top_k
            captured["semantic_context_radius"] = context_radius
            return {
                "results": [
                    {
                        "file": "src/z_cache_token.py",
                        "line": 8,
                        "line_end": 8,
                        "context": "token = warm_cache_token()",
                        "ranking_score": 0.95,
                    }
                ]
            }

    router = SearchRouter(
        repo_root=tmp_path,
        searcher=FakeSearcher(),
        metadata_store=None,
        symbol_store=object(),
        config=SearchRouterConfig(enabled_backends=("exact", "symbol", "semantic")),
    )
    pack = router.search(
        query="rg token src/",
        intent=SearchIntent(semantic_query="cache warmup", max_snippets=5),
        mode="auto",
        include_debug=True,
    )

    assert captured["exact_query"] == "token"
    assert captured["symbol_query"] == "token"
    assert captured["semantic_query"] == "cache warmup"
    assert captured["semantic_filters"] == {"ranking_mode": "balanced", "mode": "semantic"}
    assert pack.summary["strategy"] == "hybrid"
    assert pack.summary["reason"] == "semantic_query_requested"
    assert pack.hits
    assert pack.hits[0].path == "src/z_cache_token.py"
    assert isinstance(pack.debug, dict)
    assert pack.debug["queries"] == {
        "lexical_query": "rg token src/",
        "effective_lexical_query": "token",
        "semantic_query": "cache warmup",
    }
    parsed = pack.debug["parsed_query"]
    assert isinstance(parsed, dict)
    assert parsed["source"] == "grep_compat"
