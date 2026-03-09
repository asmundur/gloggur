from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from gloggur.cli import main as cli_main
from gloggur.config import GloggurConfig
from gloggur.io_failures import StorageIOError
from gloggur.models import IndexMetadata, Symbol


def _payload(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    assert isinstance(payload, dict)
    return payload


def _ready_health(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "entrypoint": "search_cli_v2",
        "contract_version": "contextpack_v2",
        "needs_reindex": False,
        "reindex_reason": None,
        "resume_contract": {},
        "warning_codes": [],
        "semantic_search_allowed": True,
        "build_state": None,
        "search_integrity": {
            "vector_cache": {"status": "passed", "reason_codes": []},
            "chunk_span": {"status": "passed", "reason_codes": []},
        },
        "expected_index_profile": "test:model",
        "cached_index_profile": "test:model",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("args", "error_code"),
    [
        (["--top-k", "0"], "search_top_k_invalid"),
        (["--max-requery-attempts", "-1"], "search_max_requery_attempts_invalid"),
        (["--max-files", "0"], "search_max_files_invalid"),
        (["--max-snippets", "0"], "search_max_snippets_invalid"),
        (["--time-budget-ms", "0"], "search_time_budget_invalid"),
    ],
)
def test_search_json_rejects_invalid_boundary_options_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    error_code: str,
) -> None:
    runner = CliRunner()
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("runtime should not be created")),
    )

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json", *args])

    assert result.exit_code == 1
    payload = _payload(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == error_code


@pytest.mark.parametrize(
    ("raw_payload", "message_fragment"),
    [
        ([], "must be an object"),
        ({}, "missing 'results'"),
        ({"results": []}, "missing 'metadata'"),
        ({"results": [], "metadata": []}, "'metadata' must be an object"),
    ],
)
def test_validate_search_payload_rejects_schema_drift(
    raw_payload: object,
    message_fragment: str,
) -> None:
    with pytest.raises(ValueError, match=message_fragment):
        cli_main._validate_search_payload(raw_payload)


def test_search_with_bounded_retry_retries_until_confident_result() -> None:
    class FakeSearcher:
        def __init__(self) -> None:
            self.top_k_calls: list[int] = []

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query, filters, context_radius
            self.top_k_calls.append(top_k)
            if len(self.top_k_calls) == 1:
                return {
                    "results": [{"similarity_score": 0.1}],
                    "metadata": {"total_results": 1},
                }
            return {
                "results": [{"similarity_score": 0.95}],
                "metadata": {"total_results": 1},
            }

    searcher = FakeSearcher()
    result, confidence = cli_main._search_with_bounded_retry(
        searcher=searcher,
        query="needle",
        filters={"language": "python"},
        initial_top_k=10,
        context_radius=8,
        confidence_threshold=0.8,
        max_requery_attempts=1,
        disable_bounded_requery=False,
    )

    assert searcher.top_k_calls == [10, 20]
    assert result["results"] == [{"similarity_score": 0.95}]
    assert confidence["retry_performed"] is True
    assert confidence["retry_attempts"] == 1
    assert confidence["initial_top_k"] == 10
    assert confidence["final_top_k"] == 20
    assert confidence["low_confidence"] is False


def test_search_with_bounded_retry_stops_when_top_k_is_already_capped() -> None:
    class FakeSearcher:
        def __init__(self) -> None:
            self.top_k_calls: list[int] = []

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query, filters, context_radius
            self.top_k_calls.append(top_k)
            return {
                "results": [{"similarity_score": 0.1}],
                "metadata": {"total_results": 1},
            }

    searcher = FakeSearcher()
    result, confidence = cli_main._search_with_bounded_retry(
        searcher=searcher,
        query="needle",
        filters={},
        initial_top_k=cli_main.MAX_REQUERY_TOP_K,
        context_radius=8,
        confidence_threshold=0.8,
        max_requery_attempts=2,
        disable_bounded_requery=False,
    )

    assert searcher.top_k_calls == [cli_main.MAX_REQUERY_TOP_K]
    assert result["results"] == [{"similarity_score": 0.1}]
    assert confidence["retry_performed"] is False
    assert confidence["retry_attempts"] == 0
    assert confidence["final_top_k"] == cli_main.MAX_REQUERY_TOP_K
    assert confidence["low_confidence"] is True


@pytest.mark.parametrize(
    ("raw_path", "expected"),
    [
        (None, None),
        ("", None),
        ("src/module.py", "/repo/src/module.py"),
        ("../escape.py", "../escape.py"),
    ],
)
def test_resolve_search_path_filter_for_routing_normalizes_expected_shapes(
    raw_path: str | None,
    expected: str | None,
) -> None:
    repo_root = Path("/repo")
    resolved = cli_main._resolve_search_path_filter_for_routing(
        repo_root=repo_root,
        raw_path=raw_path,
    )
    assert resolved == expected


def test_search_json_needs_reindex_debug_router_emits_debug_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_build_search_health_snapshot",
        lambda *args, **kwargs: _ready_health(
            needs_reindex=True,
            reindex_reason="index build in progress",
            resume_contract={"resume_decision": "reindex_required"},
            warning_codes=["reindex_required"],
        ),
    )

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json", "--debug-router"])

    assert result.exit_code == 1
    payload = _payload(result.output)
    assert payload["error"]["code"] == "search_cache_not_ready"
    debug = payload["debug"]
    assert isinstance(debug, dict)
    assert debug["search_health"]["needs_reindex"] is True
    assert debug["query_mode"] == "auto"
    assert debug["search_mode"] == "semantic"
    assert debug["bounded_retry_enabled"] is True


def test_search_json_debug_router_normalizes_debug_payload_without_eager_semantic_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakePack:
        hits = []
        debug = {}

        def to_dict(self, include_debug: bool = False) -> dict[str, object]:
            _ = include_debug
            return {
                "summary": {"warning_codes": []},
                "hits": [],
                "debug": "malformed-debug",
            }

    class FakeRouter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def search(self, **_kwargs: object) -> FakePack:
            return FakePack()

    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_build_search_health_snapshot",
        lambda *args, **kwargs: _ready_health(),
    )
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _config: object())
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("semantic init should remain lazy in debug mode")
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "_resolve_router_repo_root",
        lambda metadata_store, fallback: tmp_path,
    )
    monkeypatch.setattr(cli_main, "SymbolIndexStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_main, "load_search_router_config", lambda repo_root: object())
    monkeypatch.setattr(cli_main, "SearchRouter", FakeRouter)

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json", "--debug-router"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    debug = payload["debug"]
    assert isinstance(debug, dict)
    assert debug["resume_contract"] == {}
    assert debug["search_health"]["needs_reindex"] is False
    assert "backend_errors" not in debug


def test_search_json_compact_mode_trims_hits_and_summary_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    long_snippet = "x" * 220

    class FakePack:
        debug = None
        hits = []

        def to_dict(self, include_debug: bool = False) -> dict[str, object]:
            _ = include_debug
            return {
                "schema_version": 2,
                "query": "def escape_leading_slashes",
                "summary": {
                    "strategy": "exact",
                    "reason": "quality_threshold_met",
                    "winner": "exact",
                    "hits": 12,
                    "warning_codes": ["identifier_query_high_top_k"],
                    "backend_thresholds": {
                        "exact": {
                            "top_hit_score": 0.97,
                            "quality_score": 0.97,
                            "threshold": 0.7,
                            "threshold_met": True,
                            "evidence_kind": "lexical",
                        }
                    },
                    "query_kind": "declaration",
                    "decisive": True,
                    "next_action": "open_hit_1",
                },
                "hits": [
                    {
                        "path": f"src/file_{index}.py",
                        "span": {"start_line": index + 1, "end_line": index + 1},
                        "snippet": long_snippet,
                        "score": 0.9,
                        "tags": ["literal_match"],
                    }
                    for index in range(12)
                ],
            }

    class FakeRouter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def search(self, **_kwargs: object) -> FakePack:
            return FakePack()

    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_build_search_health_snapshot",
        lambda *args, **kwargs: _ready_health(
            warning_codes=["legacy_search_surface"],
            resume_contract={"resume_decision": "resume_ok"},
        ),
    )
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _config: object())
    monkeypatch.setattr(
        cli_main,
        "_resolve_router_repo_root",
        lambda metadata_store, fallback: tmp_path,
    )
    monkeypatch.setattr(cli_main, "SymbolIndexStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_main, "load_search_router_config", lambda repo_root: object())
    monkeypatch.setattr(cli_main, "SearchRouter", FakeRouter)

    result = runner.invoke(
        cli_main.cli,
        ["search", "def escape_leading_slashes", "--json", "--compact", "--top-k", "12"],
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert sorted(payload.keys()) == ["hits", "query", "schema_version", "summary"]
    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert summary["query_kind"] == "declaration"
    assert summary["decisive"] is True
    assert summary["next_action"] == "open_hit_1"
    assert summary["hits"] == 8
    assert "query_mode" not in summary
    assert "needs_reindex" not in summary
    assert "legacy_top_k" not in summary
    assert "resume_decision" not in summary
    warning_codes = summary["warning_codes"]
    assert isinstance(warning_codes, list)
    assert "identifier_query_high_top_k" in warning_codes
    assert "legacy_search_surface" in warning_codes
    hits = payload["hits"]
    assert isinstance(hits, list)
    assert len(hits) == 8
    assert all(len(str(hit["snippet"])) <= 164 for hit in hits if isinstance(hit, dict))


def test_search_json_stream_emits_hit_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakePack:
        hits = [{"path": "alpha.py", "score": 0.9}, {"path": "beta.py", "score": 0.8}]
        debug = {}

        def to_dict(self, include_debug: bool = False) -> dict[str, object]:
            _ = include_debug
            return {"summary": {"warning_codes": []}, "hits": list(self.hits)}

    class FakeRouter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def search(self, **_kwargs: object) -> FakePack:
            return FakePack()

    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_build_search_health_snapshot",
        lambda *args, **kwargs: _ready_health(),
    )
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _config: object())
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli_main,
        "_resolve_router_repo_root",
        lambda metadata_store, fallback: tmp_path,
    )
    monkeypatch.setattr(cli_main, "SymbolIndexStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_main, "load_search_router_config", lambda repo_root: object())
    monkeypatch.setattr(cli_main, "SearchRouter", FakeRouter)

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json", "--stream"])

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.strip().splitlines() if line.strip()]
    assert lines == FakePack.hits


def test_search_json_stream_requires_hits_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakePack:
        hits = []
        debug = {}

        def to_dict(self, include_debug: bool = False) -> dict[str, object]:
            _ = include_debug
            return {"summary": {"warning_codes": []}, "hits": {"path": "alpha.py"}}

    class FakeRouter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def search(self, **_kwargs: object) -> FakePack:
            return FakePack()

    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_build_search_health_snapshot",
        lambda *args, **kwargs: _ready_health(),
    )
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _config: object())
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli_main,
        "_resolve_router_repo_root",
        lambda metadata_store, fallback: tmp_path,
    )
    monkeypatch.setattr(cli_main, "SymbolIndexStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_main, "load_search_router_config", lambda repo_root: object())
    monkeypatch.setattr(cli_main, "SearchRouter", FakeRouter)

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json", "--stream"])

    assert result.exit_code == 1
    payload = _payload(result.output)
    assert payload["error"]["code"] == "search_result_payload_invalid"


def test_inspect_json_noops_for_unsupported_single_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    target = tmp_path / "notes.txt"
    target.write_text("plain text\n", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakeCache:
        def get_audit_file_metadata(self, _path: str) -> None:
            return None

        def delete_audit_reports_for_file(self, _path: str) -> None:
            pass

        def set_audit_report(self, *args: object, **kwargs: object) -> None:
            pass

        def upsert_audit_file_metadata(self, metadata: object) -> None:
            pass

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: FakeCache())
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda _cfg: None)
    monkeypatch.setattr(cli_main, "audit_docstrings", lambda *args, **kwargs: [])

    result = runner.invoke(cli_main.cli, ["inspect", str(target), "--json"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["files_considered"] == 0
    assert payload["reports_total"] == 0
    assert payload["failed"] == 0


def test_inspect_json_warns_on_skipped_extensions_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    target = tmp_path / "notes.txt"
    target.write_text("plain text\n", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakeCache:
        def get_audit_file_metadata(self, _path: str) -> None:
            return None

        def delete_audit_reports_for_file(self, _path: str) -> None:
            pass

        def set_audit_report(self, *args: object, **kwargs: object) -> None:
            pass

        def upsert_audit_file_metadata(self, metadata: object) -> None:
            pass

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: FakeCache())
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda _cfg: None)
    monkeypatch.setattr(cli_main, "audit_docstrings", lambda *args, **kwargs: [])

    result = runner.invoke(
        cli_main.cli,
        ["inspect", str(target), "--json", "--warn-on-skipped-extensions"],
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["warn_on_skipped_extensions"] is True
    diagnostics = payload["skipped_extension_diagnostics"]
    assert diagnostics["enabled"] is True
    assert diagnostics["warning_code"] == "unsupported_extensions_skipped"
    assert diagnostics["skipped_files"] == 1
    assert diagnostics["by_extension"] == {".txt": 1}
    assert str(target) in diagnostics["sample_paths"]
    assert payload["warning_codes"] == ["unsupported_extensions_skipped"]


def test_inspect_json_skips_excluded_single_file_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    excluded_dir = tmp_path / "excluded"
    excluded_dir.mkdir()
    target = excluded_dir / "sample.py"
    target.write_text("def sample() -> None:\n    pass\n", encoding="utf8")
    config = GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
        excluded_dirs=["excluded"],
    )

    class FakeCache:
        def get_audit_file_metadata(self, _path: str) -> None:
            return None

        def delete_audit_reports_for_file(self, _path: str) -> None:
            pass

        def set_audit_report(self, *args: object, **kwargs: object) -> None:
            pass

        def upsert_audit_file_metadata(self, metadata: object) -> None:
            pass

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: FakeCache())
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda _cfg: None)
    monkeypatch.setattr(cli_main, "audit_docstrings", lambda *args, **kwargs: [])

    result = runner.invoke(cli_main.cli, ["inspect", str(target), "--json"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["files_considered"] == 0
    assert payload["reports_total"] == 0


def test_inspect_json_reports_read_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    target = tmp_path / "sample.py"
    target.write_text("def sample() -> None:\n    pass\n", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    real_open = open

    class FakeCache:
        def get_audit_file_metadata(self, _path: str) -> None:
            return None

        def delete_audit_reports_for_file(self, _path: str) -> None:
            pass

        def set_audit_report(self, *args: object, **kwargs: object) -> None:
            pass

        def upsert_audit_file_metadata(self, metadata: object) -> None:
            pass

    def _fail_target(path: str, *args: object, **kwargs: object):
        if os.path.abspath(path) == os.path.abspath(str(target)):
            raise OSError("cannot read file")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: FakeCache())
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda _cfg: None)
    monkeypatch.setattr(cli_main, "audit_docstrings", lambda *args, **kwargs: [])
    monkeypatch.setattr("builtins.open", _fail_target)

    result = runner.invoke(cli_main.cli, ["inspect", str(target), "--json", "--allow-partial"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["failed"] == 1
    assert payload["failed_reasons"] == {"read_error": 1}
    assert "cannot read file" in str(payload["failed_samples"][0])


def test_inspect_json_reports_parser_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    target = tmp_path / "sample.py"
    target.write_text("def sample() -> None:\n    pass\n", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakeCache:
        def get_audit_file_metadata(self, _path: str) -> None:
            return None

        def delete_audit_reports_for_file(self, _path: str) -> None:
            pass

        def set_audit_report(self, *args: object, **kwargs: object) -> None:
            pass

        def upsert_audit_file_metadata(self, metadata: object) -> None:
            pass

    class FakeParserRegistry:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_parser_for_path(self, _path: str) -> None:
            return None

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: FakeCache())
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda _cfg: None)
    monkeypatch.setattr(cli_main, "ParserRegistry", FakeParserRegistry)
    monkeypatch.setattr(cli_main, "audit_docstrings", lambda *args, **kwargs: [])

    result = runner.invoke(cli_main.cli, ["inspect", str(target), "--json", "--allow-partial"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["failed_reasons"] == {"parser_unavailable": 1}
    assert "No parser registered" in str(payload["failed_samples"][0])


def test_inspect_json_reports_parser_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    target = tmp_path / "sample.py"
    target.write_text("def sample() -> None:\n    pass\n", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakeCache:
        def get_audit_file_metadata(self, _path: str) -> None:
            return None

        def delete_audit_reports_for_file(self, _path: str) -> None:
            pass

        def set_audit_report(self, *args: object, **kwargs: object) -> None:
            pass

        def upsert_audit_file_metadata(self, metadata: object) -> None:
            pass

    class FakeParserRegistry:
        def __init__(self, **_kwargs: object) -> None:
            self.parser = SimpleNamespace(extract_symbols=self._extract_symbols)

        def _extract_symbols(self, file_path: str, source: str) -> list[Symbol]:
            _ = file_path, source
            raise RuntimeError("parse exploded")

        def get_parser_for_path(self, _path: str) -> SimpleNamespace:
            return SimpleNamespace(parser=self.parser)

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: FakeCache())
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda _cfg: None)
    monkeypatch.setattr(cli_main, "ParserRegistry", FakeParserRegistry)
    monkeypatch.setattr(cli_main, "audit_docstrings", lambda *args, **kwargs: [])

    result = runner.invoke(cli_main.cli, ["inspect", str(target), "--json", "--allow-partial"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["failed_reasons"] == {"parse_error": 1}
    assert "parse exploded" in str(payload["failed_samples"][0])


def test_inspect_json_symbol_filter_can_reduce_file_to_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    target = tmp_path / "sample.py"
    target.write_text("def sample() -> None:\n    pass\n", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakeCache:
        def get_audit_file_metadata(self, _path: str) -> None:
            return None

        def delete_audit_reports_for_file(self, _path: str) -> None:
            pass

        def set_audit_report(self, *args: object, **kwargs: object) -> None:
            pass

        def upsert_audit_file_metadata(self, metadata: object) -> None:
            pass

    class FakeParserRegistry:
        def __init__(self, **_kwargs: object) -> None:
            self.parser = SimpleNamespace(extract_symbols=self._extract_symbols)

        def _extract_symbols(self, file_path: str, source: str) -> list[Symbol]:
            _ = source
            return [
                Symbol(
                    id=f"{file_path}:1:sample",
                    name="sample",
                    kind="function",
                    file_path=file_path,
                    start_line=1,
                    end_line=2,
                    signature="def sample() -> None",
                    body_hash="abc",
                )
            ]

        def get_parser_for_path(self, _path: str) -> SimpleNamespace:
            return SimpleNamespace(parser=self.parser)

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: FakeCache())
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda _cfg: None)
    monkeypatch.setattr(cli_main, "ParserRegistry", FakeParserRegistry)
    monkeypatch.setattr(cli_main, "audit_docstrings", lambda *args, **kwargs: [])

    result = runner.invoke(
        cli_main.cli,
        ["inspect", str(target), "--json", "--symbol-id", "missing:symbol"],
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["files_considered"] == 1
    assert payload["indexed"] == 0
    assert payload["reports_total"] == 0


@pytest.mark.parametrize(
    ("command", "method_name", "args"),
    [
        ("neighbors", "neighbors", ["graph", "neighbors", "sym:id", "--json", "--edge-type", "calls", "--direction", "incoming", "--k", "7"]),
        ("incoming", "incoming", ["graph", "incoming", "sym:id", "--json", "--edge-type", "calls", "--k", "5"]),
        ("outgoing", "outgoing", ["graph", "outgoing", "sym:id", "--json", "--edge-type", "calls", "--k", "4"]),
    ],
)
def test_graph_cli_commands_route_to_graph_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    method_name: str,
    args: list[str],
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"))
    calls: dict[str, object] = {}

    class FakeGraphService:
        def __init__(self, metadata_store: object, embedding_provider: object | None = None) -> None:
            calls["metadata_store"] = metadata_store
            calls["embedding_provider"] = embedding_provider

        def neighbors(self, symbol_id: str, *, edge_type: str | None, direction: str, k: int) -> dict[str, object]:
            calls["neighbors"] = (symbol_id, edge_type, direction, k)
            return {"command": command}

        def incoming(self, symbol_id: str, *, edge_type: str | None, k: int) -> dict[str, object]:
            calls["incoming"] = (symbol_id, edge_type, k)
            return {"command": command}

        def outgoing(self, symbol_id: str, *, edge_type: str | None, k: int) -> dict[str, object]:
            calls["outgoing"] = (symbol_id, edge_type, k)
            return {"command": command}

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _config: "metadata-store")
    monkeypatch.setattr(cli_main, "GraphService", FakeGraphService)

    result = runner.invoke(cli_main.cli, args)

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["command"] == command
    assert calls["metadata_store"] == "metadata-store"
    assert method_name in calls


def test_graph_search_uses_provider_override_and_routes_to_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="local")
    captured: dict[str, object] = {}

    class FakeGraphService:
        def __init__(self, metadata_store: object, embedding_provider: object | None = None) -> None:
            captured["metadata_store"] = metadata_store
            captured["embedding_provider"] = embedding_provider

        def search(self, query: str, *, edge_type: str | None, k: int) -> dict[str, object]:
            captured["search"] = (query, edge_type, k)
            return {"results": [{"query": query, "edge_type": edge_type, "k": k}]}

    def _create_provider(cfg: GloggurConfig, require_provider: bool) -> str:
        captured["provider_id"] = cfg.embedding_provider
        captured["require_provider"] = require_provider
        return "embedding-provider"

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _config: "metadata-store")
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", _create_provider)
    monkeypatch.setattr(cli_main, "GraphService", FakeGraphService)

    result = runner.invoke(
        cli_main.cli,
        [
            "graph",
            "search",
            "needle",
            "--json",
            "--edge-type",
            "calls",
            "--k",
            "3",
            "--embedding-provider",
            "test",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["results"] == [{"query": "needle", "edge_type": "calls", "k": 3}]
    assert captured["provider_id"] == "test"
    assert captured["require_provider"] is True
    assert captured["search"] == ("needle", "calls", 3)


def test_support_run_wraps_support_contract_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    def _raise_contract_error(**_kwargs: object) -> tuple[dict[str, object], int]:
        raise cli_main.SupportContractError("bad child command", code="support_command_invalid")

    monkeypatch.setattr(cli_main, "run_support_command_impl", _raise_contract_error)

    result = runner.invoke(cli_main.cli, ["support", "run", "--json", "--", "status", "--json"])

    assert result.exit_code == 1
    payload = _payload(result.output)
    assert payload["error"]["code"] == "support_command_invalid"
    assert payload["failure_codes"] == ["support_command_invalid"]


def test_status_json_rethrows_second_non_transient_retry_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"))
    transient_error = StorageIOError(
        category="unknown_io_error",
        operation="open cache database",
        path=str(tmp_path / "cache" / "index.db"),
        probable_cause="transient race",
        remediation=["retry"],
        detail="sqlite3.OperationalError: no such table",
    )
    retry_error = StorageIOError(
        category="permission_denied",
        operation="open cache database",
        path=str(tmp_path / "cache" / "index.db"),
        probable_cause="permission denied",
        remediation=["fix permissions"],
        detail="PermissionError: denied",
    )
    calls = {"count": 0}

    def _create_status_payload(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise transient_error
        raise retry_error

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_status_payload", _create_status_payload)
    monkeypatch.setattr(
        cli_main,
        "_is_transient_status_race_error",
        lambda error: error is transient_error,
    )

    result = runner.invoke(cli_main.cli, ["status", "--json"])

    assert result.exit_code == 1
    payload = _payload(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["category"] == "permission_denied"
    assert calls["count"] == 2


def test_guidance_exits_when_embedding_provider_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: None,
    )

    result = runner.invoke(cli_main.cli, ["guidance", "sample.py:1:needle", "--json"])

    assert result.exit_code == 1
    assert result.output == ""


def test_guidance_prints_error_payload_to_stderr_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakeGuidance:
        def __init__(self, searcher: object) -> None:
            self.searcher = searcher

        def generate_agent_context(self, symbol_id: str) -> dict[str, object]:
            assert symbol_id == "missing:symbol"
            return {"error": "missing symbol"}

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _config: object())
    monkeypatch.setattr("gloggur.search.guidance.AgentGuidance", FakeGuidance)

    result = runner.invoke(cli_main.cli, ["guidance", "missing:symbol", "--json"])

    assert result.exit_code == 1
    assert "missing symbol" in result.output


def test_coverage_ingest_plain_output_reports_summary_and_updates_symbols(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    coverage_file = tmp_path / "coverage.json"
    coverage_file.write_text("{}", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"))
    updated_symbols: list[object] = []

    class FakeIngester:
        def __init__(self, metadata: object) -> None:
            self.metadata = metadata

        def ingest_json(self, file_path: str) -> dict[str, object]:
            assert file_path == str(coverage_file)
            return {
                "tests_processed": 2,
                "files_affected": 1,
                "symbols_to_update": [object(), object()],
            }

    class FakeCache:
        def __init__(self, config: object) -> None:
            self.config = config

        def upsert_symbols(self, symbols: list[object]) -> None:
            updated_symbols.extend(symbols)

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _config: object())
    monkeypatch.setattr(cli_main, "CoverageIngester", FakeIngester)
    monkeypatch.setattr(cli_main, "CacheManager", FakeCache)
    monkeypatch.setattr(cli_main, "cache_write_lock", lambda _path: nullcontext())

    result = runner.invoke(cli_main.cli, ["coverage", "ingest", str(coverage_file)])

    assert result.exit_code == 0, result.output
    assert "Coverage ingestion complete:" in result.output
    assert "Tests mapped:    2" in result.output
    assert "Files affected:  1" in result.output
    assert "Symbols updated: 2" in result.output
    assert len(updated_symbols) == 2


def test_run_coverage_import_wraps_write_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"))
    source_file = tmp_path / "coverage.json"
    source_file.write_text("{}", encoding="utf8")
    real_open = open

    class FakeImporter:
        def import_contexts(self, file_path: str) -> dict[str, object]:
            assert file_path == str(source_file)
            return {"test_alpha": {"alpha.py": [1]}}

    def _fail_output(path: str, *args: object, **kwargs: object):
        if os.path.abspath(path) == os.path.abspath(str(tmp_path / "out.json")):
            raise PermissionError("permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "create_coverage_importer", lambda *_args, **_kwargs: FakeImporter())
    monkeypatch.setattr("builtins.open", _fail_output)

    with pytest.raises(StorageIOError) as exc_info:
        cli_main._run_coverage_import(
            file=str(source_file),
            output=str(tmp_path / "out.json"),
            importer_id="json",
            as_json=False,
        )

    assert exc_info.value.operation == "write JSON coverage file"


def test_coverage_import_plain_output_reports_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    source_file = tmp_path / "coverage.json"
    source_file.write_text("{}", encoding="utf8")
    output_file = tmp_path / "gloggur-coverage.json"
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"))

    class FakeImporter:
        def import_contexts(self, file_path: str) -> dict[str, object]:
            assert file_path == str(source_file)
            return {
                "test_alpha": {"alpha.py": [1, 2]},
                "test_beta": {"beta.py": [3]},
            }

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "create_coverage_importer", lambda *_args, **_kwargs: FakeImporter())

    result = runner.invoke(
        cli_main.cli,
        ["coverage", "import", str(source_file), "--importer", "json", "--output", str(output_file)],
    )

    assert result.exit_code == 0, result.output
    assert "Coverage import complete:" in result.output
    assert "Tests extracted: 2" in result.output
    assert str(output_file) in result.output
    written = json.loads(output_file.read_text(encoding="utf8"))
    assert sorted(written) == ["test_alpha", "test_beta"]
