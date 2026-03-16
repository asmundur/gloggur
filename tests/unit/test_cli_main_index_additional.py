from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from gloggur.cli import main as cli_main
from gloggur.config import GloggurConfig


def _payload(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    assert isinstance(payload, dict)
    return payload


class _FakeActiveCache:
    def __init__(self, stage_dir: Path) -> None:
        self.stage_dir = stage_dir
        self.publish_calls: list[str] = []
        self.clear_build_state_calls = 0
        self.build_state_payloads: list[dict[str, object]] = []

    def cleanup_staged_builds(self) -> None:
        pass

    def prepare_staged_build(self, build_id: str) -> str:
        return str(self.stage_dir / build_id)

    def publish_staged_build(self, build_id: str) -> None:
        self.publish_calls.append(build_id)

    def count_symbols(self) -> int:
        return 7

    def count_files(self) -> int:
        return 3

    def clear_build_state(self) -> None:
        self.clear_build_state_calls += 1

    def write_build_state(self, payload: dict[str, object]) -> dict[str, object]:
        self.build_state_payloads.append(dict(payload))
        return payload


class _FakeStageCache:
    def __init__(self) -> None:
        self.search_integrity_payload: dict[str, object] | None = None
        self.metadata_cleared = 0
        self.index_metadata = None
        self.index_profile = None
        self.clear_build_file_checkpoints_calls = 0

    def delete_index_metadata(self) -> None:
        self.metadata_cleared += 1

    def set_search_integrity(self, payload: dict[str, object]) -> None:
        self.search_integrity_payload = payload

    def count_symbols(self) -> int:
        return 5

    def count_files(self) -> int:
        return 2

    def set_index_metadata(self, metadata: object) -> None:
        self.index_metadata = metadata

    def set_index_profile(self, profile: str) -> None:
        self.index_profile = profile

    def clear_build_file_checkpoints(self) -> None:
        self.clear_build_file_checkpoints_calls += 1


def _fake_directory_result(
    *,
    failed: int = 0,
    failed_reasons: dict[str, int] | None = None,
    failed_samples: list[str] | None = None,
    extract_worker: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        failed=failed,
        failed_reasons=failed_reasons or {},
        failed_samples=failed_samples or [],
        files_considered=1,
        indexed_symbols=2,
        files_changed=1,
        files_removed=0,
        symbols_removed=0,
        parsed_files=[],
        source_files=["sample.py"],
        duration_ms=12,
        phase_timings_ms={
            "scan_source": 1,
            "extract_symbols": 2,
            "embed_chunks": 3,
            "persist_cache": 1,
            "validate_integrity": 2,
            "cleanup": 1,
            "consistency_checks": 2,
        },
        file_timings=[],
        extract_worker=extract_worker,
        as_payload=lambda: {
            "files_considered": 1,
            "indexed": 1,
            "unchanged": 0,
            "indexed_files": 1,
            "indexed_symbols": 2,
            "failed": failed,
            "failed_reasons": failed_reasons or {},
            "failed_samples": failed_samples or [],
        },
    )


def _fake_execution(
    *,
    status: str = "indexed",
    reason: str | None = None,
    detail: str | None = None,
    extract_worker: object | None = None,
) -> SimpleNamespace:
    timing = SimpleNamespace(
        parse_ms=2,
        edge_ms=1,
        embed_ms=3,
        persist_ms=4,
        total_ms=10,
        path="sample.py",
        as_payload=lambda: {"path": "sample.py", "total_ms": 10},
    )
    outcome = SimpleNamespace(
        status=status,
        symbols_indexed=2,
        symbols_added=1,
        symbols_updated=0,
        symbols_removed=0,
        reason=reason,
        detail=detail,
    )
    prepared = SimpleNamespace(snapshot=object())
    return SimpleNamespace(
        outcome=outcome,
        timing=timing,
        prepared=prepared,
        extract_worker=extract_worker,
    )


@contextmanager
def _no_lock(_path: str):
    yield


def test_index_directory_merges_symbol_failures_and_caps_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_repository(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            return _fake_directory_result(
                failed_samples=[
                    "repo-1",
                    "repo-2",
                    "repo-3",
                    "repo-4",
                    "repo-5",
                ],
            )

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: {
            "failed": 1,
            "failed_reasons": {"ignored": 0, "symbol_error": 1},
            "failed_samples": ["symbol-extra"],
            "duration_ms": 5,
        },
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json"])

    assert result.exit_code == 1
    payload = _payload(result.output)
    assert payload["failed"] == 1
    assert payload["failed_reasons"] == {"symbol_error": 1}
    assert payload["failed_samples"] == ["repo-1", "repo-2", "repo-3", "repo-4", "repo-5"]
    assert payload["failure_codes"] == ["symbol_error"]
    assert active_cache.publish_calls == []


def test_index_directory_clears_stage_checkpoints_before_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_repository(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            return _fake_directory_result()

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: {"failed": 0, "duration_ms": 0},
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json"])

    assert result.exit_code == 0, result.output
    assert stage_cache.clear_build_file_checkpoints_calls == 1
    assert len(active_cache.publish_calls) == 1


def test_index_directory_ignores_non_mapping_failure_reasons_and_non_list_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_repository(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            return _fake_directory_result(
                failed=1,
                failed_reasons={"repo_error": 1},
                failed_samples=["repo-error-sample"],
            )

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: {
            "failed": 1,
            "failed_reasons": "malformed",
            "failed_samples": "malformed",
            "duration_ms": 5,
        },
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json", "--allow-partial"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["failed"] == 2
    assert payload["failed_reasons"] == {"repo_error": 1}
    assert payload["failed_samples"] == ["repo-error-sample"]


def test_index_directory_surfaces_extract_worker_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()
    extract_worker = SimpleNamespace(
        prepare_file_mode="worker",
        build_edges_mode="inline",
        catalog_symbol_count=7000,
        reason_code="candidate_catalog_too_large",
    )

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_repository(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            return _fake_directory_result(extract_worker=extract_worker)

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: {"failed": 0, "duration_ms": 0},
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["extract_worker"] == {
        "prepare_file_mode": "worker",
        "build_edges_mode": "inline",
        "catalog_symbol_count": 7000,
        "reason_code": "candidate_catalog_too_large",
    }


def test_extract_worker_payload_passthrough_dict() -> None:
    extract_worker = {
        "prepare_file_mode": "worker",
        "build_edges_mode": "inline",
        "catalog_symbol_count": 3,
    }

    assert cli_main._extract_worker_payload(extract_worker) == extract_worker


def test_extract_worker_payload_uses_as_payload_dict() -> None:
    extract_worker = SimpleNamespace(
        as_payload=lambda: {
            "prepare_file_mode": "worker",
            "build_edges_mode": "worker",
            "catalog_symbol_count": 5,
        }
    )

    assert cli_main._extract_worker_payload(extract_worker) == {
        "prepare_file_mode": "worker",
        "build_edges_mode": "worker",
        "catalog_symbol_count": 5,
    }


def test_extract_worker_payload_rejects_non_dict_as_payload() -> None:
    extract_worker = SimpleNamespace(as_payload=lambda: ["not", "a", "dict"])

    assert cli_main._extract_worker_payload(extract_worker) is None


@pytest.mark.parametrize(
    "extract_worker",
    [
        SimpleNamespace(prepare_file_mode=None, build_edges_mode="inline"),
        SimpleNamespace(prepare_file_mode="worker", build_edges_mode=None),
    ],
)
def test_extract_worker_payload_requires_string_modes(extract_worker: object) -> None:
    assert cli_main._extract_worker_payload(extract_worker) is None


def test_extract_worker_payload_omits_empty_reason_code() -> None:
    extract_worker = SimpleNamespace(
        prepare_file_mode="worker",
        build_edges_mode="inline",
        catalog_symbol_count=None,
        reason_code="",
    )

    assert cli_main._extract_worker_payload(extract_worker) == {
        "prepare_file_mode": "worker",
        "build_edges_mode": "inline",
        "catalog_symbol_count": 0,
    }


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--embed-graph-edges"], True),
        (["--no-embed-graph-edges"], False),
    ],
)
def test_index_cli_overrides_embed_graph_edges_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    args: list[str],
    expected: bool,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
        embed_graph_edges=not expected,
    )
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()
    seen_configs: list[GloggurConfig] = []

    class FakeIndexer:
        def __init__(self, **kwargs: object) -> None:
            config_obj = kwargs.get("config")
            assert isinstance(config_obj, GloggurConfig)
            seen_configs.append(config_obj)
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_repository(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            return _fake_directory_result()

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: {"failed": 0, "duration_ms": 0},
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json", *args])

    assert result.exit_code == 0, result.output
    assert seen_configs
    assert seen_configs[0].embed_graph_edges is expected


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda root: root / "excluded" / "sample.py",
        lambda root: root / "notes.txt",
        lambda root: root / "vendor.min.js",
    ],
)
def test_index_single_file_noops_for_excluded_or_unsupported_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path_factory,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    target = path_factory(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def sample() -> None:\n    pass\n", encoding="utf8")
    config = GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
        excluded_dirs=["excluded"],
    )
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()
    symbol_calls: list[dict[str, object]] = []

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_file_with_details(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            raise AssertionError("single-file indexing should not run for skipped files")

        def prune_missing_file_entries(self) -> dict[str, object]:
            raise AssertionError("cleanup should not run for skipped files")

        def validate_vector_metadata_consistency(self) -> dict[str, object]:
            raise AssertionError("consistency should not run for skipped files")

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **kwargs: symbol_calls.append(kwargs) or {"failed": 0, "duration_ms": 0},
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(target), "--json"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["files_considered"] == 0
    assert payload["failed"] == 0
    assert payload["indexed"] == 0
    assert symbol_calls == []


def test_index_single_file_warns_on_skipped_extensions_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "notes.txt"
    target.write_text("plain text\n", encoding="utf8")
    config = GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
    )
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()
    symbol_calls: list[dict[str, object]] = []

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_file_with_details(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            raise AssertionError("single-file indexing should not run for skipped files")

        def prune_missing_file_entries(self) -> dict[str, object]:
            raise AssertionError("cleanup should not run for skipped files")

        def validate_vector_metadata_consistency(self) -> dict[str, object]:
            raise AssertionError("consistency should not run for skipped files")

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **kwargs: symbol_calls.append(kwargs) or {"failed": 0, "duration_ms": 0},
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(
        cli_main.cli,
        ["index", str(target), "--json", "--warn-on-skipped-extensions"],
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
    assert symbol_calls == []


def test_index_single_file_debug_timings_adds_slow_file_and_missing_integrity_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    target = tmp_path / "sample.py"
    target.write_text("def sample() -> None:\n    pass\n", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()

    class FakeVectorStore:
        def __init__(self) -> None:
            self.saved = 0

        def save(self) -> None:
            self.saved += 1

    vector_store = FakeVectorStore()

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_file_with_details(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            return _fake_execution(status="indexed")

        def prune_missing_file_entries(self) -> dict[str, object]:
            return {
                "files_removed": 0,
                "symbols_removed": 0,
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
            }

        def validate_vector_metadata_consistency(self) -> dict[str, object]:
            return {"failed": 0, "failed_reasons": {}, "failed_samples": []}

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, vector_store),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: {"failed": 0, "duration_ms": 6},
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(target), "--json", "--debug-timings"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["slow_files"] == [{"path": "sample.py", "total_ms": 10}]
    search_integrity = stage_cache.search_integrity_payload
    assert isinstance(search_integrity, dict)
    vector_cache = search_integrity["vector_cache"]
    assert isinstance(vector_cache, dict)
    assert vector_cache["status"] == "missing"
    assert vector_cache["reason_codes"] == ["vector_integrity_missing"]
    assert vector_store.saved == 1


def test_index_single_file_surfaces_extract_worker_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    target = tmp_path / "sample.py"
    target.write_text("def sample() -> None:\n    pass\n", encoding="utf8")
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()

    class FakeVectorStore:
        def save(self) -> None:
            pass

    vector_store = FakeVectorStore()
    extract_worker = SimpleNamespace(
        prepare_file_mode="worker",
        build_edges_mode="worker",
        catalog_symbol_count=4,
        reason_code="worker_forced",
    )

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_file_with_details(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            return _fake_execution(status="indexed", extract_worker=extract_worker)

        def prune_missing_file_entries(self) -> dict[str, object]:
            return {
                "files_removed": 0,
                "symbols_removed": 0,
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
            }

        def validate_vector_metadata_consistency(self) -> dict[str, object]:
            return {"failed": 0, "failed_reasons": {}, "failed_samples": []}

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, vector_store),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: {"failed": 0, "duration_ms": 0},
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(target), "--json"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["extract_worker"] == {
        "prepare_file_mode": "worker",
        "build_edges_mode": "worker",
        "catalog_symbol_count": 4,
        "reason_code": "worker_forced",
    }


def test_index_interrupt_handler_marks_build_interrupted_and_terminates_children(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()
    build_state_calls: list[dict[str, object]] = []
    interrupted: list[str] = []

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None

        def index_repository(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            return _fake_directory_result()

    @contextmanager
    def _fake_guard(on_interrupt):
        on_interrupt("SIGINT")
        yield

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: {"failed": 0, "duration_ms": 0},
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(cli_main, "_index_signal_guard", _fake_guard)
    monkeypatch.setattr(
        cli_main,
        "_write_cache_build_state",
        lambda _cache, **kwargs: build_state_calls.append(kwargs),
    )
    monkeypatch.setattr(
        cli_main, "_terminate_index_children", lambda: interrupted.append("terminated") or {}
    )

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json"])

    assert result.exit_code == 0, result.output
    assert interrupted == ["terminated"]
    assert any(
        call["state"] == "interrupted" and call["stage"] == "scan_source"
        for call in build_state_calls
    )


def test_index_directory_extract_timeout_preserves_progress_and_skips_symbol_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")
    active_cache = _FakeActiveCache(tmp_path / "staged")
    stage_cache = _FakeStageCache()
    progress = {
        "current_file": "hang.py",
        "subphase": "prepare_file",
        "files_done": 1,
        "files_total": 3,
        "started_at": "2026-03-14T10:00:00+00:00",
        "updated_at": "2026-03-14T10:00:05+00:00",
    }

    class FakeIndexer:
        def __init__(self, **_kwargs: object) -> None:
            self._stage_callback = None
            self._scan_callback = None
            self._progress_callback = None
            self._extract_progress_callback = None
            self._allow_partial_failures = False

        def index_repository(self, _path: str, **_kwargs: object) -> SimpleNamespace:
            if self._stage_callback is not None:
                self._stage_callback("extract_symbols")
            assert self._extract_progress_callback is not None
            self._extract_progress_callback(progress)
            return SimpleNamespace(
                failed=1,
                failed_reasons={"extract_symbols_timeout": 1},
                failed_samples=["hang.py: prepare_file timed out"],
                files_considered=2,
                indexed_symbols=0,
                files_changed=0,
                files_removed=0,
                symbols_removed=0,
                parsed_files=[],
                source_files=["hang.py", "ok.py"],
                duration_ms=12,
                phase_timings_ms={
                    "scan_source": 1,
                    "extract_symbols": 8,
                    "embed_chunks": 0,
                    "persist_cache": 0,
                    "validate_integrity": 0,
                    "cleanup": 0,
                    "consistency_checks": 0,
                },
                file_timings=[SimpleNamespace(path="hang.py", reason="extract_symbols_timeout")],
                as_payload=lambda: {
                    "files_considered": 2,
                    "indexed": 0,
                    "unchanged": 0,
                    "indexed_files": 0,
                    "indexed_symbols": 0,
                    "failed": 1,
                    "failed_reasons": {"extract_symbols_timeout": 1},
                    "failed_samples": ["hang.py: prepare_file timed out"],
                },
            )

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
    monkeypatch.setattr(cli_main, "cache_write_lock", _no_lock)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _dir: active_cache)
    monkeypatch.setattr(
        cli_main,
        "_initialize_runtime",
        lambda stage_config, **_kwargs: (stage_config, stage_cache, object()),
    )
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(cli_main, "Indexer", FakeIndexer)
    monkeypatch.setattr(
        cli_main,
        "_run_symbol_index",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("symbol index should not run")),
    )
    monkeypatch.setattr(
        cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None
    )

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json"])

    assert result.exit_code == 1, result.output
    payload = _payload(result.output)
    assert payload["failure_codes"] == ["extract_symbols_timeout"]
    stage_status = {entry["name"]: entry["status"] for entry in payload["stages"]}
    assert stage_status["update_symbol_index"] == "not_run"
    assert stage_status["commit_metadata"] == "not_run"
    interrupted_payloads = [
        item for item in active_cache.build_state_payloads if item["state"] == "interrupted"
    ]
    assert interrupted_payloads
    assert interrupted_payloads[-1]["progress"] == progress
