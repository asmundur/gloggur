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


def _fake_directory_result(
    *,
    failed: int = 0,
    failed_reasons: dict[str, int] | None = None,
    failed_samples: list[str] | None = None,
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
    return SimpleNamespace(outcome=outcome, timing=timing, prepared=prepared)


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

        def index_repository(self, _path: str) -> SimpleNamespace:
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
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object())
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
    monkeypatch.setattr(cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None)

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json"])

    assert result.exit_code == 1
    payload = _payload(result.output)
    assert payload["failed"] == 1
    assert payload["failed_reasons"] == {"symbol_error": 1}
    assert payload["failed_samples"] == ["repo-1", "repo-2", "repo-3", "repo-4", "repo-5"]
    assert payload["failure_codes"] == ["symbol_error"]
    assert active_cache.publish_calls == []


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

        def index_repository(self, _path: str) -> SimpleNamespace:
            return _fake_directory_result(
                failed=1,
                failed_reasons={"repo_error": 1},
                failed_samples=["repo-error-sample"],
            )

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object())
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
    monkeypatch.setattr(cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None)

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json", "--allow-partial"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["failed"] == 2
    assert payload["failed_reasons"] == {"repo_error": 1}
    assert payload["failed_samples"] == ["repo-error-sample"]


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda root: root / "excluded" / "sample.py",
        lambda root: root / "notes.txt",
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

        def index_file_with_details(self, _path: str) -> SimpleNamespace:
            raise AssertionError("single-file indexing should not run for skipped files")

        def prune_missing_file_entries(self) -> dict[str, object]:
            raise AssertionError("cleanup should not run for skipped files")

        def validate_vector_metadata_consistency(self) -> dict[str, object]:
            raise AssertionError("consistency should not run for skipped files")

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object())
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
    monkeypatch.setattr(cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None)

    result = runner.invoke(cli_main.cli, ["index", str(target), "--json"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.output)
    assert payload["files_considered"] == 0
    assert payload["failed"] == 0
    assert payload["indexed"] == 0
    assert symbol_calls[0]["file_paths"] is None


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

        def index_file_with_details(self, _path: str) -> SimpleNamespace:
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
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object())
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
    monkeypatch.setattr(cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None)

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

        def index_repository(self, _path: str) -> SimpleNamespace:
            return _fake_directory_result()

    @contextmanager
    def _fake_guard(on_interrupt):
        on_interrupt("SIGINT")
        yield

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_embedding_provider_for_command", lambda *_args, **_kwargs: object())
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
    monkeypatch.setattr(cli_main, "_persist_last_success_resume_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main, "_index_signal_guard", _fake_guard)
    monkeypatch.setattr(
        cli_main,
        "_write_cache_build_state",
        lambda _cache, **kwargs: build_state_calls.append(kwargs),
    )
    monkeypatch.setattr(cli_main, "_terminate_index_children", lambda: interrupted.append("terminated") or {})

    result = runner.invoke(cli_main.cli, ["index", str(repo), "--json"])

    assert result.exit_code == 0, result.output
    assert interrupted == ["terminated"]
    assert any(
        call["state"] == "interrupted" and call["stage"] == "scan_source"
        for call in build_state_calls
    )
