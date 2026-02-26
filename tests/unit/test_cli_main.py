from __future__ import annotations

import errno
import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

import gloggur.indexer.cache as cache_module
import gloggur.storage.metadata_store as metadata_store_module
import gloggur.storage.vector_store as vector_store_module
from gloggur.cli import main as cli_main
from gloggur.cli.main import (
    _build_inspect_warning_summary,
    _build_resume_contract,
    _inspect_path_class,
    _metadata_reindex_reason,
    _profile_reindex_reason,
    _should_include_inspect_path,
    _stable_fingerprint,
    _tool_version_reindex_reason,
)
from gloggur.indexer.cache import CacheConfig, CacheManager, CacheRecoveryError
from gloggur.io_failures import StorageIOError
from gloggur.models import IndexMetadata


def test_profile_reindex_reason_no_metadata_and_no_profile() -> None:
    """No index metadata/profile should not force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=False,
        cached_profile=None,
        expected_profile="local:model-a",
    )
    assert reason is None


def test_profile_reindex_reason_unknown_profile_with_metadata() -> None:
    """Index metadata without cached profile should force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile=None,
        expected_profile="local:model-a",
    )
    assert reason == "cached embedding profile is unknown"


def test_profile_reindex_reason_profile_changed() -> None:
    """Mismatched cached/expected profile should force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:model-a",
        expected_profile="local:model-b",
    )
    assert reason == "embedding profile changed (cached=local:model-a, current=local:model-b)"


def test_profile_reindex_reason_profile_matches() -> None:
    """Matching profile should not force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:model-a",
        expected_profile="local:model-a",
    )
    assert reason is None


def test_metadata_reindex_reason_missing_metadata() -> None:
    """Missing index metadata should report an explicit rebuild reason."""
    reason = _metadata_reindex_reason(metadata_present=False)
    assert reason is not None
    assert "index metadata missing" in reason


def test_metadata_reindex_reason_present_metadata() -> None:
    """Existing metadata should not add metadata-specific reindex reason."""
    reason = _metadata_reindex_reason(metadata_present=True)
    assert reason is None


def test_tool_version_reindex_reason_is_legacy_safe() -> None:
    """Missing legacy tool-version marker should not force a reindex."""
    reason = _tool_version_reindex_reason(
        last_success_tool_version=None,
        current_tool_version="0.2.0",
    )
    assert reason is None


def test_tool_version_reindex_reason_detects_version_drift() -> None:
    """Mismatched tool-version markers should force deterministic rebuild signaling."""
    reason = _tool_version_reindex_reason(
        last_success_tool_version="0.1.0",
        current_tool_version="0.2.0",
    )
    assert reason == "tool version changed (cached=0.1.0, current=0.2.0)"


def test_stable_fingerprint_is_order_independent() -> None:
    """Fingerprint helper should be deterministic regardless of dict key ordering."""
    payload_a = {
        "b": 2,
        "a": {
            "z": 1,
            "y": ["x", "w"],
        },
    }
    payload_b = {
        "a": {
            "y": ["x", "w"],
            "z": 1,
        },
        "b": 2,
    }
    assert _stable_fingerprint(payload_a) == _stable_fingerprint(payload_b)


def test_inspect_path_class_and_default_filter_scope() -> None:
    """Inspect defaults should include src/other and exclude tests/scripts noise."""
    src_path = "/tmp/repo/src/module.py"
    test_path = "/tmp/repo/tests/test_module.py"
    script_path = "/tmp/repo/scripts/tool.py"
    other_path = "/tmp/repo/examples/demo.py"

    assert _inspect_path_class(src_path) == "src"
    assert _inspect_path_class(test_path) == "tests"
    assert _inspect_path_class(script_path) == "scripts"
    assert _inspect_path_class(other_path) == "other"

    assert _should_include_inspect_path(
        src_path,
        include_tests=False,
        include_scripts=False,
    )
    assert _should_include_inspect_path(
        other_path,
        include_tests=False,
        include_scripts=False,
    )
    assert not _should_include_inspect_path(
        test_path,
        include_tests=False,
        include_scripts=False,
    )
    assert not _should_include_inspect_path(
        script_path,
        include_tests=False,
        include_scripts=False,
    )


def test_inspect_warning_summary_groups_by_type_path_class_and_top_files() -> None:
    """Inspect warning summary should be deterministic and machine-readable."""
    warning_reports = [
        {
            "symbol_id": "/tmp/repo/src/a.py:1:a",
            "warnings": [
                "Missing docstring",
                "Low semantic similarity (score=0.100, threshold=0.200)",
            ],
        },
        {
            "symbol_id": "/tmp/repo/tests/test_a.py:1:test_a",
            "warnings": ["Missing docstring"],
        },
        {
            "symbol_id": "/tmp/repo/scripts/tool.py:1:tool",
            "warnings": ["Missing docstring"],
        },
    ]
    summary = _build_inspect_warning_summary(
        warning_reports,
        symbol_file_paths={},
    )

    assert summary["total_warnings"] == 4
    assert summary["by_warning_type"] == {
        "Low semantic similarity": 1,
        "Missing docstring": 3,
    }
    assert summary["by_path_class"] == {
        "src": 2,
        "tests": 1,
        "scripts": 1,
        "other": 0,
    }
    assert summary["reports_by_path_class"] == {
        "src": 1,
        "tests": 1,
        "scripts": 1,
        "other": 0,
    }
    top_files = summary["top_files"]
    assert isinstance(top_files, list)
    assert len(top_files) == 3
    assert top_files[0]["file"] == "/tmp/repo/src/a.py"
    assert top_files[0]["warnings"] == 2
    assert top_files[0]["path_class"] == "src"


def test_resume_contract_profile_change_is_machine_readable() -> None:
    """Profile drift should produce stable machine-readable resume metadata signals."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-b",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
    )

    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["embedding_profile_changed"]
    assert payload["resume_fingerprint_match"] is False


def test_resume_contract_missing_metadata_has_machine_reason_code() -> None:
    """Missing index metadata should surface explicit stable reason codes for agents."""
    payload = _build_resume_contract(
        metadata=None,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile=None,
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
    )

    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["missing_index_metadata"]
    remediation = payload["resume_remediation"]
    assert isinstance(remediation, dict)
    assert "missing_index_metadata" in remediation
    assert isinstance(remediation["missing_index_metadata"], list)
    assert remediation["missing_index_metadata"]


def test_resume_contract_interrupted_index_has_machine_reason_and_remediation() -> None:
    """Interrupted runs should emit explicit interruption reason code and deterministic guidance."""
    payload = _build_resume_contract(
        metadata=None,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint="last-success-fingerprint",
        last_success_resume_at="2026-02-26T00:00:00+00:00",
    )

    assert payload["resume_decision"] == "reindex_required"
    codes = payload["resume_reason_codes"]
    assert isinstance(codes, list)
    assert "index_interrupted" in codes
    assert "missing_index_metadata" in codes
    remediation = payload["resume_remediation"]
    assert isinstance(remediation, dict)
    assert "index_interrupted" in remediation
    assert isinstance(remediation["index_interrupted"], list)
    assert remediation["index_interrupted"]


def test_resume_contract_reports_last_success_fingerprint_match_signal() -> None:
    """Resume contract should expose whether last-success fingerprint still matches expected."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    expected = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
    )["expected_resume_fingerprint"]
    assert isinstance(expected, str)

    match_payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=expected,
        last_success_resume_at="2026-02-26T00:00:00+00:00",
    )
    mismatch_payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint="stale-fingerprint",
        last_success_resume_at="2026-02-26T00:00:00+00:00",
    )

    assert match_payload["last_success_resume_fingerprint_match"] is True
    assert mismatch_payload["last_success_resume_fingerprint_match"] is False


def test_resume_contract_detects_tool_version_drift_since_last_success() -> None:
    """Tool-version drift should be detectable without mislabeling profile drift."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    old_fingerprint = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        tool_version="0.1.0",
    )["expected_resume_fingerprint"]
    assert isinstance(old_fingerprint, str)
    payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=old_fingerprint,
        last_success_resume_at="2026-02-26T00:00:00+00:00",
        tool_version="0.2.0",
        last_success_tool_version="0.1.0",
    )
    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["tool_version_changed"]
    assert payload["resume_fingerprint_match"] is False
    assert payload["last_success_tool_version_match"] is False
    assert payload["last_success_resume_fingerprint_match"] is False


def test_resume_contract_missing_tool_version_marker_remains_resume_ok() -> None:
    """Legacy caches without tool-version markers should keep resume compatibility logic stable."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        tool_version="0.2.0",
        last_success_tool_version=None,
    )

    assert payload["resume_decision"] == "resume_ok"
    assert payload["resume_reason_codes"] == []
    assert payload["resume_fingerprint_match"] is True
    assert payload["last_success_tool_version_match"] is None


def test_build_status_payload_requires_reindex_on_tool_version_drift() -> None:
    """Status payload should fail closed when last-success tool version drifts."""
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir="/tmp/cache",
    )

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=3, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def list_symbols(self) -> list[object]:
            return [object(), object(), object()]

    payload = cli_main._build_status_payload(config, FakeCache())

    assert payload["needs_reindex"] is True
    assert "tool version changed" in str(payload["reindex_reason"])
    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["tool_version_changed"]
    assert payload["resume_fingerprint_match"] is False


def test_status_supports_tilde_expanded_config_path(tmp_path: Path) -> None:
    """status should expand `~` in --config paths before loading files."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    config_path = fake_home / ".gloggur.yaml"
    config_path.write_text(
        f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )

    result = runner.invoke(
        cli_main.cli,
        ["status", "--json", "--config", "~/.gloggur.yaml"],
        env={"HOME": str(fake_home)},
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["cache_dir"] == str(cache_dir)


def test_create_runtime_applies_embedding_override_without_reloading_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime override should not trigger an unwrapped second config-file load."""

    base_config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            return None

        def get_index_profile(self) -> None:
            return None

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: base_config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _cache_dir: FakeCache())
    monkeypatch.setattr(cli_main, "VectorStore", lambda _cfg: object())

    def _unexpected_reload(*_args: object, **_kwargs: object) -> cli_main.GloggurConfig:
        raise AssertionError("GloggurConfig.load should not be called from _create_runtime")

    monkeypatch.setattr(cli_main.GloggurConfig, "load", _unexpected_reload)

    config, _cache, _vector_store = cli_main._create_runtime(
        config_path=None,
        embedding_provider="openai",
    )

    assert config.embedding_provider == "openai"


def test_create_cache_manager_wraps_cache_recovery_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecoverable cache corruption should map to structured IO failures."""

    class BrokenCacheManager:
        def __init__(self, _config: object) -> None:
            raise CacheRecoveryError("recovery failed")

    monkeypatch.setattr(cli_main, "CacheManager", BrokenCacheManager)
    with pytest.raises(StorageIOError) as exc_info:
        cli_main._create_cache_manager("/tmp/gloggur-cache")

    error = exc_info.value
    assert error.category == "unknown_io_error"
    assert error.operation == "recover corrupted cache database"
    assert error.path.endswith("index.db")
    assert "recovery failed" in error.detail


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo: ["status", "--json"]),
        ("search", lambda repo: ["search", "needle", "--json"]),
        ("inspect", lambda repo: ["inspect", str(repo), "--json"]),
        ("clear-cache", lambda repo: ["clear-cache", "--json"]),
        ("index", lambda repo: ["index", str(repo), "--json"]),
    ],
)
def test_core_commands_surface_cache_recovery_failure_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path], list[str]],
) -> None:
    """Core CLI commands should fail non-zero on unrecoverable cache recovery."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )

    class BrokenCacheManager:
        def __init__(self, _config: object) -> None:
            raise CacheRecoveryError("recovery failed")

    monkeypatch.setattr(cli_main, "CacheManager", BrokenCacheManager)
    result = runner.invoke(cli_main.cli, build_args(repo))

    assert result.exit_code != 0
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "recover corrupted cache database"
    assert str(error["path"]).endswith("index.db")
    assert "recovery failed" in result.output
    assert "Traceback (most recent call last)" not in result.output


def test_status_retries_transient_no_such_table_race_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """status should retry once when a concurrent recovery transiently drops tables mid-read."""
    runner = CliRunner()
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )

    class FlakyCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            raise StorageIOError(
                category="unknown_io_error",
                operation="execute cache database transaction",
                path=str(tmp_path / "cache" / "index.db"),
                probable_cause="An unclassified filesystem or database I/O failure occurred.",
                remediation=["Retry with a known-good writable cache directory."],
                detail="OperationalError: no such table: metadata",
            )

    class HealthyCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            return None

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> None:
            return None

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> None:
            return None

        def list_symbols(self) -> list[object]:
            return []

    cache_instances = [FlakyCache(), HealthyCache()]

    def _next_cache(_cache_dir: str) -> object:
        assert cache_instances
        return cache_instances.pop(0)

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", _next_cache)

    result = runner.invoke(cli_main.cli, ["status", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["cache_dir"] == str(tmp_path / "cache")
    assert payload["needs_reindex"] is True
    assert payload["total_symbols"] == 0
    assert not cache_instances


@pytest.mark.parametrize(
    ("operation", "detail"),
    [
        ("execute cache database transaction", "OperationalError: no such table: metadata"),
        ("configure cache database pragmas", "DatabaseError: file is not a database"),
    ],
)
def test_status_remaps_repeated_transient_no_such_table_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    detail: str,
) -> None:
    """status should map repeated transient race failures to cache recovery operation semantics."""
    runner = CliRunner()
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )

    class FlakyCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            raise StorageIOError(
                category="unknown_io_error",
                operation=operation,
                path=str(tmp_path / "cache" / "index.db"),
                probable_cause="An unclassified filesystem or database I/O failure occurred.",
                remediation=["Retry with a known-good writable cache directory."],
                detail=detail,
            )

    call_count = {"value": 0}

    def _flaky_cache(_cache_dir: str) -> FlakyCache:
        call_count["value"] += 1
        return FlakyCache()

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", _flaky_cache)

    result = runner.invoke(cli_main.cli, ["status", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "recover corrupted cache database"
    assert str(error["detail"]) == detail
    assert call_count["value"] == 2
    assert "Traceback (most recent call last)" not in result.output


def test_status_remaps_transient_pragmas_error_during_cache_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """status should remap repeated transient cache-init pragma failures to recovery semantics."""
    runner = CliRunner()
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )
    call_count = {"value": 0}

    def _raise_pragmas_error(_cache_dir: str) -> object:
        call_count["value"] += 1
        raise StorageIOError(
            category="unknown_io_error",
            operation="configure cache database pragmas",
            path=str(tmp_path / "cache" / "index.db"),
            probable_cause="An unclassified filesystem or database I/O failure occurred.",
            remediation=["Retry with a known-good writable cache directory."],
            detail="DatabaseError: file is not a database",
        )

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", _raise_pragmas_error)

    result = runner.invoke(cli_main.cli, ["status", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "recover corrupted cache database"
    assert str(error["detail"]) == "DatabaseError: file is not a database"
    assert call_count["value"] == 2


@pytest.mark.parametrize("as_json", [False, True])
def test_status_surfaces_unrecoverable_corruption_recovery_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    as_json: bool,
) -> None:
    """status should fail with remediation guidance if corruption recovery cannot proceed."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "index.db"
    db_path.write_bytes(b"not-a-sqlite-db")

    def _always_fail_replace(_src: str, _dst: str) -> None:
        raise OSError("replace denied")

    def _always_fail_remove(_path: str) -> None:
        raise OSError("remove denied")

    monkeypatch.setattr(cache_module.os, "replace", _always_fail_replace)
    monkeypatch.setattr(cache_module.os, "remove", _always_fail_remove)

    args = ["status", "--json"] if as_json else ["status"]
    result = runner.invoke(cli_main.cli, args, env={"GLOGGUR_CACHE_DIR": str(cache_dir)})
    assert result.exit_code != 0
    assert "Cache corruption detected but recovery failed" in result.output
    assert "Fix permissions and remove corrupted cache files manually." in result.output
    assert "Traceback (most recent call last)" not in result.output


def _parse_json_output(output: str) -> dict[str, object]:
    """Parse JSON payload from click output that may include stderr text."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    return payload


@pytest.mark.parametrize(
    ("exception_factory", "category", "detail_substring"),
    [
        (
            lambda: PermissionError(errno.EACCES, "permission denied"),
            "permission_denied",
            "PermissionError",
        ),
        (
            lambda: OSError(errno.EROFS, "read-only filesystem"),
            "read_only_filesystem",
            "read-only filesystem",
        ),
        (
            lambda: sqlite3.OperationalError("database or disk is full"),
            "disk_full_or_quota",
            "database or disk is full",
        ),
        (
            lambda: sqlite3.OperationalError("unable to open database file"),
            "path_not_writable",
            "unable to open database file",
        ),
        (
            lambda: OSError(errno.EIO, "i/o error"),
            "unknown_io_error",
            "i/o error",
        ),
    ],
)
def test_status_json_reports_structured_io_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_factory: Callable[[], Exception],
    category: str,
    detail_substring: str,
) -> None:
    """status --json should emit stable machine-readable payloads for IO failure categories."""
    runner = CliRunner()

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise exception_factory()

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(
        cli_main.cli,
        ["status", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == category
    assert "open cache database connection" in str(error["operation"])
    assert str(error["path"]).endswith("index.db")
    assert detail_substring in str(error["detail"])
    remediation = error["remediation"]
    assert isinstance(remediation, list) and remediation


def test_status_plain_output_includes_actionable_io_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """status (non-json) should include stable actionable guidance on stderr."""
    runner = CliRunner()

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(
        cli_main.cli,
        ["status"],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.exit_code == 1
    assert "IO failure [permission_denied]" in result.output
    assert "Probable cause:" in result.output
    assert "Remediation:" in result.output
    assert "PermissionError" in result.output


def test_index_json_reports_vector_store_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """index --json should surface vector id-map write errors with structured payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf8",
    )

    def _raise_disk_full(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(vector_store_module.json, "dump", _raise_disk_full)
    result = runner.invoke(
        cli_main.cli,
        ["index", str(repo), "--json"],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_LOCAL_FALLBACK": "1",
        },
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["category"] == "disk_full_or_quota"
    assert "write vector id map" in str(error["operation"])
    assert str(error["path"]).endswith("vectors.json")
    assert "no space left on device" in str(error["detail"])


def test_clear_cache_json_reports_vector_artifact_delete_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """clear-cache --json should surface vector artifact delete failures."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_map = cache_dir / "vectors.json"
    vectors_map.write_text("{}", encoding="utf8")
    original_remove = vector_store_module.os.remove

    def _remove_with_permission_denied(path: str | os.PathLike[str]) -> None:
        normalized_path = os.fspath(path)
        if normalized_path.endswith("vectors.json"):
            raise PermissionError(errno.EACCES, "permission denied")
        original_remove(normalized_path)

    monkeypatch.setattr(vector_store_module.os, "remove", _remove_with_permission_denied)
    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "permission_denied"
    assert error["operation"] == "delete vector artifact"
    assert str(error["path"]).endswith("vectors.json")
    assert "permission denied" in str(error["detail"]).lower()


def test_clear_cache_json_ignores_invalid_faiss_index_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear vector files without loading existing index artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_file = cache_dir / "vectors.index"
    index_file.write_text("invalid-faiss-bytes", encoding="utf8")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not index_file.exists()


def test_clear_cache_json_ignores_invalid_vector_id_map_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear malformed vectors.json artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_map = cache_dir / "vectors.json"
    vectors_map.write_text("{this-is-not-json", encoding="utf8")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not vectors_map.exists()


def test_clear_cache_json_ignores_invalid_fallback_vector_matrix_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear malformed fallback matrix artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback_matrix = cache_dir / "vectors.npy"
    fallback_matrix.write_bytes(b"not-a-valid-npy")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not fallback_matrix.exists()


def test_index_json_reports_missing_embedding_provider_configuration(
    tmp_path: Path,
) -> None:
    """index --json should report structured provider error when embedding provider is unset."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / ".gloggur.yaml"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )
    config_path.write_text(
        'embedding_provider: ""\n'
        f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )

    result = runner.invoke(
        cli_main.cli,
        ["index", str(repo), "--json", "--config", str(config_path)],
        env={"GLOGGUR_EMBEDDING_PROVIDER": ""},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "embedding_provider_error"
    assert error["provider"] == "unknown"
    assert error["operation"] == "initialize embedding provider"
    assert "embedding provider is not configured" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


def test_search_json_reports_missing_embedding_provider_configuration(
    tmp_path: Path,
) -> None:
    """search --json should report structured provider error when embedding provider is unset."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / ".gloggur.yaml"
    config_path.write_text(
        'embedding_provider: ""\n'
        f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )
    cache = CacheManager(CacheConfig(str(cache_dir)))
    cache.set_index_metadata(IndexMetadata(version="1", total_symbols=0, indexed_files=0))
    cache.set_index_profile(":unknown")

    result = runner.invoke(
        cli_main.cli,
        ["search", "needle", "--json", "--config", str(config_path)],
        env={"GLOGGUR_EMBEDDING_PROVIDER": ""},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "embedding_provider_error"
    assert error["provider"] == "unknown"
    assert error["operation"] == "initialize embedding provider"
    assert "embedding provider is not configured" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


def test_search_json_requires_reindex_on_tool_version_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should fail closed when cached tool-version marker drifts."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["results"] == []
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["needs_reindex"] is True
    assert "tool version changed" in str(metadata["reindex_reason"])
    assert metadata["resume_decision"] == "reindex_required"
    assert metadata["resume_reason_codes"] == ["tool_version_changed"]


def test_search_json_wraps_metadata_store_connect_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should map metadata-store sqlite failures to structured io_failure."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> None:
            return None

    class FakeVectorStore:
        def search(self, _query_vector: list[float], k: int) -> list[tuple[str, float]]:
            _ = k
            return [("symbol-1", 0.0)]

    class FakeEmbedding:
        provider = "local"

        def embed_text(self, _text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), FakeVectorStore()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(metadata_store_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "path_not_writable"
    assert error["operation"] == "open metadata database connection"
    assert str(error["path"]).endswith("index.db")
    assert "unable to open database file" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo, cfg: ["status", "--json", "--config", str(cfg)]),
        ("search", lambda repo, cfg: ["search", "needle", "--json", "--config", str(cfg)]),
        ("inspect", lambda repo, cfg: ["inspect", str(repo), "--json", "--config", str(cfg)]),
        ("clear-cache", lambda repo, cfg: ["clear-cache", "--json", "--config", str(cfg)]),
        ("index", lambda repo, cfg: ["index", str(repo), "--json", "--config", str(cfg)]),
    ],
)
def test_core_commands_wrap_malformed_config_as_structured_io_failure(
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path, Path], list[str]],
) -> None:
    """Core commands should map malformed config files to stable io_failure payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )
    bad_config = tmp_path / "bad.gloggur.json"
    bad_config.write_text("{not-valid-json", encoding="utf8")

    result = runner.invoke(cli_main.cli, build_args(repo, bad_config))

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "read gloggur config"
    assert str(error["path"]) == str(bad_config)
    assert "malformed" in str(error["probable_cause"]).lower()
    remediation = error.get("remediation")
    assert isinstance(remediation, list)
    assert remediation
    assert "fix config syntax" in str(remediation[0]).lower()
    assert "JSONDecodeError" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo, cfg: ["status", "--json", "--config", str(cfg)]),
        ("search", lambda repo, cfg: ["search", "needle", "--json", "--config", str(cfg)]),
        ("inspect", lambda repo, cfg: ["inspect", str(repo), "--json", "--config", str(cfg)]),
        ("clear-cache", lambda repo, cfg: ["clear-cache", "--json", "--config", str(cfg)]),
        ("index", lambda repo, cfg: ["index", str(repo), "--json", "--config", str(cfg)]),
    ],
)
def test_core_commands_wrap_non_mapping_config_payload_as_structured_io_failure(
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path, Path], list[str]],
) -> None:
    """Core commands should map non-mapping config payloads to stable io_failure payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )
    bad_config = tmp_path / "bad.gloggur.yaml"
    bad_config.write_text("- item1\n- item2\n", encoding="utf8")

    result = runner.invoke(cli_main.cli, build_args(repo, bad_config))

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "read gloggur config"
    assert str(error["path"]) == str(bad_config)
    assert "malformed" in str(error["probable_cause"]).lower()
    remediation = error.get("remediation")
    assert isinstance(remediation, list)
    assert remediation
    assert "top-level mapping" in str(remediation[0]).lower()
    assert "ValueError" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    "command",
    ["status", "search", "inspect", "clear-cache", "index"],
)
def test_core_commands_wrap_sqlite_database_error_as_structured_io_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    """Core commands should surface sqlite DatabaseError as structured IO failures."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf8",
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)

    args_map = {
        "status": ["status", "--json"],
        "search": ["search", "add", "--json"],
        "inspect": ["inspect", str(repo), "--json"],
        "clear-cache": ["clear-cache", "--json"],
        "index": ["index", str(repo), "--json"],
    }
    result = runner.invoke(
        cli_main.cli,
        args_map[command],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_LOCAL_FALLBACK": "1",
        },
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert "cache database" in str(error["operation"])
    assert str(error["path"]).endswith("index.db")
    assert "database disk image is malformed" in str(error["detail"])


@pytest.mark.parametrize(
    ("exception_factory", "category", "detail_substring"),
    [
        (
            lambda: PermissionError(errno.EACCES, "permission denied"),
            "permission_denied",
            "PermissionError",
        ),
        (
            lambda: OSError(errno.EROFS, "read-only filesystem"),
            "read_only_filesystem",
            "read-only filesystem",
        ),
        (
            lambda: sqlite3.OperationalError("database or disk is full"),
            "disk_full_or_quota",
            "database or disk is full",
        ),
        (
            lambda: sqlite3.OperationalError("unable to open database file"),
            "path_not_writable",
            "unable to open database file",
        ),
        (
            lambda: OSError(errno.EIO, "i/o error"),
            "unknown_io_error",
            "i/o error",
        ),
    ],
)
@pytest.mark.parametrize(
    "command",
    ["status", "search", "inspect", "clear-cache", "index"],
)
def test_core_commands_wrap_io_failure_categories_consistently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_factory: Callable[[], Exception],
    category: str,
    detail_substring: str,
    command: str,
) -> None:
    """Core commands should classify IO failure categories consistently."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf8",
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise exception_factory()

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    args_map = {
        "status": ["status", "--json"],
        "search": ["search", "add", "--json"],
        "inspect": ["inspect", str(repo), "--json"],
        "clear-cache": ["clear-cache", "--json"],
        "index": ["index", str(repo), "--json"],
    }
    result = runner.invoke(
        cli_main.cli,
        args_map[command],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_LOCAL_FALLBACK": "1",
        },
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == category
    assert str(error["path"]).endswith("index.db")
    assert detail_substring in str(error["detail"])
    remediation = error.get("remediation")
    assert isinstance(remediation, list) and remediation
