from __future__ import annotations

import errno
import json
import sqlite3
from pathlib import Path
from typing import Callable

import click
import pytest
from click.testing import CliRunner

import gloggur.indexer.cache as cache_module
import gloggur.storage.vector_store as vector_store_module
from gloggur.cli import main as cli_main
from gloggur.cli.main import _metadata_reindex_reason, _profile_reindex_reason
from gloggur.indexer.cache import CacheRecoveryError


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


def test_create_cache_manager_wraps_cache_recovery_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecoverable cache corruption should be raised as a user-facing ClickException."""

    class BrokenCacheManager:
        def __init__(self, _config: object) -> None:
            raise CacheRecoveryError("recovery failed")

    monkeypatch.setattr(cli_main, "CacheManager", BrokenCacheManager)
    with pytest.raises(click.ClickException, match="recovery failed"):
        cli_main._create_cache_manager("/tmp/gloggur-cache")


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
