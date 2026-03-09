from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from gloggur.search import attach_legacy_search_contract
from scripts.verification.fixtures import TestFixtures


def _parse_json_payload(output: str) -> dict[str, object]:
    """Parse the first JSON object from CLI output."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    payload = json.loads(output[start:])
    if isinstance(payload, dict):
        return attach_legacy_search_contract(payload)
    return payload


def _run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Execute gloggur CLI in a fresh Python process."""
    return subprocess.run(
        [sys.executable, "-m", "gloggur.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _write_fallback_marker(cache_dir: str) -> None:
    """Create local-embedding fallback marker in cache directory."""
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _set_cache_meta(cache_dir: str, key: str, value: str) -> None:
    """Update a single cache meta key directly for resilience testing."""
    db_path = Path(cache_dir) / "index.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        conn.commit()


def test_resume_markers_persist_across_fresh_processes() -> None:
    """Index + status/search from fresh processes should preserve last-success resume markers."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        }

        index_run = _run_cli(["index", str(repo), "--json"], env)
        assert index_run.returncode == 0, f"{index_run.stderr}\n{index_run.stdout}"

        status_first = _run_cli(["status", "--json"], env)
        assert status_first.returncode == 0, f"{status_first.stderr}\n{status_first.stdout}"
        first_payload = _parse_json_payload(status_first.stdout)
        assert first_payload["resume_decision"] == "resume_ok"
        assert first_payload["last_success_resume_fingerprint"] == first_payload[
            "expected_resume_fingerprint"
        ]
        assert first_payload["last_success_resume_fingerprint_match"] is True
        assert isinstance(first_payload["tool_version"], str)
        assert first_payload["last_success_tool_version"] == first_payload["tool_version"]
        assert first_payload["last_success_tool_version_match"] is True
        first_fingerprint = first_payload["last_success_resume_fingerprint"]
        assert isinstance(first_fingerprint, str)

        status_second = _run_cli(["status", "--json"], env)
        assert status_second.returncode == 0, f"{status_second.stderr}\n{status_second.stdout}"
        second_payload = _parse_json_payload(status_second.stdout)
        assert second_payload["last_success_resume_fingerprint"] == first_fingerprint
        assert second_payload["last_success_resume_fingerprint_match"] is True
        assert second_payload["last_success_tool_version"] == second_payload["tool_version"]
        assert second_payload["last_success_tool_version_match"] is True
        assert isinstance(second_payload["last_success_resume_at"], str)

        search_run = _run_cli(["search", "add", "--top-k", "1", "--json"], env)
        assert search_run.returncode == 0, f"{search_run.stderr}\n{search_run.stdout}"
        search_payload = _parse_json_payload(search_run.stdout)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["resume_decision"] == "resume_ok"
        assert metadata["last_success_resume_fingerprint"] == first_fingerprint
        assert metadata["last_success_resume_fingerprint_match"] is True
        assert metadata["last_success_tool_version"] == metadata["tool_version"]
        assert metadata["last_success_tool_version_match"] is True


def test_resume_fingerprint_stable_across_unchanged_reindex() -> None:
    """Running index twice on unchanged files should preserve the exact same resume fingerprint."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        }

        # First index run
        index_run_1 = _run_cli(["index", str(repo), "--json"], env)
        assert index_run_1.returncode == 0, f"{index_run_1.stderr}\n{index_run_1.stdout}"
        
        status_1 = _run_cli(["status", "--json"], env)
        assert status_1.returncode == 0, f"{status_1.stderr}\n{status_1.stdout}"
        payload_1 = _parse_json_payload(status_1.stdout)
        fingerprint_1 = payload_1["expected_resume_fingerprint"]

        # Second index run (unchanged files, so should just re-verify)
        index_run_2 = _run_cli(["index", str(repo), "--json"], env)
        assert index_run_2.returncode == 0, f"{index_run_2.stderr}\n{index_run_2.stdout}"
        
        status_2 = _run_cli(["status", "--json"], env)
        assert status_2.returncode == 0, f"{status_2.stderr}\n{status_2.stdout}"
        payload_2 = _parse_json_payload(status_2.stdout)
        fingerprint_2 = payload_2["expected_resume_fingerprint"]

        # Fingerprint should be identical
        assert fingerprint_1 == fingerprint_2


def test_resume_schema_version_mismatch_emits_machine_reason_codes() -> None:
    """Schema-version drift should force deterministic machine-readable reindex decisions."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        }

        index_run = _run_cli(["index", str(repo), "--json"], env)
        assert index_run.returncode == 0, f"{index_run.stderr}\n{index_run.stdout}"

        db_path = Path(cache_dir) / "index.db"
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("UPDATE meta SET value = ? WHERE key = ?", ("1", "schema_version"))
            conn.commit()

        status_run = _run_cli(["status", "--json"], env)
        assert status_run.returncode == 0, f"{status_run.stderr}\n{status_run.stdout}"
        payload = _parse_json_payload(status_run.stdout)
        assert payload["resume_decision"] == "reindex_required"
        assert payload["needs_reindex"] is True
        codes = set(payload["resume_reason_codes"])
        assert "cache_schema_rebuilt" in codes
        assert "missing_index_metadata" in codes
        assert "embedding_profile_changed" not in codes
        assert payload["last_success_resume_fingerprint"] is None
        assert payload["last_success_resume_at"] is None
        assert payload["last_success_resume_fingerprint_match"] is None
        assert isinstance(payload["tool_version"], str)
        assert payload["last_success_tool_version"] is None
        assert payload["last_success_tool_version_match"] is None


def test_resume_profile_alias_treats_hf_snapshot_and_short_name_as_compatible() -> None:
    """Snapshot-path aliases should not trigger false embedding_profile_changed reindex blocks."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
            "GLOGGUR_LOCAL_MODEL": "microsoft/codebert-base",
        }

        index_run = _run_cli(["index", str(repo), "--json"], env)
        assert index_run.returncode == 0, f"{index_run.stderr}\n{index_run.stdout}"

        _set_cache_meta(
            cache_dir,
            "index_profile",
            (
                "test:/Users/example/.cache/huggingface/hub/"
                "models--microsoft--codebert-base/snapshots/abc123"
            ),
        )

        status_run = _run_cli(["status", "--json"], env)
        assert status_run.returncode == 0, f"{status_run.stderr}\n{status_run.stdout}"
        payload = _parse_json_payload(status_run.stdout)
        assert payload["resume_decision"] == "resume_ok"
        assert payload["needs_reindex"] is False
        assert "embedding_profile_changed" not in set(payload["resume_reason_codes"])
        assert payload["expected_index_profile"] == "test:microsoft/codebert-base"
        assert payload["cached_index_profile"] == "test:microsoft/codebert-base"

        search_run = _run_cli(["search", "add", "--json"], env)
        assert search_run.returncode == 0, f"{search_run.stderr}\n{search_run.stdout}"
        search_payload = _parse_json_payload(search_run.stdout)
        results = search_payload["results"]
        assert isinstance(results, list)
        assert results
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["resume_decision"] == "resume_ok"
        assert metadata["needs_reindex"] is False
        assert "embedding_profile_changed" not in set(metadata["resume_reason_codes"])


def test_resume_warns_when_last_success_tool_version_is_tampered() -> None:
    """Tampered last-success tool-version markers should remain reusable with a warning."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        }

        index_run = _run_cli(["index", str(repo), "--json"], env)
        assert index_run.returncode == 0, f"{index_run.stderr}\n{index_run.stdout}"

        _set_cache_meta(cache_dir, "last_success_tool_version", "tampered-version")

        status_run = _run_cli(["status", "--json"], env)
        assert status_run.returncode == 0, f"{status_run.stderr}\n{status_run.stdout}"
        payload = _parse_json_payload(status_run.stdout)
        assert payload["resume_decision"] == "resume_ok"
        assert payload["needs_reindex"] is False
        assert payload["resume_reason_codes"] == []
        assert "tool_version_changed" in set(payload["warning_codes"])
        assert payload["last_success_tool_version_match"] is False
        assert payload["resume_fingerprint_match"] is True
        assert payload["last_success_resume_fingerprint_match"] is True

        search_run = _run_cli(["search", "add", "--json"], env)
        assert search_run.returncode == 0, f"{search_run.stderr}\n{search_run.stdout}"
        search_payload = _parse_json_payload(search_run.stdout)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["needs_reindex"] is False
        assert metadata["resume_reason_codes"] == []
        assert "tool_version_changed" in set(metadata["warning_codes"])
        assert metadata["resume_decision"] == "resume_ok"


def test_resume_recovers_after_successful_noop_reindex_when_only_tool_version_marker_drifts(
) -> None:
    """A successful no-op reindex should repair marker-only tool-version drift."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        }

        first_index_run = _run_cli(["index", str(repo), "--json"], env)
        assert first_index_run.returncode == 0, (
            f"{first_index_run.stderr}\n{first_index_run.stdout}"
        )

        status_before_run = _run_cli(["status", "--json"], env)
        assert status_before_run.returncode == 0, (
            f"{status_before_run.stderr}\n{status_before_run.stdout}"
        )
        status_before_payload = _parse_json_payload(status_before_run.stdout)
        assert status_before_payload["resume_decision"] == "resume_ok"
        assert status_before_payload["resume_reason_codes"] == []
        first_fingerprint = status_before_payload["last_success_resume_fingerprint"]
        first_resume_at = status_before_payload["last_success_resume_at"]
        assert isinstance(first_fingerprint, str)
        assert isinstance(first_resume_at, str)

        _set_cache_meta(cache_dir, "last_success_tool_version", "tampered-version")

        status_drift_run = _run_cli(["status", "--json"], env)
        assert status_drift_run.returncode == 0, (
            f"{status_drift_run.stderr}\n{status_drift_run.stdout}"
        )
        status_drift_payload = _parse_json_payload(status_drift_run.stdout)
        assert status_drift_payload["resume_decision"] == "resume_ok"
        assert status_drift_payload["resume_reason_codes"] == []
        assert "tool_version_changed" in set(status_drift_payload["warning_codes"])
        assert status_drift_payload["last_success_resume_fingerprint"] == first_fingerprint
        assert status_drift_payload["last_success_resume_at"] == first_resume_at
        assert status_drift_payload["last_success_tool_version_match"] is False

        second_index_run = _run_cli(["index", str(repo), "--json"], env)
        assert second_index_run.returncode == 0, (
            f"{second_index_run.stderr}\n{second_index_run.stdout}"
        )
        second_index_payload = _parse_json_payload(second_index_run.stdout)
        assert int(second_index_payload["indexed_files"]) == 0
        assert int(second_index_payload["skipped_files"]) == 1
        assert int(second_index_payload["failed"]) == 0

        status_after_run = _run_cli(["status", "--json"], env)
        assert status_after_run.returncode == 0, (
            f"{status_after_run.stderr}\n{status_after_run.stdout}"
        )
        status_after_payload = _parse_json_payload(status_after_run.stdout)
        assert status_after_payload["resume_decision"] == "resume_ok"
        assert status_after_payload["resume_reason_codes"] == []
        assert status_after_payload["resume_fingerprint_match"] is True
        assert status_after_payload["last_success_resume_fingerprint_match"] is True
        assert status_after_payload["last_success_resume_fingerprint"] == first_fingerprint
        assert status_after_payload["expected_resume_fingerprint"] == first_fingerprint
        assert status_after_payload["last_success_resume_at"] == first_resume_at
        assert status_after_payload["last_success_tool_version"] == status_after_payload[
            "tool_version"
        ]
        assert status_after_payload["last_success_tool_version_match"] is True


def test_resume_accepts_explicit_tool_version_drift_override_input_as_noop() -> None:
    """Explicit override input should not change the default advisory drift behavior."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        }

        index_run = _run_cli(["index", str(repo), "--json"], env)
        assert index_run.returncode == 0, f"{index_run.stderr}\n{index_run.stdout}"

        _set_cache_meta(cache_dir, "last_success_tool_version", "tampered-version")

        status_run = _run_cli(["status", "--json", "--allow-tool-version-drift"], env)
        assert status_run.returncode == 0, f"{status_run.stderr}\n{status_run.stdout}"
        payload = _parse_json_payload(status_run.stdout)
        assert payload["resume_decision"] == "resume_ok"
        assert payload["needs_reindex"] is False
        assert payload["allow_tool_version_drift"] is True
        assert payload["tool_version_drift_detected"] is True
        assert payload["tool_version_drift_override_applied"] is False
        assert payload["resume_reason_codes"] == []
        assert "tool_version_changed" in set(payload["warning_codes"])
        assert payload["reindex_reason"] is None

        search_run = _run_cli(
            ["search", "add", "--json", "--allow-tool-version-drift"],
            env,
        )
        assert search_run.returncode == 0, f"{search_run.stderr}\n{search_run.stdout}"
        search_payload = _parse_json_payload(search_run.stdout)
        results = search_payload["results"]
        assert isinstance(results, list)
        assert results
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["resume_decision"] == "resume_ok"
        assert metadata["needs_reindex"] is False
        assert metadata["allow_tool_version_drift"] is True
        assert metadata["tool_version_drift_detected"] is True
        assert metadata["tool_version_drift_override_applied"] is False
        assert metadata["resume_reason_codes"] == []
        assert "tool_version_changed" in set(metadata["warning_codes"])


def test_resume_accepts_tool_version_drift_override_from_env_var_as_noop() -> None:
    """Env override input should remain accepted without changing default drift behavior."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
            "GLOGGUR_ALLOW_TOOL_VERSION_DRIFT": "true",
        }

        index_run = _run_cli(["index", str(repo), "--json"], env)
        assert index_run.returncode == 0, f"{index_run.stderr}\n{index_run.stdout}"

        _set_cache_meta(cache_dir, "last_success_tool_version", "tampered-version")

        status_run = _run_cli(["status", "--json"], env)
        assert status_run.returncode == 0, f"{status_run.stderr}\n{status_run.stdout}"
        payload = _parse_json_payload(status_run.stdout)
        assert payload["resume_decision"] == "resume_ok"
        assert payload["needs_reindex"] is False
        assert payload["allow_tool_version_drift"] is True
        assert payload["tool_version_drift_detected"] is True
        assert payload["tool_version_drift_override_applied"] is False
        assert payload["resume_reason_codes"] == []
        assert "tool_version_changed" in set(payload["warning_codes"])
        assert payload["reindex_reason"] is None

        search_run = _run_cli(["search", "add", "--json"], env)
        assert search_run.returncode == 0, f"{search_run.stderr}\n{search_run.stdout}"
        search_payload = _parse_json_payload(search_run.stdout)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["resume_decision"] == "resume_ok"
        assert metadata["needs_reindex"] is False
        assert metadata["allow_tool_version_drift"] is True
        assert metadata["tool_version_drift_detected"] is True
        assert metadata["tool_version_drift_override_applied"] is False
        assert metadata["resume_reason_codes"] == []
        assert "tool_version_changed" in set(metadata["warning_codes"])
