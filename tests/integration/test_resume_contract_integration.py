from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from scripts.verification.fixtures import TestFixtures


def _parse_json_payload(output: str) -> dict[str, object]:
    """Parse the first JSON object from CLI output."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    return json.loads(output[start:])


def _run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Execute gloggur CLI in a fresh Python process."""
    return subprocess.run(
        [sys.executable, "-m", "gloggur.cli.main", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
            "GLOGGUR_LOCAL_FALLBACK": "1",
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
            "GLOGGUR_LOCAL_FALLBACK": "1",
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
            "GLOGGUR_LOCAL_FALLBACK": "1",
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


def test_resume_requires_reindex_when_last_success_tool_version_is_tampered() -> None:
    """Tampered last-success tool-version markers should fail closed with a machine-readable code."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_FALLBACK": "1",
        }

        index_run = _run_cli(["index", str(repo), "--json"], env)
        assert index_run.returncode == 0, f"{index_run.stderr}\n{index_run.stdout}"

        _set_cache_meta(cache_dir, "last_success_tool_version", "tampered-version")

        status_run = _run_cli(["status", "--json"], env)
        assert status_run.returncode == 0, f"{status_run.stderr}\n{status_run.stdout}"
        payload = _parse_json_payload(status_run.stdout)
        assert payload["resume_decision"] == "reindex_required"
        assert payload["needs_reindex"] is True
        assert "tool_version_changed" in set(payload["resume_reason_codes"])
        assert payload["last_success_tool_version_match"] is False
        assert payload["resume_fingerprint_match"] is False

        search_run = _run_cli(["search", "add", "--json"], env)
        assert search_run.returncode == 0, f"{search_run.stderr}\n{search_run.stdout}"
        search_payload = _parse_json_payload(search_run.stdout)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["needs_reindex"] is True
        assert "tool_version_changed" in set(metadata["resume_reason_codes"])
        assert metadata["resume_decision"] == "reindex_required"


def test_resume_allows_explicit_tool_version_drift_override() -> None:
    """Explicit override should allow resume/search while keeping drift diagnostics machine-readable."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_FALLBACK": "1",
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
        assert payload["tool_version_drift_override_applied"] is True
        assert payload["resume_reason_codes"] == ["tool_version_changed_override"]
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
        assert metadata["tool_version_drift_override_applied"] is True
        assert metadata["resume_reason_codes"] == ["tool_version_changed_override"]


def test_resume_allows_tool_version_drift_override_from_env_var() -> None:
    """Env override should enable tool-version drift resume behavior without CLI flag."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_FALLBACK": "1",
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
        assert payload["tool_version_drift_override_applied"] is True
        assert payload["resume_reason_codes"] == ["tool_version_changed_override"]
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
        assert metadata["tool_version_drift_override_applied"] is True
        assert metadata["resume_reason_codes"] == ["tool_version_changed_override"]
