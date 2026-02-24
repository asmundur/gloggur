from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from gloggur.cli.main import cli
from scripts.verification.fixtures import TestFixtures

pytest.importorskip("faiss")


def _write_fallback_marker(cache_dir: str) -> None:
    """Create the local embedding fallback marker file."""
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _parse_json_output(output: str) -> dict[str, object]:
    """Parse JSON output from CLI command output."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    return payload


def test_cli_index_search_status_and_clear_cache() -> None:
    """End-to-end CLI test for index/search/status/clear-cache."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0
        index_payload = _parse_json_output(index_result.output)
        assert index_payload["indexed_files"] == 1
        assert index_payload["indexed_symbols"] > 0

        status_result = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_result.exit_code == 0
        status_payload = _parse_json_output(status_result.output)
        assert status_payload["total_symbols"] > 0

        search_result = runner.invoke(cli, ["search", "add", "--json", "--top-k", "3"], env=env)
        assert search_result.exit_code == 0
        search_payload = _parse_json_output(search_result.output)
        assert search_payload["metadata"]["total_results"] > 0

        clear_result = runner.invoke(cli, ["clear-cache", "--json"], env=env)
        assert clear_result.exit_code == 0
        clear_payload = _parse_json_output(clear_result.output)
        assert clear_payload["cleared"] is True


def test_cli_detects_model_change_and_rebuilds_on_index() -> None:
    """Status/search should require reindex on model change; index should self-rebuild."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)

        env_model_a = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "model-a",
        }
        env_model_b = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "model-b",
        }

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env_model_a)
        assert first_index.exit_code == 0

        status_before = runner.invoke(cli, ["status", "--json"], env=env_model_a)
        assert status_before.exit_code == 0
        status_before_payload = _parse_json_output(status_before.output)
        assert status_before_payload["needs_reindex"] is False
        assert status_before_payload["expected_index_profile"] == "local:model-a"
        assert status_before_payload["cached_index_profile"] == "local:model-a"

        status_changed = runner.invoke(cli, ["status", "--json"], env=env_model_b)
        assert status_changed.exit_code == 0
        status_changed_payload = _parse_json_output(status_changed.output)
        assert status_changed_payload["needs_reindex"] is True
        assert status_changed_payload["expected_index_profile"] == "local:model-b"
        assert status_changed_payload["cached_index_profile"] == "local:model-a"
        assert "embedding profile changed" in str(status_changed_payload["reindex_reason"])

        search_changed = runner.invoke(cli, ["search", "add", "--json"], env=env_model_b)
        assert search_changed.exit_code == 0
        search_changed_payload = _parse_json_output(search_changed.output)
        assert search_changed_payload["results"] == []
        assert search_changed_payload["metadata"]["needs_reindex"] is True
        assert "embedding profile changed" in str(search_changed_payload["metadata"]["reindex_reason"])

        rebuild_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env_model_b)
        assert rebuild_index.exit_code == 0
        rebuild_payload = _parse_json_output(rebuild_index.output)
        assert rebuild_payload["indexed_files"] == 1
        assert rebuild_payload["indexed_symbols"] > 0

        status_after = runner.invoke(cli, ["status", "--json"], env=env_model_b)
        assert status_after.exit_code == 0
        status_after_payload = _parse_json_output(status_after.output)
        assert status_after_payload["needs_reindex"] is False
        assert status_after_payload["expected_index_profile"] == "local:model-b"
        assert status_after_payload["cached_index_profile"] == "local:model-b"


def test_cli_status_and_index_self_heal_corrupted_cache_idempotently() -> None:
    """Status/index should self-heal a corrupted DB once and remain stable afterwards."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        db_path = Path(cache_dir) / "index.db"
        db_path.write_bytes(b"broken sqlite bytes")
        Path(f"{db_path}-wal").write_bytes(b"broken wal")
        Path(f"{db_path}-shm").write_bytes(b"broken shm")

        first_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert first_status.exit_code == 0
        assert "Cache corruption detected at" in first_status.output
        first_payload = _parse_json_output(first_status.output)
        assert first_payload["needs_reindex"] is True
        assert int(first_payload["total_symbols"]) == 0
        quarantined_after_first = sorted(
            path.name for path in Path(cache_dir).iterdir() if ".corrupt." in path.name
        )
        assert quarantined_after_first
        assert not Path(f"{db_path}-wal").exists()
        assert not Path(f"{db_path}-shm").exists()

        second_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert second_status.exit_code == 0
        assert "Cache corruption detected at" not in second_status.output
        second_payload = _parse_json_output(second_status.output)
        assert second_payload["needs_reindex"] is True
        quarantined_after_second = sorted(
            path.name for path in Path(cache_dir).iterdir() if ".corrupt." in path.name
        )
        assert quarantined_after_second == quarantined_after_first

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0
        index_payload = _parse_json_output(index_result.output)
        assert index_payload["indexed_files"] == 1
        assert index_payload["indexed_symbols"] > 0

        final_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert final_status.exit_code == 0
        final_payload = _parse_json_output(final_status.output)
        assert final_payload["needs_reindex"] is False
        assert int(final_payload["total_symbols"]) > 0


def test_cli_search_self_heals_corrupted_cache() -> None:
    """Search should recover from cache corruption and return reindex metadata, not crash."""
    runner = CliRunner()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {"GLOGGUR_CACHE_DIR": cache_dir}
    db_path = Path(cache_dir) / "index.db"
    db_path.write_bytes(b"broken sqlite bytes")

    first_search = runner.invoke(cli, ["search", "add", "--json"], env=env)
    assert first_search.exit_code == 0
    assert "Cache corruption detected at" in first_search.output
    first_payload = _parse_json_output(first_search.output)
    assert first_payload["results"] == []
    metadata = first_payload["metadata"]
    assert isinstance(metadata, dict)
    assert int(metadata["total_results"]) == 0

    second_search = runner.invoke(cli, ["search", "add", "--json"], env=env)
    assert second_search.exit_code == 0
    assert "Cache corruption detected at" not in second_search.output


def test_cli_clear_cache_self_heals_corrupted_cache() -> None:
    """clear-cache should recover from corruption instead of failing with a DB error."""
    runner = CliRunner()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {"GLOGGUR_CACHE_DIR": cache_dir}
    db_path = Path(cache_dir) / "index.db"
    db_path.write_bytes(b"broken sqlite bytes")

    result = runner.invoke(cli, ["clear-cache", "--json"], env=env)
    assert result.exit_code == 0
    assert "Cache corruption detected at" in result.output
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True


def test_cli_inspect_self_heals_corrupted_cache() -> None:
    """Inspect command should run after automatic corruption recovery."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}
        db_path = Path(cache_dir) / "index.db"
        db_path.write_bytes(b"broken sqlite bytes")

        result = runner.invoke(cli, ["inspect", str(repo), "--json"], env=env)
        assert result.exit_code == 0
        assert "Cache corruption detected at" in result.output
        payload = _parse_json_output(result.output)
        assert int(payload["reports_total"]) >= 0
