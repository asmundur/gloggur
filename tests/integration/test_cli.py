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
    return json.loads(output[start:])


def test_cli_index_search_inspect_status_and_clear_cache() -> None:
    """End-to-end CLI smoke test for index/search/inspect/status/clear-cache."""
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

        inspect_result = runner.invoke(cli, ["inspect", str(repo), "--json"], env=env)
        assert inspect_result.exit_code == 0
        inspect_payload = _parse_json_output(inspect_result.output)
        assert inspect_payload["total"] >= 1

        clear_result = runner.invoke(cli, ["clear-cache", "--json"], env=env)
        assert clear_result.exit_code == 0
        clear_payload = _parse_json_output(clear_result.output)
        assert clear_payload["cleared"] is True
