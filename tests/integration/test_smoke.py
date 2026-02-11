from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
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


def _invoke_json(runner: CliRunner, args: list[str], env: dict[str, str]) -> dict[str, object]:
    """Invoke the CLI and parse its JSON payload."""
    result = runner.invoke(cli, args, env=env)
    assert result.exit_code == 0, result.output
    return _parse_json_output(result.output)


def test_smoke_basic_indexing() -> None:
    """Smoke: indexing should populate cache and report consistent symbol counts."""
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": fixtures.create_sample_python_file()})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        payload = _invoke_json(runner, ["index", str(repo), "--json"], env)
        indexed_files = int(payload.get("indexed_files", 0))
        indexed_symbols = int(payload.get("indexed_symbols", 0))
        assert indexed_files > 0
        assert indexed_symbols > 0

        db_path = Path(cache_dir) / "index.db"
        assert db_path.exists()
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
        assert int(row[0]) == indexed_symbols


def test_smoke_incremental_indexing() -> None:
    """Smoke: second index skips unchanged files and reindexes one modified file."""
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "sample.py": fixtures.create_sample_python_file(),
                "helper.py": "def helper() -> int:\n    return 1\n",
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first = _invoke_json(runner, ["index", str(repo), "--json"], env)
        second = _invoke_json(runner, ["index", str(repo), "--json"], env)

        first_indexed = int(first.get("indexed_files", 0))
        second_indexed = int(second.get("indexed_files", 0))
        second_symbols = int(second.get("indexed_symbols", 0))
        second_skipped = int(second.get("skipped_files", 0))
        assert first_indexed > 0
        assert second_indexed == 0
        assert second_symbols == 0
        assert second_skipped >= first_indexed

        target = repo / "sample.py"
        original = target.read_text(encoding="utf8")
        target.write_text(original + "\n# modified for smoke test\n", encoding="utf8")
        try:
            modified = _invoke_json(runner, ["index", str(repo), "--json"], env)
        finally:
            target.write_text(original, encoding="utf8")

        assert int(modified.get("indexed_files", 0)) == 1


def test_smoke_search_filters() -> None:
    """Smoke: search should return schema-complete results and honor filters."""
    runner = CliRunner()
    fixture_content = (
        '"""Fixture for search smoke test."""\n\n'
        "def search_fixture_function() -> str:\n"
        '    """Return a fixture-specific payload for search."""\n'
        '    return "search fixture function"\n\n'
        "def helper_function() -> int:\n"
        "    return 42\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"search_fixture.py": fixture_content})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        _invoke_json(runner, ["index", str(repo), "--json"], env)
        payload = _invoke_json(runner, ["search", "search fixture function", "--json", "--top-k", "10"], env)
        results = payload.get("results", [])
        assert isinstance(results, list) and results

        required = {"symbol", "kind", "file", "line", "signature", "similarity_score"}
        for item in results:
            assert required.issubset(item)

        expected_file = str(repo / "search_fixture.py")
        filtered = _invoke_json(
            runner,
            [
                "search",
                "search fixture function",
                "--json",
                "--top-k",
                "10",
                "--kind",
                "function",
                "--file",
                expected_file,
            ],
            env,
        )
        filtered_results = filtered.get("results", [])
        assert isinstance(filtered_results, list) and filtered_results
        assert all(item.get("kind") == "function" for item in filtered_results)
        assert all(item.get("file") == expected_file for item in filtered_results)


def test_smoke_status_and_clear_cache() -> None:
    """Smoke: status should reflect symbols and clear-cache should reset to zero."""
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": fixtures.create_sample_python_file()})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        _invoke_json(runner, ["index", str(repo), "--json"], env)
        status_before = _invoke_json(runner, ["status", "--json"], env)
        assert int(status_before.get("total_symbols", 0)) > 0

        cleared = _invoke_json(runner, ["clear-cache", "--json"], env)
        assert bool(cleared.get("cleared")) is True

        status_after = _invoke_json(runner, ["status", "--json"], env)
        assert int(status_after.get("total_symbols", 0)) == 0
