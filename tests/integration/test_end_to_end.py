from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from scripts.verification.fixtures import TestFixtures

pytest.importorskip("faiss")


def _write_fallback_marker(cache_dir: str) -> None:
    """Create the local embedding fallback marker file."""
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _run_cli(args: list[str], env: dict[str, str]) -> dict[str, object]:
    """Run the gloggur CLI and parse JSON output."""
    command = [sys.executable, "-m", "gloggur.cli.main", *args]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def test_end_to_end_index_and_search() -> None:
    """End-to-end test for index and search commands."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = dict(**{"GLOGGUR_CACHE_DIR": cache_dir}, **{"PYTHONPATH": str(Path.cwd())})

        index_payload = _run_cli(["index", str(repo), "--json"], env=env)
        assert index_payload["indexed_files"] == 1
        assert index_payload["indexed_symbols"] > 0

        search_payload = _run_cli(["search", "add", "--json", "--top-k", "3"], env=env)
        assert search_payload["metadata"]["total_results"] > 0
