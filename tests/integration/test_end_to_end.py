from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from gloggur.search import attach_legacy_search_contract
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
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)
    if isinstance(payload, dict):
        return attach_legacy_search_contract(payload)
    return payload


def test_end_to_end_index_and_search() -> None:
    """End-to-end test for index and search commands."""
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

        index_payload = _run_cli(["index", str(repo), "--json"], env=env)
        assert index_payload["indexed_files"] == 1
        assert index_payload["indexed_symbols"] > 0

        search_payload = _run_cli(["search", "add", "--json", "--top-k", "3"], env=env)
        hits = search_payload.get("hits")
        assert isinstance(hits, list)
        assert len(hits) > 0
