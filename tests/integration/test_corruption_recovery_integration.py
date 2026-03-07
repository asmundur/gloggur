from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gloggur.search import attach_legacy_search_contract


def _write_fallback_marker(cache_dir: Path) -> None:
    marker = cache_dir / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _parse_json_payload(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    payload = json.loads(output[start:])
    if isinstance(payload, dict):
        return attach_legacy_search_contract(payload)
    return payload


def _run_cli(
    args: list[str],
    env: dict[str, str],
    *,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gloggur.cli.main", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )


def _build_no_faiss_env(cache_dir: Path, fake_root: Path) -> dict[str, str]:
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "faiss.py").write_text("raise ImportError('simulated no-faiss runtime')\n")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = str(fake_root)
    if existing_pythonpath:
        pythonpath = f"{pythonpath}{os.pathsep}{existing_pythonpath}"
    return {
        **os.environ,
        "GLOGGUR_CACHE_DIR": str(cache_dir),
        "PYTHONPATH": pythonpath,
    }


@pytest.mark.parametrize(
    ("command", "extra_args"),
    [
        ("status", []),
        ("search", ["add"]),
        ("inspect", []),
        ("clear-cache", []),
        ("index", []),
    ],
)
def test_corruption_recovery_commands_work_in_simulated_no_faiss_runtime(
    tmp_path: Path,
    command: str,
    extra_args: list[str],
) -> None:
    """Core commands should recover from corrupted DB even when FAISS import is unavailable."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_fallback_marker(cache_dir)
    db_path = cache_dir / "index.db"
    db_path.write_bytes(b"broken sqlite bytes")
    (cache_dir / "index.db-wal").write_bytes(b"broken wal bytes")
    (cache_dir / "index.db-shm").write_bytes(b"broken shm bytes")

    env = _build_no_faiss_env(cache_dir, tmp_path / "no_faiss")
    args_map = {
        "status": ["status", "--json"],
        "search": ["search", *extra_args, "--json"],
        "inspect": ["inspect", str(repo), "--json"],
        "clear-cache": ["clear-cache", "--json"],
        "index": ["index", str(repo), "--json"],
    }

    result = _run_cli(args_map[command], env, timeout=120)
    expected_returncode = 1 if command == "search" else 0
    assert result.returncode == expected_returncode, f"{result.stderr}\n{result.stdout}"
    payload = _parse_json_payload(result.stdout)
    if command == "status":
        assert "needs_reindex" in payload
    elif command == "search":
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["code"] == "search_cache_not_ready"
        metadata = payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["needs_reindex"] is True
        assert metadata["resume_reason_codes"] == [
            "missing_index_metadata",
            "cache_corruption_recovered",
        ]
    elif command == "inspect":
        assert "reports_total" in payload
    elif command == "clear-cache":
        assert payload.get("cleared") is True
    elif command == "index":
        assert int(payload.get("indexed_files", 0)) >= 0
        placeholder = cache_dir / "vectors.index"
        assert placeholder.exists()
        assert "FAISS_UNAVAILABLE" in placeholder.read_text(encoding="utf8")

    combined = f"{result.stdout}\n{result.stderr}"
    assert "Traceback (most recent call last)" not in combined
    assert "Cache corruption detected at" in combined


def test_concurrent_status_recovery_attempts_are_bounded(tmp_path: Path) -> None:
    """Concurrent status calls against corrupted cache should not hang."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_fallback_marker(cache_dir)
    db_path = cache_dir / "index.db"
    db_path.write_bytes(b"broken sqlite bytes")
    (cache_dir / "index.db-wal").write_bytes(b"broken wal bytes")
    (cache_dir / "index.db-shm").write_bytes(b"broken shm bytes")
    env = {
        **os.environ,
        "GLOGGUR_CACHE_DIR": str(cache_dir),
    }

    start = time.monotonic()
    first = subprocess.Popen(
        [sys.executable, "-m", "gloggur.cli.main", "status", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    second = subprocess.Popen(
        [sys.executable, "-m", "gloggur.cli.main", "status", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    first_stdout, first_stderr = first.communicate(timeout=30)
    second_stdout, second_stderr = second.communicate(timeout=30)
    elapsed = time.monotonic() - start
    assert elapsed < 30

    results = [
        (first.returncode, first_stdout, first_stderr),
        (second.returncode, second_stdout, second_stderr),
    ]
    assert any(return_code == 0 for return_code, _stdout, _stderr in results)

    for return_code, stdout, stderr in results:
        assert return_code in {0, 1}
        combined = f"{stdout}\n{stderr}"
        assert "Traceback (most recent call last)" not in combined
        if return_code == 1:
            payload = _parse_json_payload(stdout)
            error = payload["error"]
            assert isinstance(error, dict)
            assert error.get("type") == "io_failure"
            assert error.get("operation") == "recover corrupted cache database"

    status_after = _run_cli(["status", "--json"], env, timeout=30)
    assert status_after.returncode == 0, status_after.stderr
    final_payload = _parse_json_payload(status_after.stdout)
    assert final_payload["needs_reindex"] is True
