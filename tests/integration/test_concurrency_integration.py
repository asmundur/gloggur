from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.verification.fixtures import TestFixtures


def _write_fallback_marker(cache_dir: str) -> None:
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _parse_json_payload(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    return json.loads(output[start:])


def _run_cli(
    args: list[str],
    env: dict[str, str],
    *,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gloggur.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_concurrent_index_runs_keep_cache_consistent() -> None:
    """Two concurrent index runs should produce valid searchable cache state."""
    with TestFixtures() as fixtures:
        files = {
            f"module_{idx}.py": (
                f"def handler_{idx}(value: int) -> int:\n"
                f"    return value + {idx}\n"
            )
            for idx in range(300)
        }
        repo = fixtures.create_temp_repo(files)
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
        }
        second_env = {
            **env,
            "GLOGGUR_CACHE_LOCK_TIMEOUT_MS": "150",
        }

        first = subprocess.Popen(
            [sys.executable, "-m", "gloggur.cli.main", "index", str(repo), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        time.sleep(0.05)
        second = subprocess.Popen(
            [sys.executable, "-m", "gloggur.cli.main", "index", str(repo), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=second_env,
        )

        first_stdout, first_stderr = first.communicate(timeout=240)
        second_stdout, second_stderr = second.communicate(timeout=240)

        assert first.returncode == 0, f"{first_stderr}\n{first_stdout}"
        first_payload = _parse_json_payload(first_stdout)
        assert int(first_payload["indexed_files"]) > 0

        if second.returncode == 0:
            second_payload = _parse_json_payload(second_stdout)
            assert int(second_payload["indexed_files"]) > 0
        else:
            second_payload = _parse_json_payload(second_stdout)
            error = second_payload["error"]
            assert isinstance(error, dict)
            assert error["operation"] == "acquire cache write lock"
            assert "timed out" in str(error["detail"])
            assert "IO failure [unknown_io_error]" in second_stderr

        status = _run_cli(["status", "--json"], env, timeout=60)
        assert status.returncode == 0, status.stderr
        status_payload = _parse_json_payload(status.stdout)
        assert status_payload["needs_reindex"] is False
        assert int(status_payload["total_symbols"]) > 0

        search = _run_cli(["search", "handler_299", "--json", "--top-k", "5"], env, timeout=60)
        assert search.returncode == 0, search.stderr
        search_payload = _parse_json_payload(search.stdout)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert int(metadata["total_results"]) > 0


def test_index_lock_contention_fails_fast_without_hanging() -> None:
    """When lock is held, index command should fail quickly with explicit lock timeout."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "sample.py": (
                    "def sample(value: int) -> int:\n"
                    "    return value + 1\n"
                )
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
        }
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os, time\n"
                    "from gloggur.indexer.concurrency import cache_write_lock\n"
                    "cache_dir = os.environ['GLOGGUR_CACHE_DIR']\n"
                    "ready = os.path.join(cache_dir, '.lock-held')\n"
                    "with cache_write_lock(cache_dir):\n"
                    "    with open(ready, 'w', encoding='utf8') as handle:\n"
                    "        handle.write('1')\n"
                    "    time.sleep(1.5)\n"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            ready_path = Path(cache_dir) / ".lock-held"
            for _ in range(100):
                if ready_path.exists():
                    break
                time.sleep(0.01)
            assert ready_path.exists()

            blocked_env = {**env, "GLOGGUR_CACHE_LOCK_TIMEOUT_MS": "100"}
            start = time.monotonic()
            blocked = _run_cli(["index", str(repo), "--json"], blocked_env, timeout=10)
            elapsed = time.monotonic() - start
            assert blocked.returncode == 1
            assert elapsed < 2.0
            payload = _parse_json_payload(blocked.stdout)
            error = payload["error"]
            assert isinstance(error, dict)
            assert error["operation"] == "acquire cache write lock"
            assert "timed out" in str(error["detail"])
        finally:
            holder.wait(timeout=10)


def test_clear_cache_lock_contention_fails_fast_without_hanging() -> None:
    """When writer lock is held, clear-cache should fail quickly with lock-timeout payload."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {
        **os.environ,
        "GLOGGUR_CACHE_DIR": cache_dir,
        "GLOGGUR_LOCAL_MODEL": "local",
    }
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, time\n"
                "from gloggur.indexer.concurrency import cache_write_lock\n"
                "cache_dir = os.environ['GLOGGUR_CACHE_DIR']\n"
                "ready = os.path.join(cache_dir, '.lock-held')\n"
                "with cache_write_lock(cache_dir):\n"
                "    with open(ready, 'w', encoding='utf8') as handle:\n"
                "        handle.write('1')\n"
                "    time.sleep(1.5)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        ready_path = Path(cache_dir) / ".lock-held"
        for _ in range(100):
            if ready_path.exists():
                break
            time.sleep(0.01)
        assert ready_path.exists()

        blocked_env = {**env, "GLOGGUR_CACHE_LOCK_TIMEOUT_MS": "100"}
        start = time.monotonic()
        blocked = _run_cli(["clear-cache", "--json"], blocked_env, timeout=10)
        elapsed = time.monotonic() - start
        assert blocked.returncode == 1
        assert elapsed < 2.0
        payload = _parse_json_payload(blocked.stdout)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["operation"] == "acquire cache write lock"
        assert "timed out" in str(error["detail"])
    finally:
        holder.wait(timeout=10)


def test_interrupted_index_run_preserves_needs_reindex_signal() -> None:
    """Interrupted index runs must not advertise healthy status/search state."""
    with TestFixtures() as fixtures:
        files = {
            f"module_{idx}.py": (
                f"def handler_{idx}(value: int) -> int:\n"
                f"    return value + {idx}\n"
            )
            for idx in range(120)
        }
        repo = fixtures.create_temp_repo(files)
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        base_env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
        }

        initial = _run_cli(["index", str(repo), "--json"], base_env, timeout=240)
        assert initial.returncode == 0, initial.stderr
        baseline_status = _run_cli(["status", "--json"], base_env, timeout=60)
        assert baseline_status.returncode == 0, baseline_status.stderr
        baseline_payload = _parse_json_payload(baseline_status.stdout)
        assert baseline_payload["needs_reindex"] is False

        paused_env = {
            **base_env,
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_MS": "3000",
        }
        interrupted = subprocess.Popen(
            [sys.executable, "-m", "gloggur.cli.main", "index", str(repo), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=paused_env,
        )
        try:
            saw_unhealthy = False
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                status = _run_cli(["status", "--json"], base_env, timeout=60)
                assert status.returncode == 0, status.stderr
                payload = _parse_json_payload(status.stdout)
                if payload["needs_reindex"] is True:
                    saw_unhealthy = True
                    break
                time.sleep(0.05)
            assert saw_unhealthy, "Did not observe metadata invalidation window"
        finally:
            interrupted.terminate()
            interrupted.wait(timeout=15)

        after_interrupt_status = _run_cli(["status", "--json"], base_env, timeout=60)
        assert after_interrupt_status.returncode == 0, after_interrupt_status.stderr
        after_interrupt_payload = _parse_json_payload(after_interrupt_status.stdout)
        assert after_interrupt_payload["needs_reindex"] is True

        search_after_interrupt = _run_cli(
            ["search", "handler_99", "--json", "--top-k", "5"],
            base_env,
            timeout=60,
        )
        assert search_after_interrupt.returncode == 0, search_after_interrupt.stderr
        search_payload = _parse_json_payload(search_after_interrupt.stdout)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["needs_reindex"] is True

        recovery = _run_cli(["index", str(repo), "--json"], base_env, timeout=240)
        assert recovery.returncode == 0, recovery.stderr
        final_status = _run_cli(["status", "--json"], base_env, timeout=60)
        assert final_status.returncode == 0, final_status.stderr
        final_payload = _parse_json_payload(final_status.stdout)
        assert final_payload["needs_reindex"] is False
