from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gloggur.indexer.concurrency import LockRetryPolicy, build_backoff_schedule
from gloggur.search import attach_legacy_search_contract
from scripts.verification.fixtures import TestFixtures


def _write_fallback_marker(cache_dir: str) -> None:
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _wait_for_path(path: Path, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {path}")


def _tail_text(value: str, *, lines: int = 20) -> str:
    chunks = value.splitlines()
    if not chunks:
        return "<empty>"
    return "\n".join(chunks[-lines:])


def _wait_for_path_or_process_exit(
    path: Path,
    proc: subprocess.Popen[str],
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        return_code = proc.poll()
        if return_code is not None:
            stdout, stderr = proc.communicate(timeout=5)
            raise AssertionError(
                "Process exited before readiness sentinel was created "
                f"(path={path}, timeout={timeout:.1f}s, returncode={return_code}). "
                f"stdout_tail:\n{_tail_text(stdout)}\n"
                f"stderr_tail:\n{_tail_text(stderr)}"
            )
        time.sleep(0.01)

    raise AssertionError(
        "Timed out waiting for readiness sentinel "
        f"(path={path}, timeout={timeout:.1f}s, process_still_running={proc.poll() is None})"
    )


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
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _assert_lock_timeout_contract(
    payload: dict[str, object],
    *,
    lock_timeout_ms: int,
    subprocess_timeout: float,
    holder_pid: int,
) -> None:
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["category"] == "cache_lock_held"
    assert error["operation"] == "acquire cache write lock"
    assert error["holder_pid"] == holder_pid
    assert "timed out" in str(error["detail"])

    waited_ms = int(error["waited_ms"])
    expected_attempts = 1 + len(build_backoff_schedule(LockRetryPolicy(timeout_ms=lock_timeout_ms)))
    assert waited_ms >= lock_timeout_ms
    assert waited_ms < int(subprocess_timeout * 1000)
    assert int(error["attempts"]) == expected_attempts


def _start_lock_holder(
    *,
    cache_dir: str,
    env: dict[str, str],
) -> tuple[subprocess.Popen[str], Path]:
    ready_path = Path(cache_dir) / ".lock-held"
    release_path = Path(cache_dir) / ".lock-release"
    holder_env = {
        **env,
        "GLOGGUR_TEST_LOCK_READY_FILE": str(ready_path),
        "GLOGGUR_TEST_LOCK_RELEASE_FILE": str(release_path),
    }
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, time\n"
                "from gloggur.indexer.concurrency import cache_write_lock\n"
                "cache_dir = os.environ['GLOGGUR_CACHE_DIR']\n"
                "ready = os.environ['GLOGGUR_TEST_LOCK_READY_FILE']\n"
                "release = os.environ['GLOGGUR_TEST_LOCK_RELEASE_FILE']\n"
                "with cache_write_lock(cache_dir):\n"
                "    with open(ready, 'w', encoding='utf8') as handle:\n"
                "        handle.write('1')\n"
                "    deadline = time.monotonic() + 15.0\n"
                "    while not os.path.exists(release) and time.monotonic() < deadline:\n"
                "        time.sleep(0.01)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=holder_env,
    )
    _wait_for_path(ready_path, timeout=2.0)
    return holder, release_path


def _stop_lock_holder(holder: subprocess.Popen[str], release_path: Path) -> None:
    release_path.write_text("1", encoding="utf8")
    holder.wait(timeout=10)


def test_concurrent_index_runs_keep_cache_consistent() -> None:
    """Two concurrent index runs should produce valid searchable cache state."""
    with TestFixtures() as fixtures:
        files = {
            f"module_{idx}.py": (
                f"def handler_{idx}(value: int) -> int:\n" f"    return value + {idx}\n"
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

        results = (
            ("first", first.returncode, first_stdout, first_stderr),
            ("second", second.returncode, second_stdout, second_stderr),
        )
        successful_runs = 0
        for _name, returncode, stdout, stderr in results:
            payload = _parse_json_payload(stdout)
            if returncode == 0:
                assert int(payload["failed"]) == 0
                assert int(payload["files_considered"]) > 0
                successful_runs += 1
                continue
            error = payload["error"]
            assert isinstance(error, dict)
            assert error["operation"] == "acquire cache write lock"
            assert "timed out" in str(error["detail"])
            assert stderr == ""

        assert successful_runs >= 1

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
            {"sample.py": ("def sample(value: int) -> int:\n" "    return value + 1\n")}
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
        }
        holder, release_path = _start_lock_holder(cache_dir=cache_dir, env=env)
        try:
            lock_timeout_ms = 100
            subprocess_timeout = 10.0
            blocked_env = {**env, "GLOGGUR_CACHE_LOCK_TIMEOUT_MS": str(lock_timeout_ms)}
            start = time.monotonic()
            blocked = _run_cli(
                ["index", str(repo), "--json"],
                blocked_env,
                timeout=subprocess_timeout,
            )
            elapsed = time.monotonic() - start
            assert blocked.returncode == 1
            assert elapsed < (subprocess_timeout / 2.0)
            payload = _parse_json_payload(blocked.stdout)
            _assert_lock_timeout_contract(
                payload,
                lock_timeout_ms=lock_timeout_ms,
                subprocess_timeout=subprocess_timeout,
                holder_pid=holder.pid,
            )
        finally:
            _stop_lock_holder(holder, release_path)


def test_clear_cache_lock_contention_fails_fast_without_hanging() -> None:
    """When writer lock is held, clear-cache should fail quickly with lock-timeout payload."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {
        **os.environ,
        "GLOGGUR_CACHE_DIR": cache_dir,
        "GLOGGUR_LOCAL_MODEL": "local",
    }
    holder, release_path = _start_lock_holder(cache_dir=cache_dir, env=env)
    try:
        lock_timeout_ms = 100
        subprocess_timeout = 10.0
        blocked_env = {**env, "GLOGGUR_CACHE_LOCK_TIMEOUT_MS": str(lock_timeout_ms)}
        start = time.monotonic()
        blocked = _run_cli(["clear-cache", "--json"], blocked_env, timeout=subprocess_timeout)
        elapsed = time.monotonic() - start
        assert blocked.returncode == 1
        assert elapsed < (subprocess_timeout / 2.0)
        payload = _parse_json_payload(blocked.stdout)
        _assert_lock_timeout_contract(
            payload,
            lock_timeout_ms=lock_timeout_ms,
            subprocess_timeout=subprocess_timeout,
            holder_pid=holder.pid,
        )
    finally:
        _stop_lock_holder(holder, release_path)


def test_clear_cache_during_index_run_fails_fast_and_index_completes_cleanly() -> None:
    """clear-cache during an active index run should fail fast and not corrupt the eventual index."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                f"module_{idx}.py": (
                    f"def handler_{idx}(value: int) -> int:\n" f"    return value + {idx}\n"
                )
                for idx in range(40)
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        ready_path = Path(cache_dir) / ".metadata-delete-ready"
        index_env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_MS": "1500",
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_READY_FILE": str(ready_path),
        }
        index_proc = subprocess.Popen(
            [sys.executable, "-m", "gloggur.cli.main", "index", str(repo), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=index_env,
        )
        try:
            _wait_for_path_or_process_exit(ready_path, index_proc, timeout=8.0)
            blocked_env = {
                **os.environ,
                "GLOGGUR_CACHE_DIR": cache_dir,
                "GLOGGUR_LOCAL_MODEL": "local",
                "GLOGGUR_CACHE_LOCK_TIMEOUT_MS": "100",
            }
            blocked = _run_cli(["clear-cache", "--json"], blocked_env, timeout=10)
            assert blocked.returncode == 1
            payload = _parse_json_payload(blocked.stdout)
            error = payload["error"]
            assert isinstance(error, dict)
            assert error["operation"] == "acquire cache write lock"
            assert "timed out" in str(error["detail"])
        finally:
            index_stdout, index_stderr = index_proc.communicate(timeout=240)

        assert index_proc.returncode == 0, index_stderr
        index_payload = _parse_json_payload(index_stdout)
        assert int(index_payload["failed"]) == 0

        status = _run_cli(["status", "--json"], blocked_env, timeout=60)
        assert status.returncode == 0, status.stderr
        status_payload = _parse_json_payload(status.stdout)
        assert status_payload["needs_reindex"] is False
        assert int(status_payload["total_symbols"]) > 0


def test_interrupted_index_run_preserves_needs_reindex_signal() -> None:
    """Interrupted index runs must not advertise healthy status/search state."""
    with TestFixtures() as fixtures:
        files = {
            f"module_{idx}.py": (
                f"def handler_{idx}(value: int) -> int:\n" f"    return value + {idx}\n"
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

        ready_path = Path(cache_dir) / ".metadata-delete-ready"
        paused_env = {
            **base_env,
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_MS": "3000",
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_READY_FILE": str(ready_path),
        }
        interrupted = subprocess.Popen(
            [sys.executable, "-m", "gloggur.cli.main", "index", str(repo), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=paused_env,
        )
        try:
            _wait_for_path_or_process_exit(ready_path, interrupted, timeout=20.0)
            saw_building_state = False
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                status = _run_cli(["status", "--json"], base_env, timeout=60)
                assert status.returncode == 0, status.stderr
                payload = _parse_json_payload(status.stdout)
                build_state = payload.get("build_state")
                if isinstance(build_state, dict) and build_state.get("state") == "building":
                    saw_building_state = True
                    assert payload["needs_reindex"] is False
                    break
                time.sleep(0.05)
            assert (
                saw_building_state
            ), "Did not observe staged build state after metadata-delete pause sentinel"
        finally:
            interrupted.terminate()
            interrupted.wait(timeout=15)

        after_interrupt_status = _run_cli(["status", "--json"], base_env, timeout=60)
        assert after_interrupt_status.returncode == 0, after_interrupt_status.stderr
        after_interrupt_payload = _parse_json_payload(after_interrupt_status.stdout)
        assert after_interrupt_payload["needs_reindex"] is False
        build_state = after_interrupt_payload["build_state"]
        assert isinstance(build_state, dict)
        assert build_state["state"] == "interrupted"
        assert build_state["cleanup_pending"] is True
        assert after_interrupt_payload["resume_reason_codes"] == []

        search_after_interrupt = _run_cli(
            ["search", "handler_99", "--json", "--top-k", "5"],
            base_env,
            timeout=60,
        )
        assert search_after_interrupt.returncode == 0, search_after_interrupt.stderr
        search_payload = _parse_json_payload(search_after_interrupt.stdout)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["needs_reindex"] is False
        assert metadata["resume_reason_codes"] == []

        recovery = _run_cli(["index", str(repo), "--json"], base_env, timeout=240)
        assert recovery.returncode == 0, recovery.stderr
        final_status = _run_cli(["status", "--json"], base_env, timeout=60)
        assert final_status.returncode == 0, final_status.stderr
        final_payload = _parse_json_payload(final_status.stdout)
        assert final_payload["needs_reindex"] is False
        assert final_payload["build_state"] is None


def test_first_time_interrupted_index_leaves_non_ready_cache_with_stable_search_error() -> None:
    """Interrupting the first build should leave status/search on the stable non-ready contract."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                f"module_{idx}.py": (
                    f"def handler_{idx}(value: int) -> int:\n" f"    return value + {idx}\n"
                )
                for idx in range(40)
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        ready_path = Path(cache_dir) / ".metadata-delete-ready"
        paused_env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_MS": "3000",
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_READY_FILE": str(ready_path),
        }
        interrupted = subprocess.Popen(
            [sys.executable, "-m", "gloggur.cli.main", "index", str(repo), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=paused_env,
        )
        try:
            _wait_for_path_or_process_exit(ready_path, interrupted, timeout=8.0)
        finally:
            interrupted.terminate()
            interrupted.wait(timeout=15)

        base_env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
        }
        status = _run_cli(["status", "--json"], base_env, timeout=60)
        assert status.returncode == 0, status.stderr
        status_payload = _parse_json_payload(status.stdout)
        assert status_payload["metadata"] is None
        assert status_payload["needs_reindex"] is True
        assert status_payload["resume_reason_codes"] == [
            "index_interrupted",
            "missing_index_metadata",
        ]
        assert status_payload["total_symbols"] == 0
        build_state = status_payload["build_state"]
        assert isinstance(build_state, dict)
        assert build_state["state"] == "interrupted"

        search = _run_cli(["search", "handler_10", "--json", "--top-k", "5"], base_env, timeout=60)
        assert search.returncode == 1, search.stderr
        search_payload = _parse_json_payload(search.stdout)
        error = search_payload["error"]
        assert isinstance(error, dict)
        assert error["code"] == "search_cache_not_ready"
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["build_state"]["state"] == "interrupted"

        recovery = _run_cli(["index", str(repo), "--json"], base_env, timeout=240)
        assert recovery.returncode == 0, recovery.stderr
        recovered_status = _run_cli(["status", "--json"], base_env, timeout=60)
        recovered_payload = _parse_json_payload(recovered_status.stdout)
        assert recovered_payload["needs_reindex"] is False
        assert recovered_payload["build_state"] is None


def test_sigkill_interrupted_index_is_reported_as_stale_and_recovers() -> None:
    """SIGKILL-interrupted builds should surface stale_build_state and recover on next index."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                f"module_{idx}.py": (
                    f"def handler_{idx}(value: int) -> int:\n" f"    return value + {idx}\n"
                )
                for idx in range(40)
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        ready_path = Path(cache_dir) / ".metadata-delete-ready"
        paused_env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_MS": "3000",
            "GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_READY_FILE": str(ready_path),
        }
        interrupted = subprocess.Popen(
            [sys.executable, "-m", "gloggur.cli.main", "index", str(repo), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=paused_env,
        )
        try:
            _wait_for_path_or_process_exit(ready_path, interrupted, timeout=8.0)
        finally:
            interrupted.kill()
            interrupted.wait(timeout=15)

        base_env = {
            **os.environ,
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "local",
        }
        status = _run_cli(["status", "--json"], base_env, timeout=60)
        assert status.returncode == 0, status.stderr
        status_payload = _parse_json_payload(status.stdout)
        assert status_payload["metadata"] is None
        assert status_payload["needs_reindex"] is True
        assert status_payload["resume_reason_codes"] == [
            "stale_build_state",
            "missing_index_metadata",
        ]
        warning_codes = set(status_payload.get("warning_codes", []))
        assert "stale_build_state" in warning_codes
        assert "build_in_progress" not in warning_codes
        build_state = status_payload["build_state"]
        assert isinstance(build_state, dict)
        assert build_state["state"] == "interrupted"
        assert build_state["cleanup_pending"] is True

        search = _run_cli(["search", "handler_10", "--json", "--top-k", "5"], base_env, timeout=60)
        assert search.returncode == 1, search.stderr
        search_payload = _parse_json_payload(search.stdout)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["resume_reason_codes"] == ["stale_build_state", "missing_index_metadata"]

        recovery = _run_cli(["index", str(repo), "--json"], base_env, timeout=240)
        assert recovery.returncode == 0, recovery.stderr
        recovered_status = _run_cli(["status", "--json"], base_env, timeout=60)
        recovered_payload = _parse_json_payload(recovered_status.stdout)
        assert recovered_payload["needs_reindex"] is False
        assert recovered_payload["build_state"] is None
