from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _parse_json_payload(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    return json.loads(output[start:])


def _run_cli(
    args: list[str],
    env: dict[str, str],
    *,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gloggur.cli.main", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_watch_lifecycle_commands_with_env_overrides(tmp_path: Path) -> None:
    """watch lifecycle should run with custom runtime file paths and process save events."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    target = repo / "sample.py"
    target.write_text(
        "def watch_target() -> str:\n"
        '    """before watch lifecycle update phrase"""\n'
        '    return "before"\n',
        encoding="utf8",
    )
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / ".gloggur.yaml"
    state_file = tmp_path / "runtime" / "custom_watch_state.json"
    pid_file = tmp_path / "runtime" / "custom_watch.pid"
    log_file = tmp_path / "runtime" / "custom_watch.log"
    env = {
        **os.environ,
        "GLOGGUR_CACHE_DIR": str(cache_dir),
        "GLOGGUR_LOCAL_MODEL": "local",
        "GLOGGUR_LOCAL_FALLBACK": "1",
        "WATCHFILES_FORCE_POLLING": "1",
        "WATCHFILES_POLL_DELAY_MS": "50",
        "GLOGGUR_WATCH_STATE_FILE": str(state_file),
        "GLOGGUR_WATCH_PID_FILE": str(pid_file),
        "GLOGGUR_WATCH_LOG_FILE": str(log_file),
    }

    init = _run_cli(
        ["watch", "init", str(repo), "--config", str(config_path), "--json"],
        env,
        timeout=30,
    )
    assert init.returncode == 0, f"{init.stderr}\n{init.stdout}"
    init_payload = _parse_json_payload(init.stdout)
    assert init_payload["initialized"] is True

    started_pid: int | None = None
    try:
        started = _run_cli(
            ["watch", "start", "--config", str(config_path), "--daemon", "--json"],
            env,
            timeout=30,
        )
        assert started.returncode == 0, f"{started.stderr}\n{started.stdout}"
        started_payload = _parse_json_payload(started.stdout)
        assert started_payload["started"] is True
        started_pid = int(started_payload["pid"])
        assert started_pid > 0

        running_payload: dict[str, object] | None = None
        for _ in range(50):
            status = _run_cli(
                ["watch", "status", "--config", str(config_path), "--json"],
                env,
                timeout=30,
            )
            assert status.returncode == 0, f"{status.stderr}\n{status.stdout}"
            payload = _parse_json_payload(status.stdout)
            if payload.get("running") is True:
                running_payload = payload
                break
            time.sleep(0.1)
        assert running_payload is not None, "watch daemon never reached running state"
        assert running_payload["state_file"] == str(state_file)
        assert running_payload["log_file"] == str(log_file)
        assert Path(str(running_payload["state_file"])).exists()
        assert pid_file.exists()

        updated_phrase = "after watch lifecycle update phrase"
        target.write_text(
            "def watch_target() -> str:\n"
            f'    """{updated_phrase}"""\n'
            '    return "after"\n',
            encoding="utf8",
        )

        saw_indexed_event = False
        for _ in range(100):
            status_again = _run_cli(
                ["watch", "status", "--config", str(config_path), "--json"],
                env,
                timeout=30,
            )
            assert status_again.returncode == 0, f"{status_again.stderr}\n{status_again.stdout}"
            status_again_payload = _parse_json_payload(status_again.stdout)
            assert status_again_payload.get("running") is True
            assert int(status_again_payload.get("pid", 0)) == started_pid
            indexed_files = int(status_again_payload.get("indexed_files", 0))
            if indexed_files > 0:
                saw_indexed_event = True
                break
            time.sleep(0.1)
        assert saw_indexed_event, "watch daemon did not process file-save events in time"

        search = _run_cli(["search", updated_phrase, "--json", "--top-k", "5"], env, timeout=30)
        assert search.returncode == 0, f"{search.stderr}\n{search.stdout}"
        search_payload = _parse_json_payload(search.stdout)
        metadata = search_payload.get("metadata", {})
        assert isinstance(metadata, dict)
        assert metadata.get("needs_reindex") is not True
        results = search_payload.get("results", [])
        assert isinstance(results, list)
        assert any(item.get("file") == str(target) for item in results)
    finally:
        stopped = _run_cli(
            ["watch", "stop", "--config", str(config_path), "--json"],
            env,
            timeout=30,
        )
        if stopped.returncode == 0:
            stopped_payload = _parse_json_payload(stopped.stdout)
            assert stopped_payload.get("running") is False
        elif started_pid:
            try:
                os.kill(started_pid, signal.SIGTERM)
            except OSError:
                pass

        final_status = _run_cli(
            ["watch", "status", "--config", str(config_path), "--json"],
            env,
            timeout=30,
        )
        if final_status.returncode == 0:
            final_payload = _parse_json_payload(final_status.stdout)
            assert final_payload.get("running") is False
        if log_file.exists():
            log_content = log_file.read_text(encoding="utf8")
            assert "already_running" not in log_content
