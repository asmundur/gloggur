from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from gloggur.search import attach_legacy_search_contract


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
        "GLOGGUR_EMBEDDING_PROVIDER": "test",
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

        saw_updated_search_result = False
        last_status_again_payload: dict[str, object] | None = None
        last_search_payload: dict[str, object] | None = None
        for _ in range(100):
            status_again = _run_cli(
                ["watch", "status", "--config", str(config_path), "--json"],
                env,
                timeout=30,
            )
            assert status_again.returncode == 0, f"{status_again.stderr}\n{status_again.stdout}"
            status_again_payload = _parse_json_payload(status_again.stdout)
            last_status_again_payload = status_again_payload
            assert status_again_payload.get("running") is True
            assert int(status_again_payload.get("pid", 0)) == started_pid
            search = _run_cli(["search", updated_phrase, "--json", "--top-k", "5"], env, timeout=30)
            assert search.returncode == 0, f"{search.stderr}\n{search.stdout}"
            search_payload = _parse_json_payload(search.stdout)
            last_search_payload = search_payload
            results = search_payload.get("results", [])
            assert isinstance(results, list)
            if any(item.get("file") == str(target) for item in results):
                saw_updated_search_result = True
                break
            time.sleep(0.1)
        log_content = log_file.read_text(encoding="utf8") if log_file.exists() else "<missing log file>"
        assert saw_updated_search_result, (
            "watch daemon did not surface the updated file in search results in time\n"
            f"last_status={json.dumps(last_status_again_payload, indent=2, sort_keys=True) if last_status_again_payload is not None else '<missing>'}\n"
            f"last_search={json.dumps(last_search_payload, indent=2, sort_keys=True) if last_search_payload is not None else '<missing>'}\n"
            f"log_file={log_content}"
        )

        assert last_search_payload is not None
        search_payload = last_search_payload
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


def test_watch_status_fails_closed_from_last_batch_when_summary_counters_drift(
    tmp_path: Path,
) -> None:
    """watch status should remain unhealthy when last_batch reports failure but summary counters drift."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    target = repo / "sample.py"
    target.write_text(
        "def watch_target() -> str:\n"
        '    """before watch failure contract drift"""\n'
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
        "GLOGGUR_EMBEDDING_PROVIDER": "test",
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

    index = _run_cli(["index", str(repo), "--json"], env, timeout=60)
    assert index.returncode == 0, f"{index.stderr}\n{index.stdout}"

    vectors_path = cache_dir / "vectors.json"
    vectors_payload = json.loads(vectors_path.read_text(encoding="utf8"))
    assert isinstance(vectors_payload, dict)
    raw_map = vectors_payload.get("symbol_to_vector_id", {})
    assert isinstance(raw_map, dict)
    ghost_vector_id = int(vectors_payload.get("next_vector_id", 1))
    raw_map["ghost::watch::symbol"] = ghost_vector_id
    vectors_payload["symbol_to_vector_id"] = raw_map
    vectors_payload["next_vector_id"] = ghost_vector_id + 1
    vectors_path.write_text(json.dumps(vectors_payload, indent=2), encoding="utf8")

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
            status_label = str(payload.get("status", "")).strip().lower()
            if (
                payload.get("running") is True
                and status_label in {"running", "running_with_errors"}
            ):
                running_payload = payload
                break
            time.sleep(0.1)
        assert running_payload is not None, "watch daemon never reached stable running state"

        target.write_text(
            "def watch_target() -> str:\n"
            '    """after watch failure contract drift"""\n'
            '    return "after"\n',
            encoding="utf8",
        )

        failed_payload: dict[str, object] | None = None
        last_status_payload: dict[str, object] | None = None
        for _ in range(200):
            status = _run_cli(
                ["watch", "status", "--config", str(config_path), "--json"],
                env,
                timeout=30,
            )
            assert status.returncode == 0, f"{status.stderr}\n{status.stdout}"
            payload = _parse_json_payload(status.stdout)
            last_status_payload = payload
            reasons = payload.get("failed_reasons", {})
            if isinstance(reasons, dict) and reasons.get("vector_metadata_mismatch", 0):
                failed_payload = payload
                break
            time.sleep(0.1)
        log_content = log_file.read_text(encoding="utf8") if log_file.exists() else "<missing log file>"
        assert failed_payload is not None, (
            "watch daemon did not report vector_metadata_mismatch\n"
            f"last_status={json.dumps(last_status_payload, indent=2, sort_keys=True) if last_status_payload is not None else '<missing>'}\n"
            f"log_file={log_content}"
        )

        assert failed_payload["status"] == "running_with_errors"
        failure_codes = failed_payload.get("failure_codes", [])
        assert isinstance(failure_codes, list)
        assert "vector_metadata_mismatch" in failure_codes
        guidance = failed_payload.get("failure_guidance", {})
        assert isinstance(guidance, dict)
        assert "vector_metadata_mismatch" in guidance
        assert isinstance(guidance["vector_metadata_mismatch"], list)
        assert guidance["vector_metadata_mismatch"]
        last_batch = failed_payload.get("last_batch", {})
        assert isinstance(last_batch, dict)
        last_batch_codes = last_batch.get("failure_codes", [])
        assert isinstance(last_batch_codes, list)
        assert "vector_metadata_mismatch" in last_batch_codes
        last_batch_guidance = last_batch.get("failure_guidance", {})
        assert isinstance(last_batch_guidance, dict)
        assert "vector_metadata_mismatch" in last_batch_guidance
        assert isinstance(last_batch_guidance["vector_metadata_mismatch"], list)
        assert last_batch_guidance["vector_metadata_mismatch"]

        state_payload = json.loads(state_file.read_text(encoding="utf8"))
        assert isinstance(state_payload, dict)
        state_payload["failed"] = 0
        state_payload["error_count"] = 0
        state_payload["failed_reasons"] = {}
        state_payload.pop("failure_codes", None)
        state_payload.pop("failure_guidance", None)
        state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf8")

        drifted_status = _run_cli(
            ["watch", "status", "--config", str(config_path), "--json"],
            env,
            timeout=30,
        )
        assert drifted_status.returncode == 0, f"{drifted_status.stderr}\n{drifted_status.stdout}"
        drifted_payload = _parse_json_payload(drifted_status.stdout)
        assert drifted_payload.get("running") is True
        assert drifted_payload.get("status") == "running_with_errors"
        drifted_reasons = drifted_payload.get("failed_reasons", {})
        assert isinstance(drifted_reasons, dict)
        assert drifted_reasons.get("vector_metadata_mismatch", 0) >= 1
        drifted_codes = drifted_payload.get("failure_codes", [])
        assert isinstance(drifted_codes, list)
        assert "vector_metadata_mismatch" in drifted_codes
        drifted_guidance = drifted_payload.get("failure_guidance", {})
        assert isinstance(drifted_guidance, dict)
        assert "vector_metadata_mismatch" in drifted_guidance
        assert isinstance(drifted_guidance["vector_metadata_mismatch"], list)
        assert drifted_guidance["vector_metadata_mismatch"]
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
