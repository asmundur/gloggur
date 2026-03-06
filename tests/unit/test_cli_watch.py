from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import yaml
from click.testing import CliRunner

from gloggur.cli.main import (
    _build_watch_failure_contract,
    _collect_watch_failure_signals,
    cli,
)
from gloggur.config import GloggurConfig
from gloggur.io_failures import StorageIOError
from gloggur.watch.service import is_process_running


def _parse_json_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {output!r}")
    return json.loads(output[start:])


def test_watch_init_writes_default_config(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf8")
    config_file = tmp_path / ".gloggur.yaml"
    config_file.write_text("embedding_provider: openai\ncache_dir: custom-cache\n", encoding="utf8")

    result = runner.invoke(
        cli,
        ["watch", "init", str(repo), "--config", str(config_file), "--json"],
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["initialized"] is True
    assert config_file.exists()
    config_payload = yaml.safe_load(config_file.read_text(encoding="utf8"))
    assert config_payload["watch_enabled"] is True
    assert config_payload["watch_mode"] == "daemon"
    assert config_payload["embedding_provider"] == "openai"
    assert config_payload["cache_dir"] == "custom-cache"


def test_watch_start_rejects_conflicting_mode_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["watch", "start", "--foreground", "--daemon", "--json"])
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "cli_contract_error"
    assert error["code"] == "watch_mode_conflict"
    assert "Use only one of --foreground or --daemon." in str(error["detail"])
    assert payload["failure_codes"] == ["watch_mode_conflict"]
    guidance = payload["failure_guidance"]
    assert isinstance(guidance, dict)
    assert "watch_mode_conflict" in guidance
    assert isinstance(guidance["watch_mode_conflict"], list)
    assert guidance["watch_mode_conflict"]


def test_watch_start_rejects_unsupported_mode(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()

    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="invalid-mode",
    )
    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, None, None),
    )

    result = runner.invoke(
        cli,
        ["watch", "start", "--config", str(tmp_path / "cfg.yaml"), "--json"],
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "cli_contract_error"
    assert error["code"] == "watch_mode_invalid"
    assert "Unsupported watch mode: invalid-mode" in str(error["detail"])
    assert payload["failure_codes"] == ["watch_mode_invalid"]


def test_watch_start_reports_missing_watch_path_with_failure_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(
        watch_path=str(tmp_path / "missing-repo"),
        watch_mode="daemon",
    )
    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, None, None),
    )

    result = runner.invoke(
        cli,
        ["watch", "start", "--config", str(tmp_path / "cfg.yaml"), "--json"],
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "cli_contract_error"
    assert error["code"] == "watch_path_missing"
    assert "Watch path does not exist:" in str(error["detail"])
    assert payload["failure_codes"] == ["watch_path_missing"]


def test_watch_start_foreground_fail_closed_emits_primary_error_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="foreground",
        watch_pid_file=str(tmp_path / "watch.pid"),
    )

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {
                "watch_path": str(repo),
                "files_considered": 1,
                "indexed": 0,
                "unchanged": 0,
                "failed": 1,
                "failed_reasons": {"vector_metadata_mismatch": 1},
                "failed_samples": ["sample.py: vector metadata mismatch"],
                "failure_codes": ["vector_metadata_mismatch"],
                "failure_guidance": {
                    "vector_metadata_mismatch": ["rebuild cache metadata"],
                },
                "indexed_files": 0,
                "indexed_symbols": 0,
                "skipped_files": 0,
                "error_count": 1,
            }

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)

    result = runner.invoke(cli, ["watch", "start", "--foreground", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "watch_failure"
    assert error["code"] == "vector_metadata_mismatch"
    assert "Watch foreground run completed with file-level failures." in str(error["detail"])
    assert payload["failure_codes"] == ["vector_metadata_mismatch"]


def test_watch_start_reports_already_running(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()

    config = GloggurConfig(watch_path=str(repo), watch_mode="daemon")
    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, None, None),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: 6789)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda pid: pid == 6789)

    result = runner.invoke(
        cli,
        ["watch", "start", "--config", str(tmp_path / "cfg.yaml"), "--json"],
    )
    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["started"] is False
    assert payload["reason"] == "already_running"
    assert payload["pid"] == 6789


def test_watch_start_daemon_and_stop(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf8")
    config_file = tmp_path / ".gloggur.yaml"
    init_result = runner.invoke(
        cli,
        ["watch", "init", str(repo), "--config", str(config_file), "--json"],
    )
    assert init_result.exit_code == 0

    class DummyProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        @staticmethod
        def poll() -> None:
            return None

    state = {"running": False, "daemon_child_env": None}

    def fake_running(pid: int | None) -> bool:
        return bool(pid == 4321 and state["running"])

    def fake_popen(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        env = kwargs.get("env")
        if isinstance(env, dict):
            state["daemon_child_env"] = env.get("GLOGGUR_WATCH_DAEMON_CHILD")
        state["running"] = True
        return DummyProcess(4321)

    def fake_kill(pid: int, sig: int) -> None:
        _ = pid, sig
        state["running"] = False

    monkeypatch.setattr("gloggur.cli.main.is_process_running", fake_running)
    monkeypatch.setattr("gloggur.cli.main.subprocess.Popen", fake_popen)
    monkeypatch.setattr("gloggur.cli.main.os.kill", fake_kill)
    monkeypatch.setattr("gloggur.cli.main.time.sleep", lambda _seconds: None)

    started = runner.invoke(
        cli,
        ["watch", "start", "--config", str(config_file), "--daemon", "--json"],
    )
    assert started.exit_code == 0
    started_payload = _parse_json_output(started.output)
    assert started_payload["started"] is True
    assert started_payload["pid"] == 4321
    assert state["daemon_child_env"] == "1"

    alternate_cwd = tmp_path / "other-cwd"
    alternate_cwd.mkdir()
    monkeypatch.chdir(alternate_cwd)

    status_running = runner.invoke(cli, ["watch", "status", "--config", str(config_file), "--json"])
    assert status_running.exit_code == 0
    status_payload = _parse_json_output(status_running.output)
    assert status_payload["running"] is True
    assert status_payload["state_file"] == str(tmp_path / ".gloggur-cache" / "watch_state.json")

    stopped = runner.invoke(cli, ["watch", "stop", "--config", str(config_file), "--json"])
    assert stopped.exit_code == 0
    stopped_payload = _parse_json_output(stopped.output)
    assert stopped_payload["stopped"] is True

    stopped_again = runner.invoke(cli, ["watch", "stop", "--config", str(config_file), "--json"])
    assert stopped_again.exit_code == 0
    stopped_again_payload = _parse_json_output(stopped_again.output)
    assert stopped_again_payload["stopped"] is False
    assert stopped_again_payload["running"] is False


def test_watch_start_daemon_resets_stale_last_batch_failure_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Daemon startup should clear stale last_batch/failure fields before new batches run."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    state_file = tmp_path / "watch_state.json"
    pid_file = tmp_path / "watch.pid"
    log_file = tmp_path / "watch.log"
    state_file.write_text(
        json.dumps(
            {
                "status": "running_with_errors",
                "running": False,
                "failed": 3,
                "error_count": 3,
                "failed_reasons": {"vector_metadata_mismatch": 3},
                "failure_codes": ["vector_metadata_mismatch"],
                "failure_guidance": {"vector_metadata_mismatch": ["stale guidance"]},
                "last_batch": {
                    "failed": 1,
                    "failed_reasons": {"vector_metadata_mismatch": 1},
                },
                "last_error": "stale failure",
            }
        ),
        encoding="utf8",
    )
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="daemon",
        watch_state_file=str(state_file),
        watch_pid_file=str(pid_file),
        watch_log_file=str(log_file),
    )

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {"indexed_files": 0, "indexed_symbols": 0, "skipped_files": 0, "error_count": 0}

    class DummyProcess:
        pid = 4321

        @staticmethod
        def poll() -> None:
            return None

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)
    monkeypatch.setattr("gloggur.cli.main.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "gloggur.cli.main.subprocess.Popen",
        lambda *args, **kwargs: DummyProcess(),
    )

    result = runner.invoke(cli, ["watch", "start", "--daemon", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(state_file.read_text(encoding="utf8"))
    assert payload["status"] == "starting"
    assert payload["running"] is True
    assert payload["failed"] == 0
    assert payload["error_count"] == 0
    assert payload["failed_reasons"] == {}
    assert payload["failure_codes"] == []
    assert payload["failure_guidance"] == {}
    assert payload["last_batch"] == {}
    assert payload["last_error"] is None


def test_watch_stop_handles_missing_pid_file(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf8")
    config_file = tmp_path / ".gloggur.yaml"
    init_result = runner.invoke(
        cli,
        ["watch", "init", str(repo), "--config", str(config_file), "--json"],
    )
    assert init_result.exit_code == 0

    stopped = runner.invoke(cli, ["watch", "stop", "--config", str(config_file), "--json"])
    assert stopped.exit_code == 0
    payload = _parse_json_output(stopped.output)
    assert payload["stopped"] is False
    assert payload["running"] is False


def test_watch_init_json_reports_structured_config_write_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf8")
    config_file = tmp_path / "locked" / ".gloggur.yaml"
    original_makedirs = os.makedirs

    def fail_makedirs(path: str, exist_ok: bool = False) -> None:
        _ = exist_ok
        if path == str(config_file.parent):
            raise PermissionError("permission denied")
        original_makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr("gloggur.cli.main.os.makedirs", fail_makedirs)
    result = runner.invoke(
        cli,
        ["watch", "init", str(repo), "--config", str(config_file), "--json"],
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "write watch config payload"
    assert str(error["path"]) == str(config_file)
    assert "permission denied" in str(error["detail"]).lower()
    assert "Traceback (most recent call last)" not in result.output


def test_watch_init_json_reports_invalid_non_mapping_config_payload(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf8")
    config_file = tmp_path / ".gloggur.yaml"
    config_file.write_text("- item1\n- item2\n", encoding="utf8")

    result = runner.invoke(
        cli,
        ["watch", "init", str(repo), "--config", str(config_file), "--json"],
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "read watch config payload"
    assert str(error["path"]) == str(config_file)
    assert "must be a mapping" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


def test_watch_start_json_reports_structured_log_directory_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="daemon",
        watch_log_file=str(tmp_path / "logs" / "watch.log"),
    )

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {"indexed_files": 0, "indexed_symbols": 0, "skipped_files": 0, "error_count": 0}

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)
    original_makedirs = os.makedirs

    def fail_makedirs(path: str, exist_ok: bool = False) -> None:
        _ = exist_ok
        if path == str(tmp_path / "logs"):
            raise PermissionError("permission denied")
        original_makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr("gloggur.cli.main.os.makedirs", fail_makedirs)
    result = runner.invoke(cli, ["watch", "start", "--daemon", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "prepare watch log directory"
    assert str(error["path"]) == str(tmp_path / "logs")
    assert "permission denied" in str(error["detail"]).lower()
    assert "Traceback (most recent call last)" not in result.output


def test_watch_start_json_reports_structured_daemon_spawn_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="daemon",
        watch_log_file=str(tmp_path / "logs" / "watch.log"),
    )

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {"indexed_files": 0, "indexed_symbols": 0, "skipped_files": 0, "error_count": 0}

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)

    def fail_popen(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        raise OSError("spawn failed")

    monkeypatch.setattr("gloggur.cli.main.subprocess.Popen", fail_popen)
    result = runner.invoke(cli, ["watch", "start", "--daemon", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "spawn watch daemon process"
    assert str(error["path"]) == os.sys.executable
    assert "spawn failed" in str(error["detail"]).lower()
    assert "Traceback (most recent call last)" not in result.output


def test_watch_start_json_reports_structured_daemon_early_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    log_file = tmp_path / "logs" / "watch.log"
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="daemon",
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
        watch_log_file=str(log_file),
    )
    # Simulate stale runtime artifact from a previous crash.
    pid_file.write_text("9999\n", encoding="utf8")

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {"indexed_files": 0, "indexed_symbols": 0, "skipped_files": 0, "error_count": 0}

    class EarlyExitProcess:
        pid = 4321

        @staticmethod
        def poll() -> int:
            return 2

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)
    monkeypatch.setattr("gloggur.cli.main.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "gloggur.cli.main.subprocess.Popen",
        lambda *args, **kwargs: EarlyExitProcess(),
    )
    result = runner.invoke(cli, ["watch", "start", "--daemon", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "verify watch daemon startup"
    assert str(error["path"]) == str(log_file)
    assert "exited early with code 2" in str(error["detail"]).lower()
    assert not pid_file.exists()
    assert state_file.exists()
    state_payload = json.loads(state_file.read_text(encoding="utf8"))
    assert state_payload["running"] is False
    assert state_payload["status"] == "failed_startup"
    assert "exited early with code 2" in str(state_payload.get("last_error", "")).lower()
    assert "Traceback (most recent call last)" not in result.output


def test_watch_start_json_terminates_daemon_when_pid_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    log_file = tmp_path / "logs" / "watch.log"
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="daemon",
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
        watch_log_file=str(log_file),
    )

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {"indexed_files": 0, "indexed_symbols": 0, "skipped_files": 0, "error_count": 0}

    class RunningProcess:
        pid = 6789

        @staticmethod
        def poll() -> None:
            return None

    kill_calls: list[tuple[int, int]] = []

    def fail_write_pid(_path: str, _pid: int) -> None:
        raise StorageIOError(
            category="permission_denied",
            operation="write watch pid file",
            path=str(pid_file),
            probable_cause="The process does not have permission for this filesystem operation.",
            remediation=[
                "Check path ownership and filesystem permissions.",
                "Run with a user that can read/write the cache directory.",
            ],
            detail="PermissionError: permission denied",
        )

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)
    monkeypatch.setattr("gloggur.cli.main.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "gloggur.cli.main.subprocess.Popen",
        lambda *args, **kwargs: RunningProcess(),
    )
    monkeypatch.setattr("gloggur.cli.main._write_pid_file", fail_write_pid)
    monkeypatch.setattr(
        "gloggur.cli.main.os.kill",
        lambda pid, sig: kill_calls.append((pid, sig)),
    )

    result = runner.invoke(cli, ["watch", "start", "--daemon", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "write watch pid file"
    assert str(error["path"]) == str(pid_file)
    assert kill_calls == [(6789, signal.SIGTERM)]
    assert "Traceback (most recent call last)" not in result.output


def test_watch_start_json_escalates_to_sigkill_when_termination_times_out(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    log_file = tmp_path / "logs" / "watch.log"
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="daemon",
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
        watch_log_file=str(log_file),
    )

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {"indexed_files": 0, "indexed_symbols": 0, "skipped_files": 0, "error_count": 0}

    class StuckProcess:
        pid = 6799

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(timeout: float | None = None) -> None:
            raise subprocess.TimeoutExpired(cmd="watch-daemon", timeout=timeout or 0.0)

    kill_calls: list[tuple[int, int]] = []

    def fail_write_pid(_path: str, _pid: int) -> None:
        raise StorageIOError(
            category="permission_denied",
            operation="write watch pid file",
            path=str(pid_file),
            probable_cause="The process does not have permission for this filesystem operation.",
            remediation=[
                "Check path ownership and filesystem permissions.",
                "Run with a user that can read/write the cache directory.",
            ],
            detail="PermissionError: permission denied",
        )

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)
    monkeypatch.setattr("gloggur.cli.main.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "gloggur.cli.main.subprocess.Popen",
        lambda *args, **kwargs: StuckProcess(),
    )
    monkeypatch.setattr("gloggur.cli.main._write_pid_file", fail_write_pid)
    monkeypatch.setattr(
        "gloggur.cli.main.os.kill",
        lambda pid, sig: kill_calls.append((pid, sig)),
    )

    result = runner.invoke(cli, ["watch", "start", "--daemon", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "write watch pid file"
    assert str(error["path"]) == str(pid_file)
    assert kill_calls == [(6799, signal.SIGTERM), (6799, signal.SIGKILL)]
    assert "Traceback (most recent call last)" not in result.output


def test_watch_start_json_removes_pid_file_when_state_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    log_file = tmp_path / "logs" / "watch.log"
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="daemon",
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
        watch_log_file=str(log_file),
    )

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {"indexed_files": 0, "indexed_symbols": 0, "skipped_files": 0, "error_count": 0}

    class RunningProcess:
        pid = 6790

        @staticmethod
        def poll() -> None:
            return None

    kill_calls: list[tuple[int, int]] = []

    def fail_write_state(_path: str, _updates: dict[str, object]) -> None:
        raise StorageIOError(
            category="permission_denied",
            operation="write watch state file",
            path=str(state_file),
            probable_cause="The process does not have permission for this filesystem operation.",
            remediation=[
                "Check path ownership and filesystem permissions.",
                "Run with a user that can read/write the cache directory.",
            ],
            detail="PermissionError: permission denied",
        )

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)
    monkeypatch.setattr("gloggur.cli.main.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "gloggur.cli.main.subprocess.Popen",
        lambda *args, **kwargs: RunningProcess(),
    )
    monkeypatch.setattr("gloggur.cli.main._write_watch_state", fail_write_state)
    monkeypatch.setattr(
        "gloggur.cli.main.os.kill",
        lambda pid, sig: kill_calls.append((pid, sig)),
    )

    result = runner.invoke(cli, ["watch", "start", "--daemon", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "write watch state file"
    assert str(error["path"]) == str(state_file)
    assert kill_calls == [(6790, signal.SIGTERM)]
    assert not pid_file.exists()
    assert "Traceback (most recent call last)" not in result.output


def test_watch_start_json_reports_structured_post_init_daemon_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    log_file = tmp_path / "logs" / "watch.log"
    config = GloggurConfig(
        watch_path=str(repo),
        watch_mode="daemon",
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
        watch_log_file=str(log_file),
    )

    class DummyWatchService:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_forever(self, _path: str) -> dict[str, object]:
            return {"indexed_files": 0, "indexed_symbols": 0, "skipped_files": 0, "error_count": 0}

    class PostInitExitProcess:
        pid = 6791

        def __init__(self) -> None:
            self._poll_calls = 0

        def poll(self) -> int | None:
            self._poll_calls += 1
            if self._poll_calls == 1:
                return None
            return 3

    monkeypatch.setattr(
        "gloggur.cli.main._create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: None)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)
    monkeypatch.setattr(
        "gloggur.cli.main._create_embedding_provider_for_command",
        lambda _cfg: None,
    )
    monkeypatch.setattr("gloggur.cli.main.WatchService", DummyWatchService)
    monkeypatch.setattr("gloggur.cli.main.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "gloggur.cli.main.subprocess.Popen",
        lambda *args, **kwargs: PostInitExitProcess(),
    )

    result = runner.invoke(cli, ["watch", "start", "--daemon", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "verify watch daemon startup"
    assert str(error["path"]) == str(log_file)
    assert "exited early with code 3" in str(error["detail"]).lower()
    assert not pid_file.exists()
    assert state_file.exists()
    state_payload = json.loads(state_file.read_text(encoding="utf8"))
    assert state_payload["running"] is False
    assert state_payload["status"] == "failed_startup"
    assert state_payload["watch_path"] == str(repo)
    assert "Traceback (most recent call last)" not in result.output


def test_watch_stop_json_reports_structured_signal_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(
        watch_pid_file=str(tmp_path / "watch.pid"),
        watch_state_file=str(tmp_path / "watch_state.json"),
    )
    monkeypatch.setattr("gloggur.cli.main._load_config", lambda _path: config)
    monkeypatch.setattr("gloggur.cli.main._read_pid_file", lambda _path: 12345)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: True)

    def fail_kill(pid: int, sig: int) -> None:
        _ = pid, sig
        raise ProcessLookupError("no such process")

    monkeypatch.setattr("gloggur.cli.main.os.kill", fail_kill)
    result = runner.invoke(cli, ["watch", "stop", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "signal watch process"
    assert str(error["path"]) == str(tmp_path / "watch.pid")
    assert "no such process" in str(error["detail"]).lower()
    assert "Traceback (most recent call last)" not in result.output


def test_watch_status_json_reports_malformed_pid_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    pid_file.write_text("not-a-pid\n", encoding="utf8")
    config = GloggurConfig(
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
    )
    monkeypatch.setattr("gloggur.cli.main._load_config", lambda _path: config)

    result = runner.invoke(cli, ["watch", "status", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "read watch pid file"
    assert str(error["path"]) == str(pid_file)
    assert "invalid literal for int" in str(error["detail"]).lower()
    assert "Traceback (most recent call last)" not in result.output


def test_watch_status_json_reports_malformed_state_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    state_file.write_text("{not-json", encoding="utf8")
    config = GloggurConfig(
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
    )
    monkeypatch.setattr("gloggur.cli.main._load_config", lambda _path: config)

    result = runner.invoke(cli, ["watch", "status", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "read watch state file"
    assert str(error["path"]) == str(state_file)
    assert "jsondecodeerror" in str(error["detail"]).lower()
    assert "Traceback (most recent call last)" not in result.output


def test_watch_status_json_synthesizes_inconsistent_failure_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """watch status should fail closed when failure counts lack machine-readable reason codes."""
    runner = CliRunner()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    pid_file.write_text("4242\n", encoding="utf8")
    state_file.write_text(
        json.dumps(
            {
                "status": "running",
                "running": True,
                "failed": 2,
            }
        ),
        encoding="utf8",
    )
    config = GloggurConfig(
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
    )
    monkeypatch.setattr("gloggur.cli.main._load_config", lambda _path: config)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: True)

    result = runner.invoke(cli, ["watch", "status", "--json"])

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["running"] is True
    assert payload["status"] == "running_with_errors"
    reasons = payload["failed_reasons"]
    assert isinstance(reasons, dict)
    assert reasons == {"watch_state_inconsistent": 2}
    failure_codes = payload["failure_codes"]
    assert isinstance(failure_codes, list)
    assert failure_codes == ["watch_state_inconsistent"]
    guidance = payload["failure_guidance"]
    assert isinstance(guidance, dict)
    assert "watch_state_inconsistent" in guidance
    assert isinstance(guidance["watch_state_inconsistent"], list)
    assert guidance["watch_state_inconsistent"]


def test_collect_watch_failure_signals_uses_last_batch_reason_counts_when_top_level_drift() -> None:
    failed_count, reasons = _collect_watch_failure_signals(
        {
            "status": "running",
            "running": True,
            "failed": 0,
            "failed_reasons": {},
            "last_batch": {
                "failed": 1,
                "failed_reasons": {"vector_metadata_mismatch": 1},
            },
        }
    )

    assert failed_count == 1
    assert reasons == {"vector_metadata_mismatch": 1}


def test_collect_watch_failure_signals_uses_last_batch_failure_codes() -> None:
    failed_count, reasons = _collect_watch_failure_signals(
        {
            "status": "running",
            "running": True,
            "failed": 0,
            "failed_reasons": {},
            "last_batch": {
                "failure_codes": ["vector_metadata_mismatch"],
            },
        }
    )

    assert failed_count == 1
    assert reasons == {"vector_metadata_mismatch": 1}


def test_watch_status_json_uses_last_batch_failure_reasons_when_top_level_counters_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """watch status should fail closed using last_batch reasons when top-level counters drift."""
    runner = CliRunner()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    pid_file.write_text("4242\n", encoding="utf8")
    state_file.write_text(
        json.dumps(
            {
                "status": "running",
                "running": True,
                "failed": 0,
                "failed_reasons": {},
                "last_batch": {
                    "failed": 1,
                    "failed_reasons": {"vector_metadata_mismatch": 1},
                },
            }
        ),
        encoding="utf8",
    )
    config = GloggurConfig(
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
    )
    monkeypatch.setattr("gloggur.cli.main._load_config", lambda _path: config)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: True)

    result = runner.invoke(cli, ["watch", "status", "--json"])

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["running"] is True
    assert payload["status"] == "running_with_errors"
    reasons = payload["failed_reasons"]
    assert isinstance(reasons, dict)
    assert reasons == {"vector_metadata_mismatch": 1}
    failure_codes = payload["failure_codes"]
    assert isinstance(failure_codes, list)
    assert failure_codes == ["vector_metadata_mismatch"]
    guidance = payload["failure_guidance"]
    assert isinstance(guidance, dict)
    assert "vector_metadata_mismatch" in guidance
    assert isinstance(guidance["vector_metadata_mismatch"], list)
    assert guidance["vector_metadata_mismatch"]
    last_batch = payload.get("last_batch")
    assert isinstance(last_batch, dict)
    last_batch_codes = last_batch.get("failure_codes")
    assert isinstance(last_batch_codes, list)
    assert last_batch_codes == ["vector_metadata_mismatch"]
    last_batch_guidance = last_batch.get("failure_guidance")
    assert isinstance(last_batch_guidance, dict)
    assert "vector_metadata_mismatch" in last_batch_guidance
    assert isinstance(last_batch_guidance["vector_metadata_mismatch"], list)
    assert last_batch_guidance["vector_metadata_mismatch"]


def test_watch_status_json_synthesizes_last_batch_inconsistent_failure_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """watch status should emit a deterministic code when last_batch lacks reason codes."""
    runner = CliRunner()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    pid_file.write_text("4242\n", encoding="utf8")
    state_file.write_text(
        json.dumps(
            {
                "status": "running",
                "running": True,
                "failed": 0,
                "last_batch": {"failed": 2},
            }
        ),
        encoding="utf8",
    )
    config = GloggurConfig(
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
    )
    monkeypatch.setattr("gloggur.cli.main._load_config", lambda _path: config)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: True)

    result = runner.invoke(cli, ["watch", "status", "--json"])

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["running"] is True
    assert payload["status"] == "running_with_errors"
    reasons = payload["failed_reasons"]
    assert isinstance(reasons, dict)
    assert reasons == {"watch_last_batch_inconsistent": 2}
    failure_codes = payload["failure_codes"]
    assert isinstance(failure_codes, list)
    assert failure_codes == ["watch_last_batch_inconsistent"]
    guidance = payload["failure_guidance"]
    assert isinstance(guidance, dict)
    assert "watch_last_batch_inconsistent" in guidance
    assert isinstance(guidance["watch_last_batch_inconsistent"], list)
    assert guidance["watch_last_batch_inconsistent"]


def test_build_watch_failure_contract_prefers_last_batch_failure_codes_over_inconsistency() -> None:
    contract = _build_watch_failure_contract(
        {
            "status": "running_with_errors",
            "running": True,
            "failed": 0,
            "failed_reasons": {},
            "last_batch": {
                "failure_codes": ["vector_metadata_mismatch"],
            },
        }
    )

    assert contract["failed_reasons"] == {"vector_metadata_mismatch": 1}
    assert contract["failure_codes"] == ["vector_metadata_mismatch"]
    guidance = contract["failure_guidance"]
    assert isinstance(guidance, dict)
    assert "vector_metadata_mismatch" in guidance
    assert isinstance(guidance["vector_metadata_mismatch"], list)
    assert guidance["vector_metadata_mismatch"]
    last_batch = contract.get("last_batch")
    assert isinstance(last_batch, dict)
    assert last_batch["failed_reasons"] == {"vector_metadata_mismatch": 1}
    assert last_batch["failure_codes"] == ["vector_metadata_mismatch"]
    assert last_batch["failed"] == 1
    assert last_batch["error_count"] == 1


def test_watch_status_normalizes_stale_running_state_when_process_is_dead(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()
    pid_file = tmp_path / "watch.pid"
    state_file = tmp_path / "watch_state.json"
    pid_file.write_text("4242\n", encoding="utf8")
    state_file.write_text(
        json.dumps(
            {
                "status": "running",
                "running": True,
                "last_heartbeat": "2026-02-26T18:00:00+00:00",
            }
        ),
        encoding="utf8",
    )
    config = GloggurConfig(
        watch_pid_file=str(pid_file),
        watch_state_file=str(state_file),
    )
    monkeypatch.setattr("gloggur.cli.main._load_config", lambda _path: config)
    monkeypatch.setattr("gloggur.cli.main.is_process_running", lambda _pid: False)

    result = runner.invoke(cli, ["watch", "status", "--json"])

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["running"] is False
    assert payload["status"] == "stopped"
    assert payload["last_heartbeat"] == "2026-02-26T18:00:00+00:00"


def test_is_process_running_treats_permission_error_as_alive(monkeypatch) -> None:
    def permission_denied(_pid: int, _sig: int) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr("gloggur.watch.service.os.kill", permission_denied)
    assert is_process_running(1234) is True
