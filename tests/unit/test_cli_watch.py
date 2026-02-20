from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import yaml

from gloggur.cli.main import cli
from gloggur.config import GloggurConfig


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
    assert result.exit_code != 0
    assert "Use only one of --foreground or --daemon." in result.output


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
    assert result.exit_code != 0
    assert "Unsupported watch mode: invalid-mode" in result.output


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

    state = {"running": False}

    def fake_running(pid: int | None) -> bool:
        return bool(pid == 4321 and state["running"])

    def fake_popen(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
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
