from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from gloggur.cli.main import cli


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

    result = runner.invoke(
        cli,
        ["watch", "init", str(repo), "--config", str(config_file), "--json"],
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["initialized"] is True
    assert config_file.exists()
    config_text = config_file.read_text(encoding="utf8")
    assert "watch_enabled: true" in config_text
    assert "watch_mode: daemon" in config_text


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

    status_running = runner.invoke(
        cli,
        ["watch", "status", "--config", str(config_file), "--json"],
    )
    assert status_running.exit_code == 0
    status_payload = _parse_json_output(status_running.output)
    assert status_payload["running"] is True

    stopped = runner.invoke(
        cli,
        ["watch", "stop", "--config", str(config_file), "--json"],
    )
    assert stopped.exit_code == 0
    stopped_payload = _parse_json_output(stopped.output)
    assert stopped_payload["stopped"] is True
