from __future__ import annotations

import errno
import json
import threading
import time
from pathlib import Path

import gloggur.support_runtime as support_runtime_module
from gloggur.support_runtime import (
    CommandTraceSession,
    SUPPORT_RUNTIME_DEGRADED_WARNING_CODE,
    load_support_runtime_config,
    support_runtime_paths,
    write_support_runtime_config,
)


def test_write_support_runtime_config_creates_support_section(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    config_path = write_support_runtime_config(repo, enabled=True)

    assert config_path == repo / ".gloggur" / "config.toml"
    text = config_path.read_text(encoding="utf8")
    assert "[support]" in text
    assert "enabled = true" in text
    config = load_support_runtime_config(repo)
    assert config.enabled is True


def test_write_support_runtime_config_preserves_existing_sections(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    config_path = repo / ".gloggur" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "[search_router]\ntelemetry_enabled = true\n",
        encoding="utf8",
    )

    write_support_runtime_config(repo, enabled=False)

    text = config_path.read_text(encoding="utf8")
    assert "[search_router]" in text
    assert "[support]" in text
    assert "enabled = false" in text


def test_command_trace_session_moves_completed_command_to_recent(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    write_support_runtime_config(repo, enabled=True)
    monkeypatch.chdir(repo)
    config = load_support_runtime_config(repo)

    session = CommandTraceSession(
        repo_root=repo,
        command_name="index",
        argv=["index", ".", "--json"],
        config=config,
    )
    with session:
        session.update_stage("extract_symbols", build_id="build-1")
        session.update_build_state(
            {
                "state": "building",
                "build_id": "build-1",
                "stage": "extract_symbols",
            }
        )
        print("hello from trace")
        session.set_exit_code(0)

    paths = support_runtime_paths(repo)
    active_dirs = list(paths.active_root.glob("*"))
    recent_dirs = list(paths.recent_root.glob("*"))
    assert active_dirs == []
    assert len(recent_dirs) == 1
    recent_dir = recent_dirs[0]
    payload = json.loads((recent_dir / "meta.json").read_text(encoding="utf8"))
    assert payload["command_name"] == "index"
    assert payload["status"] == "completed"
    assert payload["stage"] == "extract_symbols"
    assert payload["build_state"]["build_id"] == "build-1"
    assert "hello from trace" in (recent_dir / "stdout.log").read_text(encoding="utf8")


def test_atomic_write_json_uses_unique_temp_paths_under_concurrent_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "meta.json"
    original_replace = Path.replace
    inflight_temp_paths: set[str] = set()
    replace_lock = threading.Lock()

    def _replace_with_collision_guard(self: Path, destination: Path) -> Path:
        key = str(self)
        with replace_lock:
            if key in inflight_temp_paths:
                raise FileNotFoundError(
                    errno.ENOENT,
                    "No such file or directory",
                    str(self),
                    str(destination),
                )
            inflight_temp_paths.add(key)
        try:
            time.sleep(0.02)
            return original_replace(self, destination)
        finally:
            with replace_lock:
                inflight_temp_paths.discard(key)

    monkeypatch.setattr(Path, "replace", _replace_with_collision_guard)

    barrier = threading.Barrier(2)
    failures: list[Exception] = []

    def _writer(value: int) -> None:
        barrier.wait(timeout=2.0)
        try:
            support_runtime_module._atomic_write_json(target, {"value": value})
        except Exception as exc:  # pragma: no cover - collected for assertion
            failures.append(exc)

    left = threading.Thread(target=_writer, args=(1,))
    right = threading.Thread(target=_writer, args=(2,))
    left.start()
    right.start()
    left.join(timeout=2.0)
    right.join(timeout=2.0)

    assert failures == []
    payload = json.loads(target.read_text(encoding="utf8"))
    assert payload["value"] in {1, 2}


def test_command_trace_session_metadata_write_failure_degrades_without_raising(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    write_support_runtime_config(repo, enabled=True)
    monkeypatch.chdir(repo)
    config = load_support_runtime_config(repo)

    def _raise_metadata_write_failure(path: Path, payload: dict[str, object]) -> None:
        _ = payload
        raise FileNotFoundError(
            errno.ENOENT,
            "No such file or directory",
            f"{path}.tmp",
            str(path),
        )

    monkeypatch.setattr(support_runtime_module, "_atomic_write_json", _raise_metadata_write_failure)

    session = CommandTraceSession(
        repo_root=repo,
        command_name="index",
        argv=["index", ".", "--json"],
        config=config,
    )
    with session:
        session.update_stage("update_symbol_index", build_id="build-1")
        session.update_build_state(
            {
                "state": "building",
                "build_id": "build-1",
                "stage": "update_symbol_index",
            }
        )
        session.set_exit_code(0)
        print("trace still runs")

    captured = capsys.readouterr()
    assert captured.err.count("support runtime tracing degraded") == 1
    assert "FileNotFoundError" in captured.err
    assert session.warning_codes() == [SUPPORT_RUNTIME_DEGRADED_WARNING_CODE]
