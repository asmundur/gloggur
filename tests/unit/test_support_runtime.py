from __future__ import annotations

import json
from pathlib import Path

from gloggur.support_runtime import (
    CommandTraceSession,
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


def test_command_trace_session_moves_completed_command_to_recent(tmp_path: Path, monkeypatch) -> None:
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
