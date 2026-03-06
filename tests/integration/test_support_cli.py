from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from gloggur.cli.main import cli
from scripts.verification.fixtures import TestFixtures


def _parse_json_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    assert isinstance(payload, dict)
    return payload


def _test_env(cache_dir: Path) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    existing = os.environ.get("PYTHONPATH")
    pythonpath = str(src_root) if not existing else f"{src_root}{os.pathsep}{existing}"
    return {
        "GLOGGUR_CACHE_DIR": str(cache_dir),
        "GLOGGUR_EMBEDDING_PROVIDER": "test",
        "PYTHONPATH": pythonpath,
    }


def test_support_run_search_json_creates_session_and_preserves_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
        (repo / ".git").mkdir(exist_ok=True)
        cache_dir = tmp_path / "cache"
        env = _test_env(cache_dir)
        monkeypatch.chdir(repo)

        index_result = runner.invoke(cli, ["index", ".", "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(
            cli,
            ["support", "run", "--json", "--", "search", "add", "--json", "--top-k", "3"],
            env=env,
        )

        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        assert payload["child_exit_code"] == 0
        session_dir = Path(str(payload["session_dir"]))
        assert session_dir.exists()
        assert (session_dir / "diagnostics" / "child_payload.json").exists()
        assert (session_dir / "logs" / "child.stdout.log").exists()


def test_support_run_failure_auto_creates_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
        (repo / ".git").mkdir(exist_ok=True)
        env = _test_env(tmp_path / "cache")
        monkeypatch.chdir(repo)

        result = runner.invoke(
            cli,
            ["support", "run", "--json", "--", "search", "add", "--json", "--top-k", "0"],
            env=env,
        )

        assert result.exit_code != 0
        payload = _parse_json_output(result.output)
        assert payload["bundle_created"] is True
        bundle_path = Path(str(payload["bundle_path"]))
        assert bundle_path.exists()
        with tarfile.open(bundle_path, "r:gz") as archive:
            assert "manifest.json" in archive.getnames()


def test_support_collect_manual_snapshot_includes_runtime_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
        (repo / ".git").mkdir(exist_ok=True)
        cache_dir = tmp_path / "cache"
        env = _test_env(cache_dir)
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        bootstrap_log = runtime_dir / "bootstrap.log"
        bootstrap_state = runtime_dir / "bootstrap.state.json"
        bootstrap_log.write_text("bootstrap ready\n", encoding="utf8")
        bootstrap_state.write_text('{"ok":true}\n', encoding="utf8")
        env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(bootstrap_log)
        env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(bootstrap_state)
        env["GLOGGUR_WATCH_LOG_FILE"] = str(runtime_dir / "watch.log")
        env["GLOGGUR_WATCH_STATE_FILE"] = str(runtime_dir / "watch_state.json")
        env["GLOGGUR_WATCH_PID_FILE"] = str(runtime_dir / "watch.pid")
        Path(env["GLOGGUR_WATCH_LOG_FILE"]).write_text("watch output\n", encoding="utf8")
        Path(env["GLOGGUR_WATCH_STATE_FILE"]).write_text('{"status":"running"}\n', encoding="utf8")
        Path(env["GLOGGUR_WATCH_PID_FILE"]).write_text("123\n", encoding="utf8")
        (repo / ".gloggur" / "logs").mkdir(parents=True, exist_ok=True)
        (repo / ".gloggur" / "logs" / "search_router.jsonl").write_text(
            '{"event":"router"}\n',
            encoding="utf8",
        )
        monkeypatch.chdir(repo)

        index_result = runner.invoke(cli, ["index", ".", "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(cli, ["support", "collect", "--json"], env=env)

        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        bundle_path = Path(str(payload["bundle_path"]))
        with tarfile.open(bundle_path, "r:gz") as archive:
            names = set(archive.getnames())
        session_id = str(payload["session_id"])
        assert f"support/{session_id}/diagnostics/status.json" in names
        assert f"support/{session_id}/logs/watch.log" in names
        assert f"support/{session_id}/logs/bootstrap.log" in names
        assert f"support/{session_id}/logs/search_router.jsonl" in names


def test_support_collect_include_cache_adds_cache_and_symbol_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
        (repo / ".git").mkdir(exist_ok=True)
        cache_dir = tmp_path / "cache"
        env = _test_env(cache_dir)
        monkeypatch.chdir(repo)

        index_result = runner.invoke(cli, ["index", ".", "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(cli, ["support", "collect", "--json", "--include-cache"], env=env)

        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        bundle_path = Path(str(payload["bundle_path"]))
        with tarfile.open(bundle_path, "r:gz") as archive:
            names = set(archive.getnames())
        session_id = str(payload["session_id"])
        assert f"support/{session_id}/artifacts/.gloggur-cache/index.db" in names
        assert f"support/{session_id}/artifacts/.gloggur/index/symbols.db" in names


def test_support_collect_destination_collision_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
        (repo / ".git").mkdir(exist_ok=True)
        env = _test_env(tmp_path / "cache")
        monkeypatch.chdir(repo)

        destination = tmp_path / "bundle.tar.gz"
        first = runner.invoke(
            cli,
            ["support", "collect", "--json", "--destination", str(destination)],
            env=env,
        )
        assert first.exit_code == 0, first.output

        second = runner.invoke(
            cli,
            ["support", "collect", "--json", "--destination", str(destination)],
            env=env,
        )

        assert second.exit_code != 0
        payload = _parse_json_output(second.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["code"] == "support_destination_exists"
        assert payload["failure_codes"] == ["support_destination_exists"]


def test_support_collect_missing_session_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
        (repo / ".git").mkdir(exist_ok=True)
        env = _test_env(tmp_path / "cache")
        monkeypatch.chdir(repo)

        result = runner.invoke(
            cli,
            ["support", "collect", "--json", "--session", "missing-session"],
            env=env,
        )

        assert result.exit_code != 0
        payload = _parse_json_output(result.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["code"] == "support_session_missing"
        assert payload["failure_codes"] == ["support_session_missing"]
