from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import textwrap
import time
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


def _init_betatester_support(runner: CliRunner, repo: Path, env: dict[str, str]) -> None:
    result = runner.invoke(cli, ["init", str(repo), "--betatester-support", "--json"], env=env)
    assert result.exit_code == 0, result.output


def test_support_run_search_json_creates_session_and_preserves_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
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
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
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
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
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
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
        (repo / ".git").mkdir(exist_ok=True)
        cache_dir = tmp_path / "cache"
        env = _test_env(cache_dir)
        monkeypatch.chdir(repo)

        index_result = runner.invoke(cli, ["index", ".", "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(cli, ["support", "collect", "--json", "--include-cache"], env=env)

        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        assert payload["bundle_sensitivity"] == "sensitive"
        assert payload["includes_cache_artifacts"] is True
        assert payload["sensitive_data_acknowledged"] is False
        bundle_path = Path(str(payload["bundle_path"]))
        with tarfile.open(bundle_path, "r:gz") as archive:
            names = set(archive.getnames())
            manifest_member = archive.extractfile("manifest.json")
            assert manifest_member is not None
            manifest = json.loads(manifest_member.read().decode("utf8"))
        session_id = str(payload["session_id"])
        assert f"support/{session_id}/artifacts/.gloggur-cache/index.db" in names
        assert f"support/{session_id}/artifacts/.gloggur/index/symbols.db" in names
        assert manifest["bundle_sensitivity"] == "sensitive"
        assert manifest["includes_cache_artifacts"] is True
        assert manifest["sensitive_data_acknowledged"] is False


def test_support_collect_allow_sensitive_data_sets_acknowledgement_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
        (repo / ".git").mkdir(exist_ok=True)
        cache_dir = tmp_path / "cache"
        env = _test_env(cache_dir)
        monkeypatch.chdir(repo)

        index_result = runner.invoke(cli, ["index", ".", "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(
            cli,
            ["support", "collect", "--json", "--include-cache", "--allow-sensitive-data"],
            env=env,
        )

        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        assert payload["bundle_sensitivity"] == "sensitive"
        assert payload["includes_cache_artifacts"] is True
        assert payload["sensitive_data_acknowledged"] is True


def test_support_collect_destination_collision_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
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
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
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


def test_support_collect_degrades_gracefully_without_betatester_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
        (repo / ".git").mkdir(exist_ok=True)
        env = _test_env(tmp_path / "cache")
        monkeypatch.chdir(repo)

        result = runner.invoke(cli, ["support", "collect", "--json"], env=env)

        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        assert payload["support_mode_enabled"] is False
        assert "betatester_support_not_enabled" in payload["capture_warnings"]


def test_support_collect_bundles_recent_runtime_traces_after_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
        (repo / ".git").mkdir(exist_ok=True)
        env = _test_env(tmp_path / "cache")
        monkeypatch.chdir(repo)
        _init_betatester_support(runner, repo, env)

        index_result = runner.invoke(cli, ["index", ".", "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output
        status_result = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_result.exit_code == 0, status_result.output

        collect_result = runner.invoke(cli, ["support", "collect", "--json"], env=env)

        assert collect_result.exit_code == 0, collect_result.output
        payload = _parse_json_output(collect_result.output)
        assert payload["support_mode_enabled"] is True
        assert payload["active_commands"] == []
        bundle_path = Path(str(payload["bundle_path"]))
        with tarfile.open(bundle_path, "r:gz") as archive:
            names = set(archive.getnames())
            session_id = str(payload["session_id"])
            runtime_entries = [
                name
                for name in names
                if name.startswith(f"support/{session_id}/runtime/recent/")
                and name.endswith("/meta.json")
            ]
            assert runtime_entries
            meta_member = archive.extractfile(sorted(runtime_entries)[-1])
            assert meta_member is not None
            meta_payload = json.loads(meta_member.read().decode("utf8"))
        assert meta_payload["command_name"] in {"index", "status"}


@pytest.mark.skipif(os.name != "posix", reason="live stack-dump signaling requires POSIX")
def test_support_collect_captures_active_index_trace_and_stack_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {"sample.py": "def add(a: int, b: int) -> int:\n    return a + b\n"}
        )
        (repo / ".git").mkdir(exist_ok=True)
        env = _test_env(tmp_path / "cache")
        monkeypatch.chdir(repo)
        _init_betatester_support(runner, repo, env)

        script = textwrap.dedent("""
            import json
            import os
            import time
            from pathlib import Path

            from gloggur.support_runtime import CommandTraceSession, load_support_runtime_config

            repo = Path(os.environ["TEST_REPO"]).resolve()
            os.chdir(repo)
            config = load_support_runtime_config(repo)
            session = CommandTraceSession(
                repo_root=repo,
                command_name="index",
                argv=["index", ".", "--json"],
                config=config,
            )
            with session:
                session.update_stage("extract_symbols", build_id="build-live")
                session.update_build_state(
                    {
                        "state": "building",
                        "build_id": "build-live",
                        "pid": os.getpid(),
                        "stage": "extract_symbols",
                        "progress": {
                            "current_file": "sample.py",
                            "subphase": "prepare_file",
                            "files_done": 0,
                            "files_total": 1,
                            "started_at": "2026-03-14T00:00:00+00:00",
                            "updated_at": "2026-03-14T00:00:01+00:00",
                        },
                    }
                )
                print("index still running")
                while True:
                    time.sleep(0.5)
            """)
        proc_env = dict(env)
        proc_env["TEST_REPO"] = str(repo)
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=str(repo),
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            active_root = repo / ".gloggur" / "support" / "runtime" / "active"
            deadline = time.time() + 10
            active_dirs: list[Path] = []
            while time.time() < deadline:
                active_dirs = list(active_root.glob("*"))
                if active_dirs:
                    break
                time.sleep(0.1)
            assert active_dirs, "timed out waiting for active runtime trace"

            collect_result = runner.invoke(cli, ["support", "collect", "--json"], env=env)

            assert collect_result.exit_code == 0, collect_result.output
            payload = _parse_json_output(collect_result.output)
            assert payload["support_mode_enabled"] is True
            assert payload["includes_cache_artifacts"] is True
            assert payload["bundle_policy_applied"] == "smart"
            assert payload["active_commands"]
            bundle_path = Path(str(payload["bundle_path"]))
            with tarfile.open(bundle_path, "r:gz") as archive:
                names = set(archive.getnames())
                session_id = str(payload["session_id"])
                active_meta = f"support/{session_id}/runtime/active/{active_dirs[0].name}/meta.json"
                active_stack = (
                    f"support/{session_id}/runtime/active/{active_dirs[0].name}/stackdump.log"
                )
                assert active_meta in names
                assert active_stack in names
                meta_member = archive.extractfile(active_meta)
                assert meta_member is not None
                meta_payload = json.loads(meta_member.read().decode("utf8"))
                stack_member = archive.extractfile(active_stack)
                assert stack_member is not None
                stack_text = stack_member.read().decode("utf8")
            assert meta_payload["command_name"] == "index"
            assert meta_payload["stage"] == "extract_symbols"
            assert meta_payload["build_state"]["build_id"] == "build-live"
            assert meta_payload["build_state"]["progress"]["current_file"] == "sample.py"
            assert "Thread" in stack_text or "Current thread" in stack_text
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
