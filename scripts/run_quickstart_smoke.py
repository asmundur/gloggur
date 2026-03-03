from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Stage:
    name: str
    args: List[str]
    failure_code: str
    remediation: str
    validate_json: bool = True


STAGES: List[Stage] = [
    Stage(
        name="index",
        args=["index", ".", "--json"],
        failure_code="quickstart_index_failed",
        remediation="Inspect index JSON failure payload and fix config/provider issues before retrying.",
    ),
    Stage(
        name="watch_init",
        args=["watch", "init", ".", "--json"],
        failure_code="quickstart_watch_init_failed",
        remediation="Run from a valid repository path and ensure config is writable.",
    ),
    Stage(
        name="watch_start",
        args=["watch", "start", "--daemon", "--json"],
        failure_code="quickstart_watch_start_failed",
        remediation="Inspect watch startup diagnostics and resolve mode/path/provider problems.",
    ),
    Stage(
        name="watch_status",
        args=["watch", "status", "--json"],
        failure_code="quickstart_watch_status_failed",
        remediation="Ensure watch daemon started and state files are readable.",
    ),
    Stage(
        name="search",
        args=["search", "add numbers token", "--top-k", "5", "--json"],
        failure_code="quickstart_search_failed",
        remediation="Re-index the repository and retry with a query that exists in source symbols.",
    ),
    Stage(
        name="inspect",
        args=["inspect", ".", "--json"],
        failure_code="quickstart_inspect_failed",
        remediation="Inspect inspect JSON output and resolve parser/index errors.",
    ),
]


def _python_executable(repo_root: Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _base_env(repo_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    source_root = repo_root / "src"
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(source_root) if not existing_path else f"{source_root}{os.pathsep}{existing_path}"
    )
    env.setdefault("GLOGGUR_EMBEDDING_PROVIDER", "test")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _create_fixture_repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "math_utils.py").write_text(
        "def add_numbers(value_a: int, value_b: int) -> int:\n"
        "    \"\"\"Add two integers and return the sum.\"\"\"\n"
        "    return value_a + value_b\n",
        encoding="utf8",
    )


def _truncate(value: str, limit: int = 800) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _parse_json_output(stdout: str, stderr: str) -> Optional[Dict[str, object]]:
    for blob in (stdout, stderr):
        if not blob.strip():
            continue
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _run_stage(
    repo_root: Path,
    workspace_repo: Path,
    stage: Stage,
    *,
    timeout_seconds: float,
) -> Dict[str, object]:
    python_exec = _python_executable(repo_root)
    env = _base_env(repo_root)
    command = [python_exec, "-m", "gloggur.cli.main", *stage.args]

    started_at = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=str(workspace_repo),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration_ms = int((time.monotonic() - started_at) * 1000)

    payload = _parse_json_output(completed.stdout, completed.stderr)
    stage_payload: Dict[str, object] = {
        "name": stage.name,
        "command": command,
        "returncode": completed.returncode,
        "duration_ms": duration_ms,
        "stdout": _truncate(completed.stdout),
        "stderr": _truncate(completed.stderr),
        "payload": payload,
        "ok": completed.returncode == 0,
    }

    if stage.validate_json and payload is None:
        stage_payload["ok"] = False
        stage_payload["failure_code"] = f"{stage.failure_code}_output_invalid"

    if stage.name == "watch_start" and completed.returncode == 0:
        if not isinstance(payload, dict) or not bool(payload.get("started")):
            stage_payload["ok"] = False
            stage_payload["failure_code"] = f"{stage.failure_code}_contract_invalid"

    if stage.name == "watch_status" and completed.returncode == 0:
        if not isinstance(payload, dict) or not isinstance(payload.get("running"), bool):
            stage_payload["ok"] = False
            stage_payload["failure_code"] = f"{stage.failure_code}_contract_invalid"

    if stage.name == "search" and completed.returncode == 0:
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            stage_payload["ok"] = False
            stage_payload["failure_code"] = f"{stage.failure_code}_contract_invalid"
        elif len(payload["results"]) == 0:
            stage_payload["ok"] = False
            stage_payload["failure_code"] = f"{stage.failure_code}_no_results"

    return stage_payload


def _watch_stop_cleanup(repo_root: Path, workspace_repo: Path, timeout_seconds: float) -> Dict[str, object]:
    python_exec = _python_executable(repo_root)
    env = _base_env(repo_root)
    command = [python_exec, "-m", "gloggur.cli.main", "watch", "stop", "--json"]
    started_at = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=str(workspace_repo),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration_ms = int((time.monotonic() - started_at) * 1000)
    payload = _parse_json_output(completed.stdout, completed.stderr)
    ok = completed.returncode == 0
    if not isinstance(payload, dict):
        ok = False
    return {
        "name": "watch_stop",
        "command": command,
        "returncode": completed.returncode,
        "duration_ms": duration_ms,
        "stdout": _truncate(completed.stdout),
        "stderr": _truncate(completed.stderr),
        "payload": payload,
        "ok": ok,
        "failure_code": None if ok else "quickstart_watch_stop_failed",
    }


def run_quickstart_smoke(
    repo: Optional[Path],
    *,
    timeout_seconds: float,
    keep_artifacts: bool,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]

    created_fixture_root: Optional[Path] = None
    workspace_repo: Path
    if repo is None:
        created_fixture_root = Path(tempfile.mkdtemp(prefix="gloggur-quickstart-smoke-"))
        workspace_repo = created_fixture_root
        _create_fixture_repo(workspace_repo)
    else:
        workspace_repo = repo

    if not workspace_repo.exists():
        return {
            "ok": False,
            "failure": {
                "code": "quickstart_repo_missing",
                "detail": f"repository path does not exist: {workspace_repo}",
                "remediation": "Pass --repo pointing to an existing repository path.",
            },
            "stages": [],
        }

    if not workspace_repo.is_dir():
        return {
            "ok": False,
            "failure": {
                "code": "quickstart_repo_not_directory",
                "detail": f"repository path is not a directory: {workspace_repo}",
                "remediation": "Pass --repo pointing to a directory.",
            },
            "stages": [],
        }

    stage_results: List[Dict[str, object]] = []
    watch_started = False
    try:
        for stage in STAGES:
            stage_result = _run_stage(
                repo_root,
                workspace_repo,
                stage,
                timeout_seconds=timeout_seconds,
            )
            stage_results.append(stage_result)
            if stage.name == "watch_start" and bool(stage_result.get("ok")):
                watch_started = True

            if not bool(stage_result.get("ok")):
                failure_code = str(stage_result.get("failure_code") or stage.failure_code)
                return {
                    "ok": False,
                    "failure": {
                        "code": failure_code,
                        "detail": (
                            f"Quickstart stage '{stage.name}' failed (exit={stage_result.get('returncode')})."
                        ),
                        "remediation": stage.remediation,
                    },
                    "stages": stage_results,
                }

        if watch_started:
            stop_result = _watch_stop_cleanup(repo_root, workspace_repo, timeout_seconds)
            stage_results.append(stop_result)
            if not bool(stop_result.get("ok")):
                return {
                    "ok": False,
                    "failure": {
                        "code": "quickstart_watch_stop_failed",
                        "detail": "Watch cleanup failed after quickstart sequence.",
                        "remediation": "Run `gloggur watch stop --json` and inspect watch state/log files.",
                    },
                    "stages": stage_results,
                }

        return {
            "ok": True,
            "summary": {
                "repo": str(workspace_repo),
                "created_fixture": created_fixture_root is not None,
                "total_stages": len(stage_results),
                "stage_names": [str(stage.get("name")) for stage in stage_results],
            },
            "stages": stage_results,
        }
    finally:
        if created_fixture_root is not None and not keep_artifacts:
            shutil.rmtree(created_fixture_root, ignore_errors=True)


def _render_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = ["# Quickstart Smoke", ""]
    lines.append(f"- ok: `{payload.get('ok')}`")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        lines.append(f"- repo: `{summary.get('repo')}`")
        lines.append(f"- stage_names: `{summary.get('stage_names')}`")
    failure = payload.get("failure")
    if isinstance(failure, dict):
        lines.append(f"- failure.code: `{failure.get('code')}`")
        lines.append(f"- failure.detail: `{failure.get('detail')}`")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic quickstart smoke harness.")
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--keep-artifacts", action="store_true", default=False)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = run_quickstart_smoke(
        args.repo,
        timeout_seconds=args.timeout_seconds,
        keep_artifacts=args.keep_artifacts,
    )

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload), end="")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
