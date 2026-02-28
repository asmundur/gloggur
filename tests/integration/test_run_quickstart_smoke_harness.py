from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_quickstart(args: list[str], *, timeout: float = 240.0) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "scripts/run_quickstart_smoke.py", "--format", "json", *args]
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def test_run_quickstart_smoke_harness_passes_on_fixture_repo() -> None:
    completed = _run_quickstart([])

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    summary = payload["summary"]
    assert summary["created_fixture"] is True
    stage_names = summary["stage_names"]
    assert stage_names == [
        "index",
        "watch_init",
        "watch_start",
        "watch_status",
        "search",
        "inspect",
        "watch_stop",
    ]


def test_run_quickstart_smoke_reports_missing_repo_failure_code(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing-repo"

    completed = _run_quickstart(["--repo", str(missing_repo)])

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["failure"]["code"] == "quickstart_repo_missing"
    assert payload["stages"] == []
