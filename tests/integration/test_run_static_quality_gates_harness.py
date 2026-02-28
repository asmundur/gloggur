from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_static_quality_gates(
    args: list[str], *, timeout: float = 120.0
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "scripts/run_static_quality_gates.py", "--format", "json", *args]
    return subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_run_static_quality_gates_harness_succeeds_for_verification_surface() -> None:
    completed = _run_static_quality_gates([])

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["summary"] == {
        "total_stages": 3,
        "passed": 3,
        "failed": 0,
        "not_run": 0,
    }
    stages = payload["stages"]
    assert isinstance(stages, list)
    assert [stage["name"] for stage in stages] == ["ruff", "mypy", "black"]
    assert all(stage["status"] == "passed" for stage in stages)
