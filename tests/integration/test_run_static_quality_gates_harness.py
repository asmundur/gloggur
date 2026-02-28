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
    assert payload["target_scope"] == [
        "scripts/audit_verification_lanes.py",
        "scripts/check_error_catalog_contract.py",
        "scripts/run_static_quality_gates.py",
        "tests/unit/test_audit_verification_lanes.py",
        "tests/unit/test_verification_workflow.py",
        "tests/unit/test_run_static_quality_gates.py",
        "src/gloggur",
    ]
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
    assert stages[0]["command"][-1] == "src/gloggur"
    assert "src/gloggur" not in stages[1]["command"]
    assert stages[2]["command"][-1] == "src/gloggur"
