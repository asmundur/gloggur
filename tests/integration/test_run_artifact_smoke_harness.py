from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_artifact_smoke(
    args: list[str],
    *,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "scripts/run_artifact_smoke.py", "--format", "json", *args]
    return subprocess.run(
        command,
        cwd=str(repo_root),
        env={**os.environ, "GLOGGUR_LOCAL_FALLBACK": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_run_artifact_smoke_harness_publish_restore_passes() -> None:
    completed = _run_artifact_smoke([])
    assert completed.returncode == 0, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["ok"] is True
    assert payload["stage_order"] == [
        "index_source",
        "publish_artifact",
        "validate_artifact",
        "restore_artifact",
        "restored_status",
        "restored_search",
    ]
    stages = payload.get("stages", [])
    assert isinstance(stages, list)
    assert [stage.get("status") for stage in stages] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
    ]
    assert payload["summary"] == {
        "total_stages": 6,
        "passed": 6,
        "failed": 0,
        "not_run": 0,
    }


def test_run_artifact_smoke_harness_reports_stage_code_and_order_on_failure(
    tmp_path: Path,
) -> None:
    missing_repo = tmp_path / "missing-repo"
    completed = _run_artifact_smoke(["--repo", str(missing_repo)])
    assert completed.returncode == 1, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["ok"] is False
    assert payload["failure"]["stage"] == "index_source"
    assert payload["failure"]["code"] == "artifact_smoke_index_failed"
    assert payload["stage_order"] == [
        "index_source",
        "publish_artifact",
        "validate_artifact",
        "restore_artifact",
        "restored_status",
        "restored_search",
    ]
    stages = payload.get("stages", [])
    assert isinstance(stages, list)
    assert [stage.get("status") for stage in stages] == [
        "failed",
        "not_run",
        "not_run",
        "not_run",
        "not_run",
        "not_run",
    ]
