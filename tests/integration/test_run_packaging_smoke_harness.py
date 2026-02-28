from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_packaging(args: list[str], *, timeout: float = 240.0) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "scripts/run_packaging_smoke.py", "--format", "json", *args]
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


def test_run_packaging_smoke_reports_stage_code_and_order_on_missing_repo(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing-repo"
    completed = _run_packaging(["--repo", str(missing_repo)])
    assert completed.returncode == 1, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["ok"] is False
    assert payload["failure"]["stage"] == "build_artifacts"
    assert payload["failure"]["code"] == "packaging_build_failed"
    stages = payload.get("stages", [])
    assert isinstance(stages, list)
    assert stages[0]["name"] == "build_artifacts"
    assert stages[0]["status"] == "failed"


@pytest.mark.skipif(
    importlib.util.find_spec("build") is None,
    reason="python-build package is required for packaging smoke",
)
def test_run_packaging_smoke_full_install_upgrade_mode_passes() -> None:
    completed = _run_packaging([])
    assert completed.returncode == 0, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["ok"] is True
    assert payload["skip_install_smoke"] is False
    assert payload["stage_order"] == [
        "build_artifacts",
        "install_from_sdist",
        "upgrade_to_wheel",
        "cli_help",
        "cli_status",
    ]
    summary = payload.get("summary", {})
    assert summary == {"total_stages": 5, "passed": 5, "failed": 0, "not_run": 0}
    stages = payload.get("stages", [])
    assert isinstance(stages, list)
    install_stage = next(stage for stage in stages if stage["name"] == "install_from_sdist")
    assert install_stage["status"] == "passed"
    install_context = install_stage["context"]
    assert isinstance(install_context, dict)
    module_path = install_context["installed_module_path"]
    assert isinstance(module_path, str)
    assert "/site-packages/" in module_path
