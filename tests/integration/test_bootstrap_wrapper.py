from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _wrapper_path() -> Path:
    return _repo_root() / "scripts" / "gloggur"


def _run_wrapper(
    args: list[str],
    env: dict[str, str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(_wrapper_path()), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(cwd or _repo_root()),
        check=False,
    )


def _parse_json_stdout(output: str) -> dict[str, object]:
    return json.loads(output.strip())


def test_wrapper_fallbacks_to_system_python_in_dry_run_mode() -> None:
    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_DRY_RUN"] = "1"
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = str(_repo_root() / ".missing-venv" / "bin" / "python")
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env.pop("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", None)

    completed = _run_wrapper(["status", "--json"], env=env)

    assert completed.returncode == 0, completed.stderr
    payload = _parse_json_stdout(completed.stdout)
    assert payload["operation"] == "preflight"
    assert payload["ready"] is True
    assert payload["selected_candidate"] == "system"
    assert payload["selected_module"] == "gloggur"


def test_wrapper_returns_deterministic_json_for_missing_package() -> None:
    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_DRY_RUN"] = "1"
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = str(_repo_root() / ".missing-venv" / "bin" / "python")
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env["GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS"] = "definitely_missing_pkg_for_gloggur_bootstrap_test"

    completed = _run_wrapper(["status", "--json"], env=env)

    assert completed.returncode == 4, completed.stderr
    payload = _parse_json_stdout(completed.stdout)
    assert payload["operation"] == "preflight"
    assert payload["error_code"] == "missing_package"
    assert isinstance(payload["message"], str)
    assert isinstance(payload["remediation"], list)
    assert isinstance(payload["detected_environment"], dict)


def test_wrapper_returns_human_readable_stderr_without_json() -> None:
    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_DRY_RUN"] = "1"
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = str(_repo_root() / ".missing-venv" / "bin" / "python")
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env["GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS"] = "definitely_missing_pkg_for_gloggur_bootstrap_test"

    completed = _run_wrapper(["status"], env=env)

    assert completed.returncode == 4
    assert "gloggur preflight failed (missing_package)" in completed.stderr
    assert "Remediation:" in completed.stderr


def test_wrapper_returns_missing_python_when_no_interpreter_available(tmp_path: Path) -> None:
    source_script = _wrapper_path()
    temp_repo = tmp_path / "repo"
    temp_scripts = temp_repo / "scripts"
    temp_scripts.mkdir(parents=True)
    temp_script = temp_scripts / "gloggur"
    temp_script.write_text(source_script.read_text(encoding="utf8"), encoding="utf8")
    temp_script.chmod(0o755)

    tool_bin = tmp_path / "tool-bin"
    tool_bin.mkdir()
    dirname_bin = shutil.which("dirname")
    assert dirname_bin is not None
    os.symlink(dirname_bin, tool_bin / "dirname")

    env = os.environ.copy()
    env["PATH"] = str(tool_bin)

    completed = subprocess.run(
        ["/bin/bash", str(temp_script), "status", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(temp_repo),
        check=False,
    )

    assert completed.returncode == 3
    payload = _parse_json_stdout(completed.stdout)
    assert payload["operation"] == "preflight"
    assert payload["error_code"] == "missing_python"


def test_wrapper_reports_soft_preflight_timing_on_warm_path() -> None:
    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_DRY_RUN"] = "1"
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = str(_repo_root() / ".missing-venv" / "bin" / "python")
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env.pop("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", None)

    first = _run_wrapper(["status", "--json"], env=env)
    second = _run_wrapper(["status", "--json"], env=env)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_payload = _parse_json_stdout(first.stdout)
    second_payload = _parse_json_stdout(second.stdout)
    first_ms = int(first_payload["preflight_ms"])
    second_ms = int(second_payload["preflight_ms"])
    assert first_ms >= 0
    assert second_ms < 200
