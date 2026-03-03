from __future__ import annotations

import hashlib
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
    assert payload["ok"] is False
    assert payload["stage"] == "bootstrap"
    assert payload["error_code"] == "missing_package"
    compatibility = payload.get("compatibility")
    assert isinstance(compatibility, dict)
    assert isinstance(compatibility.get("message"), str)
    assert isinstance(compatibility.get("remediation"), list)
    assert isinstance(compatibility.get("detected_environment"), dict)


def test_wrapper_unknown_command_returns_dispatch_json_envelope() -> None:
    env = os.environ.copy()
    env.pop("GLOGGUR_PREFLIGHT_DRY_RUN", None)
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = str(_repo_root() / ".missing-venv" / "bin" / "python")
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env.pop("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", None)

    completed = _run_wrapper(["definitely-unknown-command", "--json"], env=env)
    payload = _parse_json_stdout(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["stage"] == "dispatch"
    assert payload["error_code"] == "cli_usage_error"
    assert "\n" not in completed.stdout.strip()


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


def test_wrapper_exit_code_mapping_for_missing_venv_is_stable() -> None:
    false_bin = shutil.which("false")
    assert false_bin is not None

    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_DRY_RUN"] = "1"
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = str(_repo_root() / ".missing-venv" / "bin" / "python")
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = false_bin
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env.pop("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", None)

    completed = _run_wrapper(["status", "--json"], env=env)

    assert completed.returncode == 2, completed.stderr
    payload = _parse_json_stdout(completed.stdout)
    assert payload["ok"] is False
    assert payload["stage"] == "bootstrap"
    assert payload["error_code"] == "missing_venv"


def test_wrapper_exit_code_mapping_for_broken_environment_is_stable() -> None:
    false_bin = shutil.which("false")
    assert false_bin is not None

    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_DRY_RUN"] = "1"
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = false_bin
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = false_bin
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env.pop("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", None)

    completed = _run_wrapper(["status", "--json"], env=env)

    assert completed.returncode == 5, completed.stderr
    payload = _parse_json_stdout(completed.stdout)
    assert payload["ok"] is False
    assert payload["stage"] == "bootstrap"
    assert payload["error_code"] == "broken_environment"


def test_wrapper_strict_bootstrap_mode_fails_when_optional_bootstrap_capabilities_are_degraded() -> (
    None
):
    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_DRY_RUN"] = "1"
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = sys.executable
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env["GLOGGUR_BOOTSTRAP_STRICT"] = "1"
    env.pop("BOOTSTRAP_GLOGGUR_LOG_FILE", None)
    env.pop("BOOTSTRAP_GLOGGUR_STATE_FILE", None)

    completed = _run_wrapper(["status", "--json"], env=env)
    payload = _parse_json_stdout(completed.stdout)

    assert completed.returncode == 1
    assert payload["ok"] is False
    assert payload["stage"] == "bootstrap"
    assert payload["error_code"] == "bootstrap_capability_degraded"


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
    assert payload["ok"] is False
    assert payload["stage"] == "bootstrap"
    assert payload["error_code"] == "missing_python"


def test_wrapper_executes_status_via_system_fallback_when_venv_missing(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("GLOGGUR_PREFLIGHT_DRY_RUN", None)
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = str(_repo_root() / ".missing-venv" / "bin" / "python")
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env.pop("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", None)
    env["GLOGGUR_CACHE_DIR"] = str(tmp_path / "cache")

    completed = _run_wrapper(["status", "--json"], env=env)

    assert completed.returncode == 0, completed.stderr
    payload = _parse_json_stdout(completed.stdout)
    assert "cache_dir" in payload
    assert payload.get("operation") != "preflight"


def test_wrapper_preserves_caller_cwd_when_requested(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("GLOGGUR_PREFLIGHT_DRY_RUN", None)
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = str(_repo_root() / ".missing-venv" / "bin" / "python")
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env.pop("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", None)
    env["GLOGGUR_RUN_FROM_CALLER_CWD"] = "1"
    workspace = tmp_path / "external-workspace"
    workspace.mkdir()

    completed = _run_wrapper(["status", "--json"], env=env, cwd=workspace)

    assert completed.returncode == 0, completed.stderr
    payload = _parse_json_stdout(completed.stdout)
    expected_workspace_hash = hashlib.sha256(
        os.path.abspath(str(workspace)).encode("utf8")
    ).hexdigest()
    assert payload.get("workspace_path_hash") == expected_workspace_hash


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


def test_wrapper_reports_soft_preflight_timing_on_warm_venv_path() -> None:
    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_DRY_RUN"] = "1"
    env["GLOGGUR_PREFLIGHT_VENV_PYTHON"] = sys.executable
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable
    env["GLOGGUR_PREFLIGHT_PROBE_MODULE"] = "gloggur.bootstrap_launcher"
    env.pop("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", None)

    first = _run_wrapper(["status", "--json"], env=env)
    second = _run_wrapper(["status", "--json"], env=env)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_payload = _parse_json_stdout(first.stdout)
    second_payload = _parse_json_stdout(second.stdout)
    assert first_payload["selected_candidate"] == "venv"
    assert second_payload["selected_candidate"] == "venv"
    assert int(second_payload["preflight_ms"]) < 200
