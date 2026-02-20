from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROBE_MODULE_DEFAULT = "gloggur.cli.main"
VENV_EXEC_MODULE = "gloggur.cli.main"
SYSTEM_EXEC_MODULE = "gloggur"
CHECK_OPERATION = "preflight"

EXIT_CODES = {
    "missing_venv": 2,
    "missing_python": 3,
    "missing_package": 4,
    "broken_environment": 5,
}


@dataclass
class CandidateProbe:
    candidate_type: str
    interpreter: str
    module: str
    exists: bool
    healthy: bool
    reason: Optional[str] = None
    detail: Optional[str] = None
    returncode: Optional[int] = None


@dataclass
class LaunchPlan:
    ready: bool
    args: List[str]
    interpreter: Optional[str]
    module: Optional[str]
    candidate_type: Optional[str]
    env: Dict[str, str]
    repo_root: str
    probes: List[CandidateProbe]
    error_code: Optional[str] = None
    message: Optional[str] = None
    remediation: Optional[List[str]] = None


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_json_mode(args: Sequence[str]) -> bool:
    return "--json" in args


def _repo_root() -> str:
    return str(Path(__file__).resolve().parent.parent)


def _prepend_pythonpath(env: Dict[str, str], repo_root: str) -> Dict[str, str]:
    result = dict(env)
    existing = result.get("PYTHONPATH", "")
    if existing:
        result["PYTHONPATH"] = os.pathsep.join([repo_root, existing])
    else:
        result["PYTHONPATH"] = repo_root
    return result


def _collect_required_imports(env: Dict[str, str]) -> List[str]:
    raw = env.get("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values


def _resolve_system_candidates(env: Dict[str, str]) -> List[str]:
    raw = env.get("GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS")
    entries: List[str] = []
    if raw:
        entries = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    else:
        entries = ["python3", "python"]

    resolved: List[str] = []
    seen: set[str] = set()
    for entry in entries:
        candidate = entry
        if os.path.sep not in entry:
            resolved_candidate = shutil.which(entry)
            if resolved_candidate:
                candidate = resolved_candidate
            else:
                candidate = entry
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return resolved


def _resolve_venv_python(repo_root: str, env: Dict[str, str]) -> str:
    override = env.get("GLOGGUR_PREFLIGHT_VENV_PYTHON")
    if override:
        return override
    return os.path.join(repo_root, ".venv", "bin", "python")


def _probe_script(module: str, required_imports: Sequence[str]) -> str:
    commands = [
        "import importlib",
        f"importlib.import_module({module!r})",
    ]
    for item in required_imports:
        commands.append(f"importlib.import_module({item!r})")
    return "; ".join(commands)


def _probe_failure_reason(output: str) -> str:
    if "ModuleNotFoundError" in output or "ImportError" in output:
        return "missing_package"
    return "broken_environment"


def _probe_candidate(
    candidate_type: str,
    interpreter: str,
    module: str,
    repo_root: str,
    env: Dict[str, str],
    required_imports: Sequence[str],
) -> CandidateProbe:
    if not interpreter:
        return CandidateProbe(
            candidate_type=candidate_type,
            interpreter=interpreter,
            module=module,
            exists=False,
            healthy=False,
            reason="interpreter_not_found",
            detail="empty interpreter path",
        )

    exists = (
        os.path.exists(interpreter)
        if os.path.sep in interpreter
        else bool(shutil.which(interpreter))
    )
    if not exists:
        return CandidateProbe(
            candidate_type=candidate_type,
            interpreter=interpreter,
            module=module,
            exists=False,
            healthy=False,
            reason="interpreter_not_found",
            detail=f"interpreter not found: {interpreter}",
        )

    probe_env = _prepend_pythonpath(env, repo_root)
    command = [interpreter, "-c", _probe_script(module, required_imports)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=probe_env,
            timeout=15,
            check=False,
        )
    except FileNotFoundError as exc:
        return CandidateProbe(
            candidate_type=candidate_type,
            interpreter=interpreter,
            module=module,
            exists=False,
            healthy=False,
            reason="interpreter_not_found",
            detail=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        return CandidateProbe(
            candidate_type=candidate_type,
            interpreter=interpreter,
            module=module,
            exists=True,
            healthy=False,
            reason="broken_environment",
            detail=f"probe timed out: {exc}",
        )
    except OSError as exc:
        return CandidateProbe(
            candidate_type=candidate_type,
            interpreter=interpreter,
            module=module,
            exists=True,
            healthy=False,
            reason="broken_environment",
            detail=str(exc),
        )

    if completed.returncode == 0:
        return CandidateProbe(
            candidate_type=candidate_type,
            interpreter=interpreter,
            module=module,
            exists=True,
            healthy=True,
            returncode=0,
        )

    output = "\n".join(
        part.strip()
        for part in (completed.stderr or "", completed.stdout or "")
        if part and part.strip()
    )
    reason = _probe_failure_reason(output)
    detail_line = (
        output.splitlines()[-1]
        if output
        else f"probe failed with exit code {completed.returncode}"
    )
    return CandidateProbe(
        candidate_type=candidate_type,
        interpreter=interpreter,
        module=module,
        exists=True,
        healthy=False,
        reason=reason,
        detail=detail_line,
        returncode=completed.returncode,
    )


def _classify_failure(probes: Sequence[CandidateProbe], venv_exists: bool) -> str:
    any_existing = any(probe.exists for probe in probes)
    if not any_existing:
        return "missing_python"

    if any(probe.reason == "missing_package" for probe in probes):
        return "missing_package"

    if not venv_exists and not any(
        probe.healthy for probe in probes if probe.candidate_type == "system"
    ):
        return "missing_venv"

    return "broken_environment"


def _failure_message(error_code: str, repo_root: str, venv_python: str) -> str:
    if error_code == "missing_python":
        return "No usable Python interpreter was found for gloggur preflight."
    if error_code == "missing_venv":
        return (
            "Repository virtualenv is missing and no healthy fallback runtime was found "
            f"(expected {venv_python})."
        )
    if error_code == "missing_package":
        return "Python runtime found, but required gloggur dependencies are missing."
    return f"Detected a broken Python environment while preparing gloggur in {repo_root}."


def _remediation_steps(error_code: str, repo_root: str) -> List[str]:
    bootstrap = os.path.join(repo_root, "scripts", "bootstrap_gloggur_env.sh")
    common = [
        f"Run `{bootstrap}` to create/repair .venv and install dependencies.",
        "Re-run `scripts/gloggur status --json` after bootstrap completes.",
    ]
    if error_code == "missing_python":
        return [
            "Install Python 3.10+ and ensure `python3` is on PATH.",
            *common,
        ]
    if error_code == "missing_package":
        return [
            *common,
            (
                "If using a custom interpreter, install with "
                "`pip install -e '.[all,dev]'` in repo root."
            ),
        ]
    if error_code == "missing_venv":
        return common
    return [
        *common,
        "If issues persist, remove `.venv` and rerun bootstrap with `--recreate`.",
    ]


def build_launch_plan(
    args: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    repo_root: Optional[str] = None,
) -> LaunchPlan:
    runtime_env = dict(os.environ if env is None else env)
    root = repo_root or _repo_root()
    venv_python = _resolve_venv_python(root, runtime_env)
    venv_exists = os.path.exists(venv_python)
    required_imports = _collect_required_imports(runtime_env)
    probe_module = runtime_env.get("GLOGGUR_PREFLIGHT_PROBE_MODULE", PROBE_MODULE_DEFAULT)
    probes: List[CandidateProbe] = []

    venv_probe = _probe_candidate(
        candidate_type="venv",
        interpreter=venv_python,
        module=probe_module,
        repo_root=root,
        env=runtime_env,
        required_imports=required_imports,
    )
    probes.append(venv_probe)
    launch_env = _prepend_pythonpath(runtime_env, root)
    if venv_probe.healthy:
        return LaunchPlan(
            ready=True,
            args=list(args),
            interpreter=venv_python,
            module=VENV_EXEC_MODULE,
            candidate_type="venv",
            env=launch_env,
            repo_root=root,
            probes=probes,
        )

    for system_python in _resolve_system_candidates(runtime_env):
        system_probe = _probe_candidate(
            candidate_type="system",
            interpreter=system_python,
            module=probe_module,
            repo_root=root,
            env=runtime_env,
            required_imports=required_imports,
        )
        probes.append(system_probe)
        if system_probe.healthy:
            return LaunchPlan(
                ready=True,
                args=list(args),
                interpreter=system_probe.interpreter,
                module=SYSTEM_EXEC_MODULE,
                candidate_type="system",
                env=launch_env,
                repo_root=root,
                probes=probes,
            )

    error_code = _classify_failure(probes=probes, venv_exists=venv_exists)
    return LaunchPlan(
        ready=False,
        args=list(args),
        interpreter=None,
        module=None,
        candidate_type=None,
        env=launch_env,
        repo_root=root,
        probes=probes,
        error_code=error_code,
        message=_failure_message(error_code, root, venv_python),
        remediation=_remediation_steps(error_code, root),
    )


def _serialize_probes(probes: Sequence[CandidateProbe]) -> List[Dict[str, object]]:
    return [asdict(probe) for probe in probes]


def build_failure_payload(plan: LaunchPlan, preflight_ms: int) -> Dict[str, object]:
    assert not plan.ready
    venv_python = _resolve_venv_python(plan.repo_root, plan.env)
    system_candidates = _resolve_system_candidates(plan.env)
    return {
        "operation": CHECK_OPERATION,
        "error_code": plan.error_code,
        "message": plan.message,
        "remediation": plan.remediation or [],
        "detected_environment": {
            "repo_root": plan.repo_root,
            "venv_python": venv_python,
            "venv_exists": os.path.exists(venv_python),
            "system_candidates": system_candidates,
            "pythonpath": plan.env.get("PYTHONPATH"),
            "probes": _serialize_probes(plan.probes),
        },
        "preflight_ms": preflight_ms,
    }


def build_ready_payload(plan: LaunchPlan, preflight_ms: int) -> Dict[str, object]:
    assert plan.ready
    return {
        "operation": CHECK_OPERATION,
        "ready": True,
        "selected_candidate": plan.candidate_type,
        "selected_interpreter": plan.interpreter,
        "selected_module": plan.module,
        "argv": plan.args,
        "preflight_ms": preflight_ms,
        "detected_environment": {
            "repo_root": plan.repo_root,
            "pythonpath": plan.env.get("PYTHONPATH"),
            "probes": _serialize_probes(plan.probes),
        },
    }


def _emit(payload: Dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return

    if payload.get("ready") is True:
        print(
            "gloggur preflight OK: "
            f"{payload.get('selected_candidate')} -> {payload.get('selected_interpreter')}",
            file=sys.stderr,
        )
        return

    print(
        f"gloggur preflight failed ({payload.get('error_code')}): {payload.get('message')}",
        file=sys.stderr,
    )
    print("Remediation:", file=sys.stderr)
    for index, step in enumerate(payload.get("remediation", []), start=1):
        print(f"{index}. {step}", file=sys.stderr)


def _execute(plan: LaunchPlan) -> None:
    assert plan.ready
    assert plan.interpreter is not None
    assert plan.module is not None
    command = [plan.interpreter, "-m", plan.module, *plan.args]
    os.execvpe(plan.interpreter, command, plan.env)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = _is_json_mode(args)
    started = time.perf_counter()

    try:
        plan = build_launch_plan(args=args)
    except Exception as exc:  # pragma: no cover - defensive path
        preflight_ms = int((time.perf_counter() - started) * 1000)
        payload = {
            "operation": CHECK_OPERATION,
            "error_code": "broken_environment",
            "message": "Unexpected error while running gloggur preflight.",
            "remediation": _remediation_steps("broken_environment", _repo_root()),
            "detected_environment": {
                "repo_root": _repo_root(),
                "exception": repr(exc),
            },
            "preflight_ms": preflight_ms,
        }
        _emit(payload, as_json=as_json)
        return EXIT_CODES["broken_environment"]

    preflight_ms = int((time.perf_counter() - started) * 1000)
    dry_run = _is_truthy(os.environ.get("GLOGGUR_PREFLIGHT_DRY_RUN"))
    if plan.ready:
        if dry_run:
            _emit(build_ready_payload(plan, preflight_ms=preflight_ms), as_json=as_json)
            return 0
        _execute(plan)
        return 0

    payload = build_failure_payload(plan, preflight_ms=preflight_ms)
    _emit(payload, as_json=as_json)
    return EXIT_CODES.get(str(plan.error_code), EXIT_CODES["broken_environment"])


if __name__ == "__main__":
    raise SystemExit(main())
