"""Preflight launcher that locates a healthy Python runtime before executing gloggur.

When ``scripts/gloggur`` is invoked in a fresh environment, it runs this module
to probe candidate interpreters (venv first, then system Python fallbacks) and
exec into the first one that can successfully import the gloggur package.  If
all probes fail, it emits a structured JSON or human-readable failure payload
with an actionable error code and remediation steps, then exits non-zero.

Exit codes (see ``EXIT_CODES``):
  2  missing_venv       – repository virtualenv missing and no healthy fallback
  3  missing_python     – no usable Python interpreter found on PATH
  4  missing_package    – interpreter found but gloggur dependencies absent
  5  broken_environment – other runtime/import failure
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

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
    """Result of probing a single Python interpreter candidate.

    ``exists`` indicates whether the interpreter binary was found on disk or
    PATH.  ``healthy`` indicates the interpreter could successfully import the
    required gloggur module.  ``reason`` and ``detail`` are populated on failure.
    """

    candidate_type: str
    interpreter: str
    module: str
    exists: bool
    healthy: bool
    reason: str | None = None
    detail: str | None = None
    returncode: int | None = None


@dataclass
class LaunchPlan:
    """Resolved execution plan produced by ``build_launch_plan``.

    When ``ready`` is ``True``, ``args``, ``interpreter``, and ``module`` are
    populated and the caller may safely exec into the interpreter.  When
    ``ready`` is ``False``, ``error_code``, ``message``, and ``remediation``
    describe the failure and no exec should be attempted.
    """

    ready: bool
    args: list[str]
    interpreter: str | None
    module: str | None
    candidate_type: str | None
    env: dict[str, str]
    repo_root: str
    probes: list[CandidateProbe]
    error_code: str | None = None
    message: str | None = None
    remediation: list[str] | None = None


def _is_truthy(value: str | None) -> bool:
    """Return ``True`` for environment-variable strings that represent a truthy boolean."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_json_mode(args: Sequence[str]) -> bool:
    """Return ``True`` when ``--json`` is present in the argument list."""
    return "--json" in args


def _repo_root() -> str:
    """Resolve the repository root from this module's on-disk location."""
    module_dir = Path(__file__).resolve().parent
    if module_dir.parent.name == "src":
        return str(module_dir.parent.parent)
    return str(module_dir.parent)


def _import_root(repo_root: str) -> str:
    """Return the directory that should be prepended to ``PYTHONPATH`` for imports.

    Returns ``<repo_root>/src`` when that directory exists (src-layout repos),
    otherwise falls back to ``repo_root`` directly.
    """
    src_root = os.path.join(repo_root, "src")
    if os.path.isdir(src_root):
        return src_root
    return repo_root


def _prepend_pythonpath(env: dict[str, str], repo_root: str) -> dict[str, str]:
    """Return a copy of ``env`` with the import root prepended to ``PYTHONPATH``.

    Deduplicates the import root so repeated calls are idempotent.
    """
    result = dict(env)
    import_root = _import_root(repo_root)
    existing = result.get("PYTHONPATH", "")
    parts = [entry for entry in existing.split(os.pathsep) if entry]
    parts = [entry for entry in parts if entry != import_root]
    result["PYTHONPATH"] = os.pathsep.join([import_root, *parts])
    return result


def _collect_required_imports(env: dict[str, str]) -> list[str]:
    """Parse the ``GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS`` env var into a list of module names."""
    raw = env.get("GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS", "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values


def _resolve_system_candidates(env: dict[str, str]) -> list[str]:
    """Return a deduplicated list of system Python interpreter paths to probe.

    Reads ``GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS`` (colon-separated) when set;
    defaults to ``["python3", "python"]``.  PATH resolution is applied to
    bare names so the returned list contains absolute paths where available.
    """
    raw = env.get("GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS")
    entries: list[str] = []
    if raw:
        entries = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    else:
        entries = ["python3", "python"]

    resolved: list[str] = []
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


def _resolve_venv_python(repo_root: str, env: dict[str, str]) -> str:
    (
        "Return the expected venv Python path, honouring"
        " ``GLOGGUR_PREFLIGHT_VENV_PYTHON`` overrides."
    )
    override = env.get("GLOGGUR_PREFLIGHT_VENV_PYTHON")
    if override:
        return override
    return os.path.join(repo_root, ".venv", "bin", "python")


def _probe_script(module: str, required_imports: Sequence[str]) -> str:
    """Build a Python one-liner that imports ``module`` and any ``required_imports``.

    The resulting script is passed to the candidate interpreter via ``-c`` to
    verify that the runtime can successfully import gloggur and its dependencies.
    """
    commands = [
        "import importlib",
        f"importlib.import_module({module!r})",
    ]
    for item in required_imports:
        commands.append(f"importlib.import_module({item!r})")
    return "; ".join(commands)


def _probe_failure_reason(output: str) -> str:
    """Classify probe stderr/stdout into a stable failure reason code."""
    if "ModuleNotFoundError" in output or "ImportError" in output:
        return "missing_package"
    return "broken_environment"


def _probe_candidate(
    candidate_type: str,
    interpreter: str,
    module: str,
    repo_root: str,
    env: dict[str, str],
    required_imports: Sequence[str],
) -> CandidateProbe:
    """Probe a single interpreter candidate and return a ``CandidateProbe`` result.

    Runs ``interpreter -c <probe_script>`` with a 15-second timeout.  Returns a
    healthy probe on exit code 0, or a failure probe with a classified reason
    code on any error (not found, import failure, timeout, OS error).
    """
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
        output.splitlines()[-1] if output else f"probe failed with exit code {completed.returncode}"
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
    """Derive the top-level failure error code from a set of failed probes.

    Priority order: missing_python > missing_package > missing_venv > broken_environment.
    """
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
    """Return a human-readable one-sentence failure description for the given error code."""
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


def _remediation_steps(error_code: str, repo_root: str) -> list[str]:
    """Return an ordered list of remediation steps for the given error code."""
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
    env: dict[str, str] | None = None,
    repo_root: str | None = None,
) -> LaunchPlan:
    """Probe available Python interpreters and return a ready-or-failed ``LaunchPlan``.

    Probes the venv interpreter first; falls back to system Python candidates
    in order.  Returns a ``LaunchPlan`` with ``ready=True`` and a selected
    interpreter as soon as a healthy candidate is found, or ``ready=False``
    with a classified error code and remediation steps if all probes fail.
    """
    runtime_env = dict(os.environ if env is None else env)
    root = repo_root or _repo_root()
    venv_python = _resolve_venv_python(root, runtime_env)
    venv_exists = os.path.exists(venv_python)
    required_imports = _collect_required_imports(runtime_env)
    probe_module = runtime_env.get("GLOGGUR_PREFLIGHT_PROBE_MODULE", PROBE_MODULE_DEFAULT)
    probes: list[CandidateProbe] = []

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


def _serialize_probes(probes: Sequence[CandidateProbe]) -> list[dict[str, object]]:
    """Convert a sequence of ``CandidateProbe`` instances to plain dicts for JSON output."""
    return [asdict(probe) for probe in probes]


def build_failure_payload(plan: LaunchPlan, preflight_ms: int) -> dict[str, object]:
    """Build the structured JSON/human-readable failure payload for a failed ``LaunchPlan``.

    Includes error code, message, remediation steps, detected environment details,
    and the duration of the preflight check in milliseconds.
    """
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


def build_ready_payload(plan: LaunchPlan, preflight_ms: int) -> dict[str, object]:
    """Build the structured JSON/human-readable success payload for a ready ``LaunchPlan``.

    Includes the selected candidate type, interpreter path, module, and
    detected environment details for downstream diagnostics.
    """
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


def _emit(payload: dict[str, object], as_json: bool) -> None:
    """Print ``payload`` as JSON to stdout or as a human-readable summary to stderr."""
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
    """Exec into the selected interpreter via ``os.execvpe``, replacing the current process.

    This function does not return on success.
    """
    assert plan.ready
    assert plan.interpreter is not None
    assert plan.module is not None
    command = [plan.interpreter, "-m", plan.module, *plan.args]
    os.execvpe(plan.interpreter, command, plan.env)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the gloggur preflight bootstrap launcher.

    Builds a launch plan, emits a ready or failure payload, and either execs
    into the selected interpreter or returns a non-zero exit code.  Dry-run
    mode (``GLOGGUR_PREFLIGHT_DRY_RUN=1``) emits the ready payload without
    execing, which is useful for testing and diagnostics.
    """
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
