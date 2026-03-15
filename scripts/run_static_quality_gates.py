from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Verification control-plane files — all three gates (ruff, mypy, black) apply.
CONTROL_PLANE_TARGETS = [
    "scripts/audit_verification_lanes.py",
    "scripts/check_coverage_baseline.py",
    "scripts/check_error_catalog_contract.py",
    "scripts/run_static_quality_gates.py",
    "tests/unit/test_audit_verification_lanes.py",
    "tests/unit/test_check_coverage_baseline.py",
    "tests/unit/test_verification_workflow.py",
    "tests/unit/test_run_static_quality_gates.py",
]

# Runtime package — ruff and black are clean; mypy has outstanding debt.
RUNTIME_PACKAGE_DIR = "src/gloggur"

# Combined targets used for ruff and black (control-plane + runtime package).
GATE_TARGETS = [*CONTROL_PLANE_TARGETS, RUNTIME_PACKAGE_DIR]

# Mypy targets remain narrower until runtime mypy debt is cleared.
MYPY_TARGETS = [
    "scripts/audit_verification_lanes.py",
    "scripts/check_coverage_baseline.py",
    "scripts/check_error_catalog_contract.py",
    "scripts/run_static_quality_gates.py",
]


def _resolve_tool_python() -> str:
    """Prefer the repo virtualenv interpreter for tool modules when it exists."""
    candidates = (
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


TOOL_PYTHON = _resolve_tool_python()


@dataclass(frozen=True)
class StageSpec:
    """Definition for one static-quality verification stage."""

    name: str
    failure_code: str
    remediation: str
    command: list[str]


@dataclass
class StageResult:
    """Machine-readable result for one static-quality verification stage."""

    name: str
    status: str
    duration_ms: int
    command: list[str]
    failure_code: str | None = None
    remediation: str | None = None
    detail: str | None = None
    stdout: str | None = None
    stderr: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the stage result."""
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "command": self.command,
            "failure_code": self.failure_code,
            "remediation": self.remediation,
            "detail": self.detail,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


STAGE_SPECS = [
    StageSpec(
        name="ruff",
        failure_code="static_gate_ruff_failed",
        remediation=(
            "Run the emitted Ruff command locally and fix lint violations in the gated files."
        ),
        command=[TOOL_PYTHON, "-m", "ruff", "check", *GATE_TARGETS],
    ),
    StageSpec(
        name="mypy",
        failure_code="static_gate_mypy_failed",
        remediation=(
            "Run the emitted mypy command locally and fix typing issues in the gated files."
        ),
        command=[TOOL_PYTHON, "-m", "mypy", *MYPY_TARGETS],
    ),
    StageSpec(
        name="black",
        failure_code="static_gate_black_failed",
        remediation=(
            "Run the emitted Black command locally and format the gated files before retrying."
        ),
        command=[TOOL_PYTHON, "-m", "black", "--check", *GATE_TARGETS],
    ),
]


def _truncate(value: str, limit: int = 1200) -> str:
    """Bound captured subprocess output so JSON stays readable."""
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


def _missing_targets() -> list[str]:
    """Return gate targets that do not exist in the workspace."""
    return [target for target in GATE_TARGETS if not (REPO_ROOT / target).exists()]


def _execute_stage_plan(specs: list[StageSpec]) -> tuple[list[StageResult], StageResult | None]:
    """Run stages in order and block downstream stages after the first failure."""
    results: list[StageResult] = []
    failed_result: StageResult | None = None
    for spec in specs:
        if failed_result is not None:
            results.append(
                StageResult(
                    name=spec.name,
                    status="not_run",
                    duration_ms=0,
                    command=spec.command,
                    failure_code="blocked_by_prior_stage_failure",
                    remediation="Fix the previous failed static-quality stage and rerun the gate.",
                    detail=f"Blocked by {failed_result.name} ({failed_result.failure_code})",
                )
            )
            continue

        started = time.monotonic()
        completed = subprocess.run(
            spec.command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout = _truncate(completed.stdout.strip()) if completed.stdout else None
        stderr = _truncate(completed.stderr.strip()) if completed.stderr else None
        if completed.returncode == 0:
            results.append(
                StageResult(
                    name=spec.name,
                    status="passed",
                    duration_ms=duration_ms,
                    command=spec.command,
                    stdout=stdout,
                    stderr=stderr,
                )
            )
            continue

        failed_result = StageResult(
            name=spec.name,
            status="failed",
            duration_ms=duration_ms,
            command=spec.command,
            failure_code=spec.failure_code,
            remediation=spec.remediation,
            detail=f"{spec.name} exited with code {completed.returncode}",
            stdout=stdout,
            stderr=stderr,
        )
        results.append(failed_result)
    return results, failed_result


def run_static_quality_gates() -> dict[str, object]:
    """Run the required-lane static gates and return structured results."""
    missing_targets = _missing_targets()
    if missing_targets:
        return {
            "ok": False,
            "target_scope": GATE_TARGETS,
            "summary": {"total_stages": len(STAGE_SPECS), "passed": 0, "failed": 1, "not_run": 2},
            "stages": [
                StageResult(
                    name=STAGE_SPECS[0].name,
                    status="failed",
                    duration_ms=0,
                    command=STAGE_SPECS[0].command,
                    failure_code="static_gate_targets_missing",
                    remediation=(
                        "Restore the expected verification control-plane files before "
                        "running the gate."
                    ),
                    detail="One or more gated files are missing from the workspace.",
                    stderr=", ".join(missing_targets),
                ).as_dict(),
                *[
                    StageResult(
                        name=spec.name,
                        status="not_run",
                        duration_ms=0,
                        command=spec.command,
                        failure_code="blocked_by_prior_stage_failure",
                        remediation=(
                            "Fix the previous failed static-quality stage and rerun the gate."
                        ),
                        detail="Blocked by missing gate targets.",
                    ).as_dict()
                    for spec in STAGE_SPECS[1:]
                ],
            ],
            "failure": {
                "code": "static_gate_targets_missing",
                "detail": "One or more gated files are missing from the workspace.",
                "remediation": (
                    "Restore the expected verification control-plane files before "
                    "running the gate."
                ),
                "missing_targets": missing_targets,
            },
        }

    stages, failed_result = _execute_stage_plan(STAGE_SPECS)
    passed = sum(result.status == "passed" for result in stages)
    failed = sum(result.status == "failed" for result in stages)
    not_run = sum(result.status == "not_run" for result in stages)
    payload: dict[str, object] = {
        "ok": failed_result is None,
        "target_scope": GATE_TARGETS,
        "summary": {
            "total_stages": len(STAGE_SPECS),
            "passed": passed,
            "failed": failed,
            "not_run": not_run,
        },
        "stages": [stage.as_dict() for stage in stages],
    }
    if failed_result is not None:
        payload["failure"] = {
            "code": failed_result.failure_code,
            "detail": failed_result.detail,
            "remediation": failed_result.remediation,
            "stage": failed_result.name,
        }
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the static-quality gate runner."""
    parser = argparse.ArgumentParser(
        description=("Run fail-closed static quality gates for the required verification lane.")
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def _render_markdown(payload: dict[str, object]) -> str:
    """Render a compact markdown summary for local investigation."""
    lines = ["# Static Quality Gates", ""]
    lines.append(f"- ok: `{payload.get('ok')}`")
    lines.append(f"- target_scope: `{payload.get('target_scope')}`")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        lines.append(f"- summary: `{summary}`")
    failure = payload.get("failure")
    if isinstance(failure, dict):
        lines.append(f"- failure.code: `{failure.get('code')}`")
        lines.append(f"- failure.detail: `{failure.get('detail')}`")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the static-quality gates and emit either JSON or markdown output."""
    args = _parse_args(argv)
    payload = run_static_quality_gates()
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload), end="")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
