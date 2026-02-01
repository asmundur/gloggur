from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation.logging_utils import configure_logging, log_event
from scripts.validation.report_templates import (
    PhaseReport,
    TestCaseResult,
    ValidationReport,
    build_validation_report,
    render_json,
    render_markdown,
)


@dataclass(frozen=True)
class _PhaseDefinition:
    title: str
    script: Path


PHASE_DEFINITIONS: Dict[int, _PhaseDefinition] = {
    1: _PhaseDefinition(title="Smoke Tests", script=Path("scripts/validate_phase1.py")),
    2: _PhaseDefinition(title="Embedding Providers", script=Path("scripts/validate_phase2.py")),
    3: _PhaseDefinition(title="Edge Cases", script=Path("scripts/validate_phase3_4.py")),
    4: _PhaseDefinition(title="Performance Benchmarks", script=Path("scripts/validate_phase3_4.py")),
}

logger = logging.getLogger(__name__)


class ValidationRunner:
    def __init__(
        self,
        phases: Optional[List[int]] = None,
        quick: bool = False,
        verbose: bool = False,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> None:
        self.phases = phases
        self.quick = quick
        self.verbose = verbose
        self.parallel = parallel
        self.max_workers = max_workers

    def run_all_phases(self) -> ValidationReport:
        phases = self._resolve_phases()
        if self.parallel and len(phases) > 1:
            logger.info("Running phases in parallel: %s", phases)
            return self._run_all_phases_parallel(phases)
        logger.info("Running phases in serial: %s", phases)
        return self._run_all_phases_serial(phases)

    def run_phase(self, phase_num: int) -> PhaseReport:
        reports = self._run_single_phase(phase_num)
        for report in reports:
            if report.phase == phase_num:
                return report
        return _missing_phase_report(phase_num, PHASE_DEFINITIONS[phase_num].title, "Phase output missing.")

    def generate_comprehensive_report(self, report: ValidationReport) -> str:
        return render_markdown(report)

    def save_report(self, report: ValidationReport, path: str, fmt: str) -> None:
        output_path = Path(path)
        if fmt == "json":
            output = json.dumps(render_json(report), indent=2)
        else:
            output = render_markdown(report)
        output_path.write_text(output, encoding="utf8")

    def _resolve_phases(self) -> List[int]:
        if self.phases:
            return sorted(set(self.phases))
        if self.quick:
            return [1, 2]
        return [1, 2, 3, 4]

    def _run_single_phase(self, phase: int) -> List[PhaseReport]:
        definition = PHASE_DEFINITIONS[phase]
        return _execute_phase_script(
            definition.script,
            requested_phases=[phase],
            fallback_titles={phase: definition.title},
            verbose=self.verbose,
        )

    def _run_combined_phase34(self, phases: List[int]) -> List[PhaseReport]:
        script = PHASE_DEFINITIONS[3].script
        fallback_titles = {
            3: PHASE_DEFINITIONS[3].title,
            4: PHASE_DEFINITIONS[4].title,
        }
        return _execute_phase_script(
            script,
            requested_phases=[phase for phase in phases if phase in (3, 4)],
            fallback_titles=fallback_titles,
            verbose=self.verbose,
        )

    def _run_all_phases_serial(self, phases: List[int]) -> ValidationReport:
        phase_reports: List[PhaseReport] = []
        handle_phase34 = any(phase in (3, 4) for phase in phases)
        for phase in phases:
            if phase in (3, 4):
                if not handle_phase34:
                    continue
                handle_phase34 = False
                phase_reports.extend(self._run_combined_phase34(phases))
                continue
            phase_reports.extend(self._run_single_phase(phase))
        phase_reports = [report for report in phase_reports if report.phase in phases]
        phase_reports.sort(key=lambda report: report.phase)
        return build_validation_report(phase_reports)

    def _run_all_phases_parallel(self, phases: List[int]) -> ValidationReport:
        tasks = self._build_parallel_tasks(phases)
        phase_reports: List[PhaseReport] = []
        max_workers = self.max_workers
        if max_workers is None:
            cpu_count = os.cpu_count() or 1
            max_workers = max(1, min(len(tasks), cpu_count))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _execute_phase_script,
                    task["script"],
                    task["requested_phases"],
                    task["fallback_titles"],
                    self.verbose,
                ): task
                for task in tasks
            }
            for future in as_completed(future_map):
                phase_reports.extend(future.result())
        phase_reports = [report for report in phase_reports if report.phase in phases]
        phase_reports.sort(key=lambda report: report.phase)
        return build_validation_report(phase_reports)

    def _build_parallel_tasks(self, phases: List[int]) -> List[Dict[str, object]]:
        tasks: List[Dict[str, object]] = []
        handle_phase34 = any(phase in (3, 4) for phase in phases)
        for phase in phases:
            if phase in (3, 4):
                if not handle_phase34:
                    continue
                handle_phase34 = False
                tasks.append(
                    {
                        "script": PHASE_DEFINITIONS[3].script,
                        "requested_phases": [phase for phase in phases if phase in (3, 4)],
                        "fallback_titles": {
                            3: PHASE_DEFINITIONS[3].title,
                            4: PHASE_DEFINITIONS[4].title,
                        },
                    }
                )
                continue
            definition = PHASE_DEFINITIONS[phase]
            tasks.append(
                {
                    "script": definition.script,
                    "requested_phases": [phase],
                    "fallback_titles": {phase: definition.title},
                }
            )
        return tasks


def _execute_phase_script(
    script_path: Path,
    requested_phases: List[int],
    fallback_titles: Dict[int, str],
    verbose: bool = False,
) -> List[PhaseReport]:
    if not script_path.exists():
        return [
            _missing_phase_report(phase, fallback_titles.get(phase, f"Phase {phase}"), "Phase script missing.")
            for phase in requested_phases
        ]

    cmd = [sys.executable, str(script_path), "--format", "json"]
    if verbose:
        cmd.append("--verbose")

    cache_dir = tempfile.mkdtemp(prefix="gloggur-validate-")
    env = os.environ.copy()
    env["GLOGGUR_CACHE_DIR"] = cache_dir
    if os.getenv("GLOGGUR_LOG_FILE"):
        env["GLOGGUR_LOG_FILE"] = os.getenv("GLOGGUR_LOG_FILE", "")
    if os.getenv("GLOGGUR_TRACE_ID"):
        env["GLOGGUR_TRACE_ID"] = os.getenv("GLOGGUR_TRACE_ID", "")
    if os.getenv("GLOGGUR_LOG_LEVEL"):
        env["GLOGGUR_LOG_LEVEL"] = os.getenv("GLOGGUR_LOG_LEVEL", "")
    if os.getenv("GLOGGUR_DEBUG_LOGS"):
        env["GLOGGUR_DEBUG_LOGS"] = os.getenv("GLOGGUR_DEBUG_LOGS", "")
    try:
        start = time.perf_counter()
        if os.getenv("GLOGGUR_DEBUG_LOGS"):
            log_event(
                logger,
                logging.DEBUG,
                "phase.execute.start",
                command=cmd,
                env=env,
                requested_phases=requested_phases,
            )
        else:
            logger.debug("Executing phase script: %s", " ".join(cmd))
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        duration_ms = (time.perf_counter() - start) * 1000
        if os.getenv("GLOGGUR_DEBUG_LOGS"):
            log_event(
                logger,
                logging.DEBUG,
                "phase.execute.finish",
                command=cmd,
                duration_ms=duration_ms,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    payload = _parse_json_output(completed.stdout)
    if payload is None and completed.stderr:
        payload = _parse_json_output(completed.stderr)

    if payload is None:
        message = _format_execution_error(cmd, completed.returncode, completed.stdout, completed.stderr)
        return [
            _missing_phase_report(phase, fallback_titles.get(phase, f"Phase {phase}"), message)
            for phase in requested_phases
        ]

    reports = _phase_reports_from_payload(payload, requested_phases, fallback_titles, duration_ms)
    found_phases = {report.phase for report in reports}
    for phase in requested_phases:
        if phase not in found_phases:
            reports.append(
                _missing_phase_report(
                    phase,
                    fallback_titles.get(phase, f"Phase {phase}"),
                    "Phase report missing from payload.",
                )
            )
    if completed.returncode != 0:
        for report in reports:
            report.status = "failed"
            report.tests.append(
                TestCaseResult(
                    name="Phase runner",
                    status="failed",
                    message=f"Non-zero exit code {completed.returncode}",
                )
            )
            report.summary = _summarize_tests(report.tests)
    return reports


def _parse_json_output(raw: str) -> Optional[Dict[str, object]]:
    data = raw.strip()
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        pass
    for line in reversed(data.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _phase_reports_from_payload(
    payload: Dict[str, object],
    requested_phases: List[int],
    fallback_titles: Dict[int, str],
    duration_ms: float,
) -> List[PhaseReport]:
    if "phases" in payload and isinstance(payload["phases"], list):
        reports: List[PhaseReport] = []
        for item in payload["phases"]:
            if isinstance(item, dict):
                reports.append(_phase_report_from_payload(item, fallback_titles, duration_ms))
        return reports
    return [_phase_report_from_payload(payload, fallback_titles, duration_ms)]


def _phase_report_from_payload(
    payload: Dict[str, object], fallback_titles: Dict[int, str], duration_ms: float
) -> PhaseReport:
    phase_num = int(payload.get("phase", payload.get("phase_number", 0)) or 0)
    if phase_num == 0:
        phase_num = min(fallback_titles.keys()) if fallback_titles else 0
    title = str(payload.get("title") or fallback_titles.get(phase_num, f"Phase {phase_num}"))

    tests = _tests_from_payload(payload)
    summary = _summary_from_payload(payload, tests)
    status = _status_from_summary(summary)
    performance = payload.get("performance") if isinstance(payload.get("performance"), dict) else None
    issues = payload.get("issues") if isinstance(payload.get("issues"), list) else None
    duration = float(payload.get("duration_ms", duration_ms) or duration_ms)

    return PhaseReport(
        phase=phase_num,
        title=title,
        status=status,
        summary=summary,
        tests=tests,
        duration_ms=duration,
        performance=performance,
        issues=issues,
    )


def _tests_from_payload(payload: Dict[str, object]) -> List[TestCaseResult]:
    if "tests" in payload and isinstance(payload["tests"], list):
        return [_test_from_payload(item) for item in payload["tests"] if isinstance(item, dict)]
    if "sections" in payload and isinstance(payload["sections"], list):
        tests: List[TestCaseResult] = []
        for section in payload["sections"]:
            if not isinstance(section, dict):
                continue
            section_title = str(section.get("title", "Section"))
            results = section.get("results", [])
            if not isinstance(results, list):
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                name = str(result.get("name", "test"))
                message = str(result.get("message", result.get("details", "")))
                status = "passed" if result.get("passed", False) else "failed"
                details = result.get("details") if isinstance(result.get("details"), dict) else None
                tests.append(
                    TestCaseResult(
                        name=f"{section_title}: {name}",
                        status=status,
                        message=message,
                        details=details,
                    )
                )
        return tests
    return []


def _test_from_payload(payload: Dict[str, object]) -> TestCaseResult:
    name = str(payload.get("name", "test"))
    message = str(payload.get("message", payload.get("detail", "")) or "")
    status = str(payload.get("status", "")).lower()
    skipped = bool(payload.get("skipped", False))
    passed = payload.get("passed")
    if status not in ("passed", "failed", "skipped"):
        if skipped:
            status = "skipped"
        elif isinstance(passed, bool):
            status = "passed" if passed else "failed"
        else:
            status = "failed"
    details = payload.get("details") if isinstance(payload.get("details"), dict) else None
    return TestCaseResult(name=name, status=status, message=message, details=details)


def _summary_from_payload(payload: Dict[str, object], tests: List[TestCaseResult]) -> Dict[str, int]:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return {
            "total": int(summary.get("total", len(tests))),
            "passed": int(summary.get("passed", 0)),
            "failed": int(summary.get("failed", 0)),
            "skipped": int(summary.get("skipped", 0)),
        }
    return _summarize_tests(tests)


def _summarize_tests(tests: List[TestCaseResult]) -> Dict[str, int]:
    total = len(tests)
    passed = sum(1 for test in tests if test.status == "passed")
    failed = sum(1 for test in tests if test.status == "failed")
    skipped = sum(1 for test in tests if test.status == "skipped")
    return {"total": total, "passed": passed, "failed": failed, "skipped": skipped}


def _status_from_summary(summary: Dict[str, int]) -> str:
    if summary.get("failed", 0):
        return "failed"
    if summary.get("skipped", 0):
        return "passed_with_skips"
    return "passed"


def _missing_phase_report(phase: int, title: str, message: str) -> PhaseReport:
    tests = [TestCaseResult(name="Phase runner", status="failed", message=message)]
    return PhaseReport(
        phase=phase,
        title=title,
        status="failed",
        summary=_summarize_tests(tests),
        tests=tests,
        duration_ms=0.0,
    )


def _missing_dependency_message(stdout: str, stderr: str) -> Optional[str]:
    combined = "\n".join([stderr or "", stdout or ""]).strip()
    if not combined:
        return None
    match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", combined)
    if not match:
        return None
    module = match.group(1)
    install_hint = "pip install -e ."
    if module in {"pytest", "black", "mypy", "ruff"}:
        install_hint = "pip install -e '.[dev]'"
    return (
        f"Missing dependency '{module}'. Install project requirements (e.g. {install_hint}) "
        "and re-run the validation."
    )


def _format_execution_error(cmd: List[str], returncode: int, stdout: str, stderr: str) -> str:
    missing = _missing_dependency_message(stdout, stderr)
    if missing:
        return missing
    if returncode == 0 and (stdout or stderr):
        snippet = (stderr or stdout).strip().splitlines()
        return snippet[-1] if snippet else "Phase script output not in JSON format."
    parts = [f"Command failed with exit code {returncode}."]
    if stderr.strip():
        parts.append(stderr.strip().splitlines()[-1])
    elif stdout.strip():
        parts.append(stdout.strip().splitlines()[-1])
    return " ".join(parts)


def _parse_phases(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    phases = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            phases.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid phase value: {part}") from None
    return phases or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all gloggur validation phases.")
    parser.add_argument("--phases", type=str, default=None, help="Comma-separated phase list, e.g. 1,2,4")
    parser.add_argument("--output", type=str, default=None, help="Write report to file.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (phases 1-2).")
    parser.add_argument("--serial", action="store_true", help="Disable parallel phase execution.")
    parser.add_argument("--jobs", type=int, default=None, help="Max parallel workers for phase execution.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--log-level", type=str, default=None, help="Log level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--log-file", type=str, default=None, help="Write logs to file.")
    parser.add_argument("--trace-id", type=str, default=None, help="Trace ID for log correlation.")
    args = parser.parse_args()

    configure_logging(
        debug=args.debug,
        log_level=args.log_level,
        log_file=args.log_file,
        trace_id=args.trace_id,
        stream="stderr" if args.format == "json" else None,
        force=True,
    )

    phases = _parse_phases(args.phases)
    if phases and any(phase not in PHASE_DEFINITIONS for phase in phases):
        invalid = ", ".join(str(phase) for phase in phases if phase not in PHASE_DEFINITIONS)
        raise SystemExit(f"Unknown phase(s): {invalid}")
    runner = ValidationRunner(
        phases=phases,
        quick=args.quick,
        verbose=args.verbose,
        parallel=not args.serial,
        max_workers=args.jobs,
    )
    report = runner.run_all_phases()

    output = render_markdown(report) if args.format == "markdown" else json.dumps(render_json(report), indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf8")
    else:
        print(output)

    return 0 if report.summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
