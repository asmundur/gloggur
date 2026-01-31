from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class TestCaseResult:
    name: str
    status: str
    message: str
    details: Optional[Dict[str, object]] = None


@dataclass
class PhaseReport:
    phase: int
    title: str
    status: str
    summary: Dict[str, int]
    tests: List[TestCaseResult]
    duration_ms: float = 0.0
    performance: Optional[Dict[str, object]] = None
    issues: Optional[List[str]] = None


@dataclass
class ValidationReport:
    generated_at: str
    status: str
    summary: Dict[str, int]
    phases: List[PhaseReport]
    performance_summary: Dict[str, object]
    issues: List[str]
    recommendation: str


def build_validation_report(
    phases: List[PhaseReport], generated_at: Optional[str] = None
) -> ValidationReport:
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    for phase in phases:
        summary = phase.summary
        total += summary.get("total", len(phase.tests))
        passed += summary.get("passed", 0)
        failed += summary.get("failed", 0)
        skipped += summary.get("skipped", 0)

    if failed:
        status = "failed"
    elif skipped:
        status = "passed_with_skips"
    else:
        status = "passed"

    issues: List[str] = []
    for phase in phases:
        if phase.issues:
            issues.extend(phase.issues)
        for test in phase.tests:
            if test.status == "failed":
                issues.append(f"Phase {phase.phase} - {test.name}: {test.message}")

    performance_summary: Dict[str, object] = {}
    for phase in phases:
        if phase.performance:
            performance_summary[f"Phase {phase.phase}: {phase.title}"] = phase.performance

    if status == "failed":
        recommendation = "Needs fixes before production use."
    elif status == "passed_with_skips":
        recommendation = "Ready with skipped tests; complete skipped phases before release."
    else:
        recommendation = "Ready for production use."

    return ValidationReport(
        generated_at=generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status=status,
        summary={"total": total, "passed": passed, "failed": failed, "skipped": skipped},
        phases=phases,
        performance_summary=performance_summary,
        issues=issues,
        recommendation=recommendation,
    )


def render_markdown(report: ValidationReport) -> str:
    total = report.summary["total"]
    passed = report.summary["passed"]
    failed = report.summary["failed"]
    skipped = report.summary["skipped"]
    status_label = _status_label(report.status, total, passed, failed, skipped)

    lines = [
        "# Gloggur Validation Report",
        "",
        f"**Generated:** {report.generated_at}",
        f"**Status:** {status_label}",
        "",
        "## Executive Summary",
        "",
        f"- Total tests: {total}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Skipped: {skipped}",
        f"- Recommendation: {report.recommendation}",
        "",
    ]

    for phase in report.phases:
        phase_summary = phase.summary
        phase_label = _phase_summary_label(phase, phase_summary)
        lines.append(f"## Phase {phase.phase}: {phase.title}")
        lines.append("")
        lines.append(phase_label)
        lines.append("")
        for test in phase.tests:
            lines.append(_test_line(test))
        lines.append("")

    lines.append("## Performance Summary")
    lines.append("")
    if report.performance_summary:
        for phase_name, metrics in report.performance_summary.items():
            lines.append(f"- {phase_name}")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    lines.append(f"  - {key}: {value}")
            else:
                lines.append(f"  - {metrics}")
    else:
        lines.append("- No performance metrics captured.")
    lines.append("")

    lines.append("## Issues & Recommendations")
    lines.append("")
    if report.issues:
        for item in report.issues:
            lines.append(f"- {item}")
    else:
        lines.append("- No issues reported.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_json(report: ValidationReport) -> Dict[str, object]:
    return {
        "generated_at": report.generated_at,
        "status": report.status,
        "summary": report.summary,
        "recommendation": report.recommendation,
        "issues": report.issues,
        "performance_summary": report.performance_summary,
        "phases": [
            {
                "phase": phase.phase,
                "title": phase.title,
                "status": phase.status,
                "summary": phase.summary,
                "duration_ms": phase.duration_ms,
                "performance": phase.performance,
                "issues": phase.issues,
                "tests": [
                    {
                        "name": test.name,
                        "status": test.status,
                        "message": test.message,
                        "details": test.details,
                    }
                    for test in phase.tests
                ],
            }
            for phase in report.phases
        ],
    }


def _status_label(status: str, total: int, passed: int, failed: int, skipped: int) -> str:
    if status == "failed":
        return f"FAILED ({passed}/{total} passed, {failed} failed)"
    if status == "passed_with_skips":
        return f"PASSED ({passed}/{total} passed, {skipped} skipped)"
    return f"PASSED ({passed}/{total} passed)"


def _phase_summary_label(phase: PhaseReport, summary: Dict[str, int]) -> str:
    total = summary.get("total", len(phase.tests))
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    if failed:
        return f"[FAIL] {passed}/{total} passed, {failed} failed"
    if skipped:
        return f"[PASS] {passed}/{total} passed, {skipped} skipped"
    return f"[PASS] {passed}/{total} passed"


def _test_line(test: TestCaseResult) -> str:
    if test.status == "failed":
        prefix = "[FAIL]"
    elif test.status == "skipped":
        prefix = "[SKIP]"
    else:
        prefix = "[PASS]"
    return f"- {prefix} {test.name}: {test.message}"
