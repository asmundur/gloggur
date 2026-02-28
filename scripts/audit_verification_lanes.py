from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_LANES: dict[str, bool] = {
    "3.10": True,
    "3.11": True,
    "3.12": True,
    "3.13": True,
    "3.14": False,
}

ALLOWED_STATUSES = {"success", "failure", "cancelled", "skipped"}


def _parse_required_flag(value: object) -> bool:
    """Parse required lane flag from bool/string JSON values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"invalid required flag: {value!r}")


def _load_report(path: Path) -> dict[str, object]:
    """Load one lane report JSON and validate required fields."""
    payload = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        raise ValueError(f"lane report is not an object: {path}")
    version = payload.get("python_version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"lane report missing python_version: {path}")
    status = payload.get("status")
    if not isinstance(status, str) or status not in ALLOWED_STATUSES:
        raise ValueError(f"lane report has invalid status {status!r}: {path}")
    payload["required"] = _parse_required_flag(payload.get("required"))
    return payload


def audit_lane_reports(
    reports_dir: Path,
    *,
    expected_lanes: dict[str, bool] | None = None,
) -> dict[str, object]:
    """Audit lane reports and return deterministic pass/fail summary payload."""
    expected = dict(EXPECTED_LANES if expected_lanes is None else expected_lanes)
    if not reports_dir.exists():
        raise ValueError(f"reports directory does not exist: {reports_dir}")
    if not reports_dir.is_dir():
        raise ValueError(f"reports path is not a directory: {reports_dir}")

    reports: list[dict[str, object]] = []
    by_lane: dict[str, dict[str, object]] = {}
    duplicate_lanes: list[str] = []
    malformed_reports: list[str] = []

    for report_path in sorted(reports_dir.glob("*.json")):
        try:
            report = _load_report(report_path)
        except (json.JSONDecodeError, ValueError) as exc:
            malformed_reports.append(f"{report_path.name}: {exc}")
            continue
        lane = str(report["python_version"])
        if lane in by_lane:
            duplicate_lanes.append(lane)
            continue
        by_lane[lane] = report
        reports.append(report)

    observed_lanes = set(by_lane)
    expected_lane_set = set(expected)
    missing_lanes = sorted(expected_lane_set - observed_lanes)
    unexpected_lanes = sorted(observed_lanes - expected_lane_set)

    mismatched_required: list[dict[str, object]] = []
    required_lane_failures: list[dict[str, object]] = []
    provisional_lane_failures: list[dict[str, object]] = []
    for lane, report in sorted(by_lane.items()):
        status = str(report["status"])
        observed_required = bool(report["required"])
        expected_required = bool(expected.get(lane, False))
        if lane in expected and observed_required != expected_required:
            mismatched_required.append(
                {
                    "python_version": lane,
                    "expected_required": expected_required,
                    "observed_required": observed_required,
                }
            )
        failure_payload = {
            "python_version": lane,
            "required": observed_required,
            "status": status,
        }
        if observed_required and status != "success":
            required_lane_failures.append(failure_payload)
        if (not observed_required) and status != "success":
            provisional_lane_failures.append(failure_payload)

    ok = (
        not malformed_reports
        and not duplicate_lanes
        and not missing_lanes
        and not unexpected_lanes
        and not mismatched_required
        and not required_lane_failures
    )

    return {
        "ok": ok,
        "summary": {
            "expected_lanes": sorted(expected_lane_set),
            "observed_lanes": sorted(observed_lanes),
            "total_reports": len(reports),
            "required_lane_failures": len(required_lane_failures),
            "provisional_lane_failures": len(provisional_lane_failures),
        },
        "missing_lanes": missing_lanes,
        "unexpected_lanes": unexpected_lanes,
        "duplicate_lanes": sorted(set(duplicate_lanes)),
        "malformed_reports": malformed_reports,
        "required_policy_mismatches": mismatched_required,
        "required_lane_failures": required_lane_failures,
        "provisional_lane_failures": provisional_lane_failures,
        "reports": reports,
    }


def _render_markdown(payload: dict[str, object]) -> str:
    """Render compact markdown summary for audit output."""
    lines: list[str] = ["# Verification Lane Audit", ""]
    lines.append(f"- ok: `{payload.get('ok')}`")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        lines.append(f"- expected_lanes: `{summary.get('expected_lanes')}`")
        lines.append(f"- observed_lanes: `{summary.get('observed_lanes')}`")
        lines.append(f"- required_lane_failures: `{summary.get('required_lane_failures')}`")
        lines.append(f"- provisional_lane_failures: `{summary.get('provisional_lane_failures')}`")
    for field in (
        "missing_lanes",
        "unexpected_lanes",
        "duplicate_lanes",
        "malformed_reports",
        "required_policy_mismatches",
        "required_lane_failures",
    ):
        value = payload.get(field)
        if value:
            lines.append(f"- {field}: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit GitHub Actions verification lane artifacts."
    )
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = audit_lane_reports(args.reports_dir)
    except ValueError as exc:
        payload = {
            "ok": False,
            "failure": {
                "code": "lane_audit_input_invalid",
                "detail": str(exc),
                "remediation": (
                    "Pass a valid reports directory path containing lane report JSON files."
                ),
            },
        }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload), end="")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
