from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_verification_lanes import audit_lane_reports


def _write_lane_report(
    reports_dir: Path,
    *,
    version: str,
    required: bool | str,
    status: str,
) -> None:
    payload = {
        "python_version": version,
        "required": required,
        "status": status,
    }
    path = reports_dir / f"verification-lane-{version}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")


def test_audit_lane_reports_passes_with_expected_lanes_and_required_success(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for version, required in (
        ("3.10", True),
        ("3.11", True),
        ("3.12", True),
        ("3.13", True),
        ("3.14", False),
    ):
        status = "success" if required else "failure"
        _write_lane_report(reports_dir, version=version, required=required, status=status)

    payload = audit_lane_reports(reports_dir)

    assert payload["ok"] is True
    assert payload["missing_lanes"] == []
    assert payload["unexpected_lanes"] == []
    assert payload["required_lane_failures"] == []
    assert len(payload["provisional_lane_failures"]) == 1


def test_audit_lane_reports_fails_when_lane_is_missing(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for version in ("3.10", "3.11", "3.12", "3.13"):
        _write_lane_report(reports_dir, version=version, required=True, status="success")

    payload = audit_lane_reports(reports_dir)

    assert payload["ok"] is False
    assert payload["missing_lanes"] == ["3.14"]


def test_audit_lane_reports_fails_when_required_lane_is_not_success(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_lane_report(reports_dir, version="3.10", required=True, status="failure")
    _write_lane_report(reports_dir, version="3.11", required=True, status="success")
    _write_lane_report(reports_dir, version="3.12", required=True, status="success")
    _write_lane_report(reports_dir, version="3.13", required=True, status="success")
    _write_lane_report(reports_dir, version="3.14", required=False, status="success")

    payload = audit_lane_reports(reports_dir)

    assert payload["ok"] is False
    failures = payload["required_lane_failures"]
    assert len(failures) == 1
    assert failures[0]["python_version"] == "3.10"


def test_audit_lane_reports_accepts_string_required_flags(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_lane_report(reports_dir, version="3.10", required="true", status="success")
    _write_lane_report(reports_dir, version="3.11", required="true", status="success")
    _write_lane_report(reports_dir, version="3.12", required="true", status="success")
    _write_lane_report(reports_dir, version="3.13", required="yes", status="success")
    _write_lane_report(reports_dir, version="3.14", required="off", status="failure")

    payload = audit_lane_reports(reports_dir)

    assert payload["ok"] is True
    assert payload["required_lane_failures"] == []
    assert len(payload["provisional_lane_failures"]) == 1


def test_audit_lane_reports_fails_on_invalid_required_flag(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_lane_report(reports_dir, version="3.10", required="sometimes", status="success")

    payload = audit_lane_reports(reports_dir)

    assert payload["ok"] is False
    malformed = payload["malformed_reports"]
    assert malformed
    assert "invalid required flag" in malformed[0]
