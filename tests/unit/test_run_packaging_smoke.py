from __future__ import annotations

from pathlib import Path

from scripts.run_packaging_smoke import (
    BASE_STAGE_SPECS,
    PackagingSmokeHarness,
    StageResult,
    StageSpec,
    _execute_stage_plan,
    _parse_json_payload,
    _stage_specs,
)


def test_parse_json_payload_supports_prefixed_output() -> None:
    raw = "note line\n{\n  \"ok\": true,\n  \"value\": 1\n}\n"
    payload = _parse_json_payload(raw)
    assert payload == {"ok": True, "value": 1}


def test_execute_stage_plan_blocks_following_stages_after_failure() -> None:
    specs = [
        StageSpec("a", "a_failed", "fix a"),
        StageSpec("b", "b_failed", "fix b"),
        StageSpec("c", "c_failed", "fix c"),
    ]
    calls: list[str] = []

    def runner(spec: StageSpec) -> StageResult:
        calls.append(spec.name)
        if spec.name == "b":
            return StageResult(
                name=spec.name,
                status="failed",
                duration_ms=5,
                failure_code=spec.failure_code,
                remediation=spec.remediation,
                detail="intentional test failure",
                context={},
            )
        return StageResult(name=spec.name, status="passed", duration_ms=2, context={})

    results, failed = _execute_stage_plan(specs, runner)

    assert calls == ["a", "b"]
    assert [result.name for result in results] == ["a", "b", "c"]
    assert results[2].status == "not_run"
    assert results[2].failure_code == "blocked_by_prior_stage_failure"
    assert failed is results[1]


def test_stage_specs_support_build_only_mode() -> None:
    specs = _stage_specs(skip_install_smoke=True)
    assert [spec.name for spec in specs] == ["build_artifacts"]

    full_specs = _stage_specs(skip_install_smoke=False)
    assert [spec.name for spec in full_specs] == [spec.name for spec in BASE_STAGE_SPECS]


def test_harness_reports_missing_repo_as_stage_failure(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing-repo"
    harness = PackagingSmokeHarness(
        repo=missing_repo,
        keep_artifacts=False,
        timeout_seconds=1.0,
        skip_install_smoke=True,
    )

    payload = harness.run()

    assert payload["ok"] is False
    assert payload["summary"] == {
        "total_stages": 1,
        "passed": 0,
        "failed": 1,
        "not_run": 0,
    }
    stages = payload["stages"]
    assert isinstance(stages, list)
    assert stages[0]["name"] == "build_artifacts"
    assert stages[0]["status"] == "failed"
    assert stages[0]["failure_code"] == "packaging_build_failed"
    assert payload["failure"]["code"] == "packaging_build_failed"


def test_harness_reports_missing_pyproject_as_stage_failure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    harness = PackagingSmokeHarness(
        repo=repo,
        keep_artifacts=False,
        timeout_seconds=1.0,
        skip_install_smoke=True,
    )
    payload = harness.run()

    assert payload["ok"] is False
    stages = payload["stages"]
    assert isinstance(stages, list)
    assert stages[0]["name"] == "build_artifacts"
    assert stages[0]["failure_code"] == "packaging_build_failed"
