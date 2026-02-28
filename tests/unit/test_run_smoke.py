from __future__ import annotations

from pathlib import Path

from scripts.run_smoke import (
    STAGE_SPECS,
    SmokeHarness,
    StageResult,
    StageSpec,
    _execute_stage_plan,
    _parse_json_payload,
)


def test_parse_json_payload_supports_prefixed_output() -> None:
    raw = "info line before payload\n{\n  \"ok\": true,\n  \"value\": 1\n}\n"
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
                duration_ms=7,
                failure_code=spec.failure_code,
                remediation=spec.remediation,
                detail="intentional test failure",
                context={},
            )
        return StageResult(name=spec.name, status="passed", duration_ms=5, context={})

    results, failed = _execute_stage_plan(specs, runner)

    assert calls == ["a", "b"]
    assert [result.name for result in results] == ["a", "b", "c"]
    assert results[0].status == "passed"
    assert results[1].status == "failed"
    assert results[2].status == "not_run"
    assert results[2].failure_code == "blocked_by_prior_stage_failure"
    assert failed is results[1]


def test_harness_reports_missing_repo_as_stage_failure(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing-repo"
    harness = SmokeHarness(repo=missing_repo, keep_artifacts=False, timeout_seconds=1.0)

    payload = harness.run()

    assert payload["ok"] is False
    assert payload["summary"] == {
        "total_stages": len(STAGE_SPECS),
        "passed": 0,
        "failed": 1,
        "not_run": len(STAGE_SPECS) - 1,
    }
    stages = payload["stages"]
    assert isinstance(stages, list)
    assert stages[0]["name"] == "index"
    assert stages[0]["status"] == "failed"
    assert stages[0]["failure_code"] == "smoke_index_failed"
    assert stages[1]["status"] == "not_run"
    assert payload["failure"]["code"] == "smoke_index_failed"
