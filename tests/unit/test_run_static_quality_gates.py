from __future__ import annotations

import subprocess

import pytest

from scripts import run_static_quality_gates as static_gates


def test_execute_stage_plan_blocks_downstream_stages_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner should mark later stages as not_run after the first tool failure."""

    def fake_run(
        command: list[str],
        *,
        cwd: object,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is False
        if "ruff" in command:
            return subprocess.CompletedProcess(command, 0, stdout="ruff ok\n", stderr="")
        if "mypy" in command:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="mypy broke\n")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(static_gates.subprocess, "run", fake_run)

    specs = [
        static_gates.StageSpec("ruff", "ruff_failed", "fix ruff", ["python", "-m", "ruff"]),
        static_gates.StageSpec("mypy", "mypy_failed", "fix mypy", ["python", "-m", "mypy"]),
        static_gates.StageSpec("black", "black_failed", "fix black", ["python", "-m", "black"]),
    ]

    stages, failed = static_gates._execute_stage_plan(specs)

    assert [stage.status for stage in stages] == ["passed", "failed", "not_run"]
    assert failed is stages[1]
    assert stages[2].failure_code == "blocked_by_prior_stage_failure"


def test_run_static_quality_gates_fails_loud_when_targets_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner should return a structured setup failure when a gated file is missing."""
    monkeypatch.setattr(static_gates, "_missing_targets", lambda: ["scripts/missing_gate.py"])

    payload = static_gates.run_static_quality_gates()

    assert payload["ok"] is False
    failure = payload["failure"]
    assert isinstance(failure, dict)
    assert failure["code"] == "static_gate_targets_missing"
    assert failure["missing_targets"] == ["scripts/missing_gate.py"]
    stages = payload["stages"]
    assert isinstance(stages, list)
    assert stages[0]["status"] == "failed"
    assert stages[1]["status"] == "not_run"
    assert stages[2]["status"] == "not_run"


def test_static_gate_target_scope_keeps_runtime_package_in_ruff_and_black_only() -> None:
    """Runtime package should stay in lint/format scope until mypy debt is cleared."""
    assert static_gates.RUNTIME_PACKAGE_DIR == "src/gloggur"
    assert static_gates.GATE_TARGETS == [
        *static_gates.CONTROL_PLANE_TARGETS,
        static_gates.RUNTIME_PACKAGE_DIR,
    ]
    assert static_gates.MYPY_TARGETS == [
        "scripts/audit_verification_lanes.py",
        "scripts/check_error_catalog_contract.py",
        "scripts/run_static_quality_gates.py",
    ]

    ruff_stage, mypy_stage, black_stage = static_gates.STAGE_SPECS

    assert ruff_stage.name == "ruff"
    assert ruff_stage.command[-1] == static_gates.RUNTIME_PACKAGE_DIR

    assert mypy_stage.name == "mypy"
    assert static_gates.RUNTIME_PACKAGE_DIR not in mypy_stage.command
    assert mypy_stage.command[-3:] == static_gates.MYPY_TARGETS

    assert black_stage.name == "black"
    assert black_stage.command[-1] == static_gates.RUNTIME_PACKAGE_DIR
