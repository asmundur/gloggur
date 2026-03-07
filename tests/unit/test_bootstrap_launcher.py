from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gloggur import bootstrap_launcher


def _probe(
    candidate_type: str,
    interpreter: str,
    healthy: bool,
    exists: bool = True,
    reason: str | None = None,
) -> bootstrap_launcher.CandidateProbe:
    return bootstrap_launcher.CandidateProbe(
        candidate_type=candidate_type,
        interpreter=interpreter,
        module="gloggur.cli.main",
        exists=exists,
        healthy=healthy,
        reason=reason,
        detail=reason,
        returncode=0 if healthy else 1,
    )


def test_build_launch_plan_prefers_healthy_venv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = str(tmp_path)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env bash\n", encoding="utf8")

    def fake_probe(
        candidate_type: str,
        interpreter: str,
        module: str,
        repo_root: str,
        env: dict[str, str],
        required_imports: list[str],
    ) -> bootstrap_launcher.CandidateProbe:
        _ = module, repo_root, env, required_imports
        if candidate_type == "venv":
            return _probe(candidate_type, interpreter, healthy=True)
        return _probe(candidate_type, interpreter, healthy=False, reason="broken_environment")

    monkeypatch.setattr(bootstrap_launcher, "_probe_candidate", fake_probe)

    plan = bootstrap_launcher.build_launch_plan(
        args=["status", "--json"],
        repo_root=repo_root,
        env={},
    )

    assert plan.ready is True
    assert plan.candidate_type == "venv"
    assert plan.module == "gloggur.cli.main"
    assert plan.interpreter == str(venv_python)


def test_build_launch_plan_falls_back_to_system_when_venv_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = str(tmp_path)
    system_python = "/usr/bin/python3"
    monkeypatch.setattr(
        bootstrap_launcher,
        "_resolve_system_candidates",
        lambda _env: [system_python],
    )

    def fake_probe(
        candidate_type: str,
        interpreter: str,
        module: str,
        repo_root: str,
        env: dict[str, str],
        required_imports: list[str],
    ) -> bootstrap_launcher.CandidateProbe:
        _ = module, repo_root, env, required_imports
        if candidate_type == "venv":
            return _probe(
                candidate_type,
                interpreter,
                healthy=False,
                exists=False,
                reason="interpreter_not_found",
            )
        return _probe(candidate_type, interpreter, healthy=True)

    monkeypatch.setattr(bootstrap_launcher, "_probe_candidate", fake_probe)
    plan = bootstrap_launcher.build_launch_plan(
        args=["status", "--json"],
        repo_root=repo_root,
        env={},
    )

    assert plan.ready is True
    assert plan.candidate_type == "system"
    assert plan.module == "gloggur"
    assert plan.interpreter == system_python
    assert plan.env["PYTHONPATH"].split(":")[0] == repo_root


def test_build_launch_plan_returns_missing_python_when_no_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = str(tmp_path)
    monkeypatch.setattr(
        bootstrap_launcher,
        "_resolve_system_candidates",
        lambda _env: [],
    )

    def fake_probe(
        candidate_type: str,
        interpreter: str,
        module: str,
        repo_root: str,
        env: dict[str, str],
        required_imports: list[str],
    ) -> bootstrap_launcher.CandidateProbe:
        _ = candidate_type, module, repo_root, env, required_imports
        return _probe(
            "venv",
            interpreter,
            healthy=False,
            exists=False,
            reason="interpreter_not_found",
        )

    monkeypatch.setattr(bootstrap_launcher, "_probe_candidate", fake_probe)
    plan = bootstrap_launcher.build_launch_plan(
        args=["status", "--json"],
        repo_root=repo_root,
        env={},
    )

    assert plan.ready is False
    assert plan.error_code == "missing_python"


def test_build_launch_plan_returns_missing_venv_when_venv_missing_and_system_unhealthy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = str(tmp_path)
    monkeypatch.setattr(
        bootstrap_launcher,
        "_resolve_system_candidates",
        lambda _env: ["/usr/bin/python3"],
    )

    def fake_probe(
        candidate_type: str,
        interpreter: str,
        module: str,
        repo_root: str,
        env: dict[str, str],
        required_imports: list[str],
    ) -> bootstrap_launcher.CandidateProbe:
        _ = module, repo_root, env, required_imports
        if candidate_type == "venv":
            return _probe(
                candidate_type,
                interpreter,
                healthy=False,
                exists=False,
                reason="interpreter_not_found",
            )
        return _probe(
            candidate_type,
            interpreter,
            healthy=False,
            exists=True,
            reason="broken_environment",
        )

    monkeypatch.setattr(bootstrap_launcher, "_probe_candidate", fake_probe)
    plan = bootstrap_launcher.build_launch_plan(
        args=["status", "--json"],
        repo_root=repo_root,
        env={},
    )

    assert plan.ready is False
    assert plan.error_code == "missing_venv"


def test_build_launch_plan_returns_missing_package_when_imports_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = str(tmp_path)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("", encoding="utf8")
    monkeypatch.setattr(
        bootstrap_launcher,
        "_resolve_system_candidates",
        lambda _env: ["/usr/bin/python3"],
    )

    def fake_probe(
        candidate_type: str,
        interpreter: str,
        module: str,
        repo_root: str,
        env: dict[str, str],
        required_imports: list[str],
    ) -> bootstrap_launcher.CandidateProbe:
        _ = candidate_type, module, repo_root, env, required_imports
        return _probe("system", interpreter, healthy=False, exists=True, reason="missing_package")

    monkeypatch.setattr(bootstrap_launcher, "_probe_candidate", fake_probe)
    plan = bootstrap_launcher.build_launch_plan(
        args=["status", "--json"],
        repo_root=repo_root,
        env={},
    )

    assert plan.ready is False
    assert plan.error_code == "missing_package"


def test_build_launch_plan_returns_broken_environment_when_runtime_is_unhealthy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = str(tmp_path)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("", encoding="utf8")
    monkeypatch.setattr(
        bootstrap_launcher,
        "_resolve_system_candidates",
        lambda _env: ["/usr/bin/python3"],
    )

    def fake_probe(
        candidate_type: str,
        interpreter: str,
        module: str,
        repo_root: str,
        env: dict[str, str],
        required_imports: list[str],
    ) -> bootstrap_launcher.CandidateProbe:
        _ = candidate_type, module, repo_root, env, required_imports
        return _probe(
            "system",
            interpreter,
            healthy=False,
            exists=True,
            reason="broken_environment",
        )

    monkeypatch.setattr(bootstrap_launcher, "_probe_candidate", fake_probe)
    plan = bootstrap_launcher.build_launch_plan(
        args=["status", "--json"],
        repo_root=repo_root,
        env={},
    )

    assert plan.ready is False
    assert plan.error_code == "broken_environment"


def test_failure_payload_contains_stable_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = str(tmp_path)
    monkeypatch.setattr(
        bootstrap_launcher,
        "_resolve_system_candidates",
        lambda _env: ["/usr/bin/python3"],
    )

    def fake_probe(
        candidate_type: str,
        interpreter: str,
        module: str,
        repo_root: str,
        env: dict[str, str],
        required_imports: list[str],
    ) -> bootstrap_launcher.CandidateProbe:
        _ = candidate_type, module, repo_root, env, required_imports
        if interpreter.endswith("/python"):
            return _probe(
                "venv",
                interpreter,
                healthy=False,
                exists=False,
                reason="interpreter_not_found",
            )
        return _probe("system", interpreter, healthy=False, exists=True, reason="missing_package")

    monkeypatch.setattr(bootstrap_launcher, "_probe_candidate", fake_probe)
    plan = bootstrap_launcher.build_launch_plan(
        args=["status", "--json"],
        repo_root=repo_root,
        env={},
    )
    payload = bootstrap_launcher.build_failure_payload(plan, preflight_ms=12)

    assert payload["ok"] is False
    assert payload["stage"] == "bootstrap"
    assert payload["error_code"] == "missing_package"
    compatibility = payload.get("compatibility")
    assert isinstance(compatibility, dict)
    assert isinstance(compatibility.get("message"), str)
    assert isinstance(compatibility.get("remediation"), list)
    assert compatibility.get("preflight_ms") == 12
    assert "detected_environment" in compatibility


def test_resolve_bootstrap_status_defaults_to_degraded_optional_capabilities() -> None:
    status = bootstrap_launcher.resolve_bootstrap_status({})
    assert status.log_enabled is False
    assert status.state_enabled is False
    assert status.strict_mode is False
    assert any("BOOTSTRAP_GLOGGUR_LOG_FILE is unset" in reason for reason in status.degraded_reason)
    assert any(
        "BOOTSTRAP_GLOGGUR_STATE_FILE is unset" in reason for reason in status.degraded_reason
    )


def test_resolve_bootstrap_status_marks_capabilities_enabled_when_paths_are_writable(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "bootstrap.log"
    state_path = tmp_path / "bootstrap.state"
    status = bootstrap_launcher.resolve_bootstrap_status(
        {
            "BOOTSTRAP_GLOGGUR_LOG_FILE": str(log_path),
            "BOOTSTRAP_GLOGGUR_STATE_FILE": str(state_path),
        }
    )
    assert status.log_enabled is True
    assert status.state_enabled is True
    assert status.degraded_reason == []


def test_resolve_bootstrap_status_degrades_when_paths_are_unwritable(
    tmp_path: Path,
) -> None:
    readonly = tmp_path / "readonly"
    readonly.mkdir(parents=True, exist_ok=True)
    readonly.chmod(0o555)
    if os.access(readonly, os.W_OK):
        pytest.skip("Environment does not enforce read-only permissions for this test.")

    status = bootstrap_launcher.resolve_bootstrap_status(
        {
            "BOOTSTRAP_GLOGGUR_LOG_FILE": str(readonly / "bootstrap.log"),
            "BOOTSTRAP_GLOGGUR_STATE_FILE": str(readonly / "bootstrap.state"),
        }
    )
    assert status.log_enabled is False
    assert status.state_enabled is False
    assert len(status.degraded_reason) == 2


def test_main_strict_mode_fails_on_degraded_optional_bootstrap_status(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("GLOGGUR_BOOTSTRAP_STRICT", "1")
    monkeypatch.delenv("BOOTSTRAP_GLOGGUR_LOG_FILE", raising=False)
    monkeypatch.delenv("BOOTSTRAP_GLOGGUR_STATE_FILE", raising=False)

    exit_code = bootstrap_launcher.main(["status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["stage"] == "bootstrap"
    assert payload["error_code"] == "bootstrap_capability_degraded"


def test_resolve_system_candidates_deduplicates_and_resolves_named_interpreters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS",
        f"python3{os.pathsep}/opt/custom/python{os.pathsep}python3{os.pathsep}python",
    )
    monkeypatch.setattr(
        bootstrap_launcher.shutil,
        "which",
        lambda name: {
            "python3": "/usr/bin/python3",
            "python": "/usr/local/bin/python",
        }.get(name),
    )

    candidates = bootstrap_launcher._resolve_system_candidates(dict(os.environ))

    assert candidates == [
        "/usr/bin/python3",
        "/opt/custom/python",
        "/usr/local/bin/python",
    ]


def test_emit_ready_payload_plain_mode_reports_selected_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap_launcher._emit(
        {
            "ready": True,
            "selected_candidate": "system",
            "selected_interpreter": "/usr/bin/python3",
        },
        as_json=False,
    )

    assert "gloggur preflight OK: system -> /usr/bin/python3" in capsys.readouterr().err


def test_emit_failure_payload_plain_mode_reports_remediation_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap_launcher._emit(
        {
            "ok": False,
            "error_code": "missing_package",
            "error": "Dependencies missing.",
            "compatibility": {
                "remediation": [
                    "Run bootstrap.",
                    "Retry status.",
                ]
            },
        },
        as_json=False,
    )

    stderr = capsys.readouterr().err
    assert "gloggur preflight failed (missing_package): Dependencies missing." in stderr
    assert "1. Run bootstrap." in stderr
    assert "2. Retry status." in stderr


def test_main_dry_run_emits_ready_payload_without_exec(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = bootstrap_launcher.BootstrapStatus(
        log_enabled=True,
        state_enabled=True,
        degraded_reason=[],
        strict_mode=False,
        log_path="/tmp/bootstrap.log",
        state_path="/tmp/bootstrap.state.json",
    )
    plan = bootstrap_launcher.LaunchPlan(
        ready=True,
        args=["status", "--json"],
        interpreter="/usr/bin/python3",
        module="gloggur",
        candidate_type="system",
        env={"PYTHONPATH": "/tmp/repo"},
        repo_root="/tmp/repo",
        probes=[],
    )
    execute_calls: list[bootstrap_launcher.LaunchPlan] = []

    monkeypatch.setenv("GLOGGUR_PREFLIGHT_DRY_RUN", "1")
    monkeypatch.setattr(bootstrap_launcher, "resolve_bootstrap_status", lambda: status)
    monkeypatch.setattr(bootstrap_launcher, "build_launch_plan", lambda args: plan)
    monkeypatch.setattr(
        bootstrap_launcher, "_execute", lambda current_plan: execute_calls.append(current_plan)
    )

    exit_code = bootstrap_launcher.main(["status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert execute_calls == []
    assert payload["ready"] is True
    assert payload["selected_candidate"] == "system"
    assert payload["bootstrap_status"]["log_enabled"] is True


def test_main_executes_ready_plan_when_not_in_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = bootstrap_launcher.BootstrapStatus(
        log_enabled=False,
        state_enabled=False,
        degraded_reason=[],
        strict_mode=False,
    )
    plan = bootstrap_launcher.LaunchPlan(
        ready=True,
        args=["status"],
        interpreter="/usr/bin/python3",
        module="gloggur",
        candidate_type="system",
        env={"PYTHONPATH": "/tmp/repo"},
        repo_root="/tmp/repo",
        probes=[],
    )
    execute_calls: list[bootstrap_launcher.LaunchPlan] = []

    monkeypatch.delenv("GLOGGUR_PREFLIGHT_DRY_RUN", raising=False)
    monkeypatch.setattr(bootstrap_launcher, "resolve_bootstrap_status", lambda: status)
    monkeypatch.setattr(bootstrap_launcher, "build_launch_plan", lambda args: plan)
    monkeypatch.setattr(
        bootstrap_launcher, "_execute", lambda current_plan: execute_calls.append(current_plan)
    )

    exit_code = bootstrap_launcher.main(["status"])

    assert exit_code == 0
    assert execute_calls == [plan]


def test_main_maps_unexpected_preflight_exception_to_broken_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        bootstrap_launcher,
        "resolve_bootstrap_status",
        lambda: bootstrap_launcher.BootstrapStatus(
            log_enabled=False,
            state_enabled=False,
            degraded_reason=[],
            strict_mode=False,
        ),
    )
    monkeypatch.setattr(bootstrap_launcher, "_repo_root", lambda: "/tmp/repo")
    monkeypatch.setattr(
        bootstrap_launcher,
        "build_launch_plan",
        lambda args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code = bootstrap_launcher.main(["status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == bootstrap_launcher.EXIT_CODES["broken_environment"]
    assert payload["error_code"] == "broken_environment"
    compatibility = payload["compatibility"]
    assert compatibility["detected_environment"]["repo_root"] == "/tmp/repo"
    assert "RuntimeError('boom')" in compatibility["detected_environment"]["exception"]
