from __future__ import annotations

from pathlib import Path

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

    assert payload["operation"] == "preflight"
    assert payload["error_code"] == "missing_package"
    assert isinstance(payload["message"], str)
    assert isinstance(payload["remediation"], list)
    assert payload["preflight_ms"] == 12
    assert "detected_environment" in payload
