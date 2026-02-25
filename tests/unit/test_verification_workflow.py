from __future__ import annotations

from pathlib import Path

import yaml


def _load_verification_workflow() -> dict[str, object]:
    workflow_path = Path(".github/workflows/verification.yml")
    payload = yaml.safe_load(workflow_path.read_text(encoding="utf8"))
    assert isinstance(payload, dict)
    return payload


def _verification_tests_job() -> dict[str, object]:
    payload = _load_verification_workflow()
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    tests_job = jobs.get("tests")
    assert isinstance(tests_job, dict)
    return tests_job


def test_verification_workflow_python_matrix_policy_is_stable() -> None:
    """Workflow should preserve required/provisional lane policy and non-masking behavior."""
    tests_job = _verification_tests_job()

    continue_on_error = tests_job.get("continue-on-error")
    assert continue_on_error == "${{ !matrix.required }}"

    strategy = tests_job.get("strategy")
    assert isinstance(strategy, dict)
    assert strategy.get("fail-fast") is False

    matrix = strategy.get("matrix")
    assert isinstance(matrix, dict)
    include = matrix.get("include")
    assert isinstance(include, list)

    required_versions: set[str] = set()
    provisional_versions: set[str] = set()
    for lane in include:
        assert isinstance(lane, dict)
        version = lane.get("python-version")
        required = lane.get("required")
        assert isinstance(version, str)
        assert isinstance(required, bool)
        if required:
            required_versions.add(version)
        else:
            provisional_versions.add(version)

    assert required_versions == {"3.10", "3.11", "3.12", "3.13"}
    assert provisional_versions == {"3.14"}


def test_verification_workflow_includes_runtime_and_resolver_diagnostics() -> None:
    """Install step should emit interpreter/pip diagnostics for failure triage."""
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    install_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Install dependencies"
        ),
        None,
    )
    assert isinstance(install_step, dict)
    run_script = install_step.get("run")
    assert isinstance(run_script, str)
    assert "python --version" in run_script
    assert "python -m pip --version" in run_script
    assert "python -m pip debug --verbose" in run_script
    assert "python -m pip install -e \".[dev,openai,gemini]\" -v" in run_script
