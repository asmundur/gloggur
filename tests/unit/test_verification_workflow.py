from __future__ import annotations

from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 lanes.
    import tomli as tomllib


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


def _load_pyproject_text() -> str:
    pyproject_path = Path("pyproject.toml")
    return pyproject_path.read_text(encoding="utf8")


def _load_pyproject() -> dict[str, object]:
    pyproject_path = Path("pyproject.toml")
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf8"))
    assert isinstance(payload, dict)
    return payload


def test_pytest_coverage_target_points_to_runtime_src_package() -> None:
    """Pytest coverage should target src runtime package, not repo-root shim package."""
    pyproject = _load_pyproject_text()
    assert "--cov=src/gloggur" in pyproject
    assert "--cov=gloggur" not in pyproject


def test_static_tooling_excludes_shadow_worktrees_and_cache_dirs() -> None:
    """Static tooling should ignore shadow worktrees and cache directories."""
    pyproject = _load_pyproject()
    tool = pyproject.get("tool")
    assert isinstance(tool, dict)

    ruff = tool.get("ruff")
    assert isinstance(ruff, dict)
    ruff_exclude = ruff.get("extend-exclude")
    assert isinstance(ruff_exclude, list)
    assert ".claude" in ruff_exclude
    assert ".gloggur-cache" in ruff_exclude

    black = tool.get("black")
    assert isinstance(black, dict)
    black_exclude = black.get("extend-exclude")
    assert isinstance(black_exclude, str)
    assert r"\.claude" in black_exclude
    assert r"\.gloggur-cache" in black_exclude


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


def test_verification_workflow_matrix_has_no_hidden_exclusions_or_duplicates() -> None:
    """Workflow should not silently mask lanes via matrix exclusions or duplicate entries."""
    tests_job = _verification_tests_job()
    strategy = tests_job.get("strategy")
    assert isinstance(strategy, dict)

    matrix = strategy.get("matrix")
    assert isinstance(matrix, dict)
    assert "exclude" not in matrix

    include = matrix.get("include")
    assert isinstance(include, list)
    versions = []
    for lane in include:
        assert isinstance(lane, dict)
        version = lane.get("python-version")
        assert isinstance(version, str)
        versions.append(version)

    assert len(versions) == len(set(versions))


def test_verification_workflow_pytest_lane_is_unconditional() -> None:
    """Pytest execution should not be conditionally skipped for any matrix lane."""
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    pytest_step = next(
        (step for step in steps if isinstance(step, dict) and step.get("name") == "Run pytest"),
        None,
    )
    assert isinstance(pytest_step, dict)
    assert "if" not in pytest_step
    run_script = pytest_step.get("run")
    assert isinstance(run_script, str)
    assert "pytest" in run_script
    assert '-m "not performance"' in run_script


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
    assert 'python -m pip install -e ".[dev,openai,gemini]" -v' in run_script


def test_verification_workflow_includes_packaging_smoke_harness() -> None:
    """Verification workflow should run packaging smoke harness on required runtime lane."""
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    packaging_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Run packaging smoke harness"
        ),
        None,
    )
    assert isinstance(packaging_step, dict)
    assert packaging_step.get("if") == "${{ matrix.python-version == '3.13' }}"
    run_script = packaging_step.get("run")
    assert isinstance(run_script, str)
    assert "python scripts/run_packaging_smoke.py --format json" in run_script
    assert "--skip-install-smoke" not in run_script


def test_verification_workflow_includes_static_quality_gate() -> None:
    """Verification workflow should run the static-quality gate on the required 3.13 lane."""
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    static_gate_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("name") == "Run verification static quality gates"
        ),
        None,
    )
    assert isinstance(static_gate_step, dict)
    assert static_gate_step.get("if") == "${{ matrix.python-version == '3.13' }}"
    run_script = static_gate_step.get("run")
    assert isinstance(run_script, str)
    assert "python scripts/run_static_quality_gates.py --format json" in run_script


def test_verification_workflow_includes_coverage_baseline_gate_after_pytest() -> None:
    """Verification workflow should validate coverage floors on the required 3.13 lane."""
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    pytest_index = next(
        index
        for index, step in enumerate(steps)
        if isinstance(step, dict) and step.get("name") == "Run pytest"
    )
    coverage_step = steps[pytest_index + 1]
    assert isinstance(coverage_step, dict)
    assert coverage_step.get("name") == "Run coverage baseline contract"
    assert coverage_step.get("if") == "${{ matrix.python-version == '3.13' }}"
    run_script = coverage_step.get("run")
    assert isinstance(run_script, str)
    assert "python scripts/check_coverage_baseline.py --format json" in run_script


def test_verification_workflow_includes_error_catalog_contract_check() -> None:
    """Verification workflow should run the published error-catalog contract.

    The check must remain on the required lane.
    """
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    contract_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Run error catalog contract check"
        ),
        None,
    )
    assert isinstance(contract_step, dict)
    assert contract_step.get("if") == "${{ matrix.python-version == '3.13' }}"
    run_script = contract_step.get("run")
    assert isinstance(run_script, str)
    assert "python scripts/check_error_catalog_contract.py --format json" in run_script


def test_verification_workflow_includes_artifact_smoke_harness() -> None:
    """Verification workflow should run artifact smoke harness on required runtime lane."""
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    artifact_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Run artifact smoke harness"
        ),
        None,
    )
    assert isinstance(artifact_step, dict)
    assert artifact_step.get("if") == "${{ matrix.python-version == '3.13' }}"
    run_script = artifact_step.get("run")
    assert isinstance(run_script, str)
    assert "python scripts/run_artifact_smoke.py --format json" in run_script


def test_verification_workflow_excludes_performance_regression_benchmark() -> None:
    """Required verification workflow should not own non-blocking performance execution."""
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    benchmark_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Run performance regression benchmark"
        ),
        None,
    )
    assert benchmark_step is None

    upload_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("name") == "Upload performance benchmark artifact"
        ),
        None,
    )
    assert upload_step is None


def test_verification_workflow_emits_lane_reports_for_all_matrix_jobs() -> None:
    """Matrix lanes should always write and upload per-lane JSON evidence artifacts."""
    tests_job = _verification_tests_job()
    steps = tests_job.get("steps")
    assert isinstance(steps, list)

    write_report_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Write lane report"
        ),
        None,
    )
    assert isinstance(write_report_step, dict)
    assert write_report_step.get("if") == "${{ always() }}"
    env_payload = write_report_step.get("env")
    assert isinstance(env_payload, dict)
    assert env_payload.get("GLOGGUR_MATRIX_PYTHON_VERSION") == "${{ matrix.python-version }}"
    assert env_payload.get("GLOGGUR_MATRIX_REQUIRED") == "${{ matrix.required }}"
    assert env_payload.get("GLOGGUR_LANE_STATUS") == "${{ job.status }}"
    run_script = write_report_step.get("run")
    assert isinstance(run_script, str)
    assert "verification-lane-" in run_script

    upload_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Upload lane report artifact"
        ),
        None,
    )
    assert isinstance(upload_step, dict)
    assert upload_step.get("if") == "${{ always() }}"
    assert upload_step.get("uses") == "actions/upload-artifact@v4"
    upload_with = upload_step.get("with")
    assert isinstance(upload_with, dict)
    assert upload_with.get("name") == "verification-lane-${{ matrix.python-version }}"
    assert upload_with.get("if-no-files-found") == "error"


def test_verification_workflow_has_lane_audit_job_with_policy_gate() -> None:
    """Workflow should keep the lane-audit policy gate wired and always runnable."""
    payload = _load_verification_workflow()
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    lane_audit = jobs.get("lane-audit")
    assert isinstance(lane_audit, dict)
    assert lane_audit.get("needs") == "tests"
    assert lane_audit.get("if") == "${{ always() }}"

    steps = lane_audit.get("steps")
    assert isinstance(steps, list)
    download_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Download lane report artifacts"
        ),
        None,
    )
    assert isinstance(download_step, dict)
    assert download_step.get("uses") == "actions/download-artifact@v4"
    with_payload = download_step.get("with")
    assert isinstance(with_payload, dict)
    assert with_payload.get("pattern") == "verification-lane-*"
    assert with_payload.get("merge-multiple") is True

    audit_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("name") == "Audit lane coverage and required/provisional policy"
        ),
        None,
    )
    assert isinstance(audit_step, dict)
    run_script = audit_step.get("run")
    assert isinstance(run_script, str)
    assert (
        "python scripts/audit_verification_lanes.py "
        "--reports-dir verification-lane-artifacts --format json"
    ) in run_script
