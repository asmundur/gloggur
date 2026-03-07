from __future__ import annotations

from pathlib import Path

import yaml


def _load_performance_workflow() -> dict[str, object]:
    workflow_path = Path(".github/workflows/performance.yml")
    payload = yaml.safe_load(workflow_path.read_text(encoding="utf8"))
    assert isinstance(payload, dict)
    return payload


def _performance_job() -> dict[str, object]:
    payload = _load_performance_workflow()
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    job = jobs.get("performance")
    assert isinstance(job, dict)
    return job


def _workflow_on_block(payload: dict[str, object]) -> dict[str, object]:
    on_block = payload.get("on", payload.get(True))
    assert isinstance(on_block, dict)
    return on_block


def test_performance_workflow_trigger_policy_is_stable() -> None:
    """Performance workflow should run on PRs, main/develop pushes, and manual dispatch."""
    payload = _load_performance_workflow()
    on_block = _workflow_on_block(payload)

    push = on_block.get("push")
    assert isinstance(push, dict)
    assert push.get("branches") == ["main", "develop"]

    pull_request = on_block.get("pull_request")
    assert isinstance(pull_request, dict)
    assert pull_request.get("branches") == ["main"]

    assert "workflow_dispatch" in on_block


def test_performance_workflow_runs_non_blocking_benchmark_and_pytest() -> None:
    """Performance workflow should own non-blocking benchmark and perf-marked pytest execution."""
    performance_job = _performance_job()
    assert performance_job.get("runs-on") == "ubuntu-latest"

    steps = performance_job.get("steps")
    assert isinstance(steps, list)

    setup_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Set up Python"
        ),
        None,
    )
    assert isinstance(setup_step, dict)
    setup_with = setup_step.get("with")
    assert isinstance(setup_with, dict)
    assert setup_with.get("python-version") == "3.13"

    benchmark_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Run performance regression benchmark"
        ),
        None,
    )
    assert isinstance(benchmark_step, dict)
    assert benchmark_step.get("continue-on-error") is True
    benchmark_env = benchmark_step.get("env")
    assert isinstance(benchmark_env, dict)
    assert benchmark_env.get("GLOGGUR_EMBEDDING_PROVIDER") == "test"
    benchmark_run = benchmark_step.get("run")
    assert isinstance(benchmark_run, str)
    assert "set -o pipefail" in benchmark_run
    assert (
        "python scripts/run_edge_bench.py --benchmark-only "
        "--baseline-file benchmarks/performance_baseline.json --format json"
    ) in benchmark_run
    assert "tee performance-benchmark.json" in benchmark_run

    pytest_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Run performance pytest"
        ),
        None,
    )
    assert isinstance(pytest_step, dict)
    assert pytest_step.get("continue-on-error") is True
    pytest_run = pytest_step.get("run")
    assert isinstance(pytest_run, str)
    assert "pytest -m performance" in pytest_run
    assert "--no-cov" in pytest_run
    assert "--junitxml performance-pytest.junit.xml" in pytest_run


def test_performance_workflow_uploads_artifacts_and_publishes_summary() -> None:
    """Performance workflow should preserve artifacts and job-summary triage output."""
    performance_job = _performance_job()
    steps = performance_job.get("steps")
    assert isinstance(steps, list)

    upload_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Upload performance artifacts"
        ),
        None,
    )
    assert isinstance(upload_step, dict)
    assert upload_step.get("if") == "${{ always() }}"
    assert upload_step.get("uses") == "actions/upload-artifact@v4"
    upload_with = upload_step.get("with")
    assert isinstance(upload_with, dict)
    assert upload_with.get("name") == "performance-artifacts"
    upload_path = upload_with.get("path")
    assert isinstance(upload_path, str)
    assert "performance-benchmark.json" in upload_path
    assert "performance-pytest.junit.xml" in upload_path
    assert upload_with.get("if-no-files-found") == "warn"

    summary_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "Publish performance summary"
        ),
        None,
    )
    assert isinstance(summary_step, dict)
    assert summary_step.get("if") == "${{ always() }}"
    summary_run = summary_step.get("run")
    assert isinstance(summary_run, str)
    assert "steps.benchmark.outcome" in summary_run
    assert "steps.performance_pytest.outcome" in summary_run
    assert "performance-artifacts" in summary_run
