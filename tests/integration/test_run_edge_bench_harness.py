from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_edge_bench(args: list[str], *, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "scripts/run_edge_bench.py", "--format", "json", *args]
    return subprocess.run(
        command,
        cwd=str(repo_root),
        env={**os.environ, "GLOGGUR_LOCAL_FALLBACK": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_run_edge_bench_benchmark_only_passes_on_generated_fixture() -> None:
    completed = _run_edge_bench(["--benchmark-only"])
    assert completed.returncode == 0, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["ok"] is True
    assert payload["mode"] == "benchmark_only"
    fixture = payload.get("fixture", {})
    assert fixture["kind"] == "generated"
    assert fixture["file_count"] == 32

    phases = payload.get("phases", [])
    assert isinstance(phases, list)
    assert len(phases) == 1
    phase = phases[0]
    assert phase["phase"] == 4
    assert [test["status"] for test in phase["tests"]] == ["passed", "passed", "passed"]


def test_run_edge_bench_writes_and_reuses_baseline_file(tmp_path: Path) -> None:
    baseline_path = tmp_path / "performance-baseline.json"

    first = _run_edge_bench(
        ["--benchmark-only", "--baseline-file", str(baseline_path), "--write-baseline"]
    )
    assert first.returncode == 0, f"{first.stderr}\n{first.stdout}"
    assert baseline_path.exists()

    second = _run_edge_bench(["--benchmark-only", "--baseline-file", str(baseline_path)])
    assert second.returncode in (0, 1), f"{second.stderr}\n{second.stdout}"
    payload = json.loads(second.stdout)

    assert payload["benchmark_contract"]["baseline_file"] == str(baseline_path)
    performance = payload["benchmark_contract"]["performance"]
    comparisons = performance["comparisons"]
    assert "cold_index_duration" in comparisons
    assert "search_average_latency" in comparisons
    assert performance["baseline"]["cold_index_duration"]["duration_ms"] is not None
    if second.returncode == 1:
        assert payload["failure"]["code"] == "performance_threshold_exceeded"


def test_run_edge_bench_reports_missing_repo_failure_code(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing-repo"
    completed = _run_edge_bench(["--benchmark-only", "--repo", str(missing_repo)])
    assert completed.returncode == 1, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["ok"] is False
    assert payload["failure"]["code"] == "performance_repo_missing"


def test_run_edge_bench_reports_threshold_failure_code(tmp_path: Path) -> None:
    baseline_path = tmp_path / "too-fast-baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "performance": {
                    "metrics": {
                        "cold_index_duration": {"duration_ms": 1.0},
                        "unchanged_incremental_duration": {"duration_ms": 1.0},
                        "search_average_latency": {"duration_ms": 1.0},
                        "index_throughput": {"throughput": 100000.0, "throughput_unit": "symbols/s"},
                    }
                }
            }
        ),
        encoding="utf8",
    )

    completed = _run_edge_bench(["--benchmark-only", "--baseline-file", str(baseline_path)])
    assert completed.returncode == 1, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["failure"]["code"] == "performance_threshold_exceeded"
    warnings = payload["failure"]["warnings"]
    assert isinstance(warnings, list)
    assert warnings
