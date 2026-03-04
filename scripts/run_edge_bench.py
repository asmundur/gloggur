from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verification import Checks, CommandRunner, Reporter, TestFixtures
from scripts.verification.logging_utils import configure_logging
from scripts.verification.report_templates import (
    PhaseReport,
    TestCaseResult,
    VerificationReport,
    build_verification_report,
    render_json,
    render_markdown,
)

logger = logging.getLogger(__name__)

BENCHMARK_THRESHOLD_POLICY = {
    "cold_index_duration": {"field": "duration_ms", "max_multiplier": 1.20},
    "unchanged_incremental_duration": {"field": "duration_ms", "max_multiplier": 1.25},
    "search_average_latency": {"field": "duration_ms", "max_multiplier": 1.20},
    "index_throughput": {"field": "throughput", "min_multiplier": 0.85},
}


@dataclass(frozen=True)
class BenchmarkMetric:
    """Single benchmark metric emitted by the harness."""

    name: str
    duration_ms: Optional[float] = None
    throughput: Optional[float] = None
    throughput_unit: Optional[str] = None
    details: Optional[Dict[str, object]] = None


def _format_duration(duration_ms: float) -> str:
    """Format a duration in milliseconds."""
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f}s"
    return f"{int(duration_ms)}ms"


def _summarize_tests(tests: List[TestCaseResult]) -> Dict[str, int]:
    """Summarize test results by status."""
    total = len(tests)
    passed = sum(1 for test in tests if test.status == "passed")
    failed = sum(1 for test in tests if test.status == "failed")
    skipped = sum(1 for test in tests if test.status == "skipped")
    return {"total": total, "passed": passed, "failed": failed, "skipped": skipped}


def _status_from_summary(summary: Dict[str, int]) -> str:
    """Derive phase status from summary counts."""
    if summary.get("failed", 0):
        return "failed"
    if summary.get("skipped", 0):
        return "passed_with_skips"
    return "passed"


def _build_phase_report(
    phase: int,
    title: str,
    tests: List[TestCaseResult],
    duration_ms: float,
    performance: Optional[Dict[str, object]] = None,
    issues: Optional[List[str]] = None,
) -> PhaseReport:
    """Build a phase report from test results and metrics."""
    summary = _summarize_tests(tests)
    return PhaseReport(
        phase=phase,
        title=title,
        status=_status_from_summary(summary),
        summary=summary,
        tests=tests,
        duration_ms=duration_ms,
        performance=performance,
        issues=issues,
    )


def _new_runner(cache_dir: str, timeout: float = 120.0) -> CommandRunner:
    """Create a command runner with an isolated cache directory."""
    return CommandRunner(
        env={
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        },
        default_timeout=timeout,
    )


def _cleanup_cache_dir(cache_dir: str) -> None:
    """Delete a cache directory if it exists."""
    if cache_dir and Path(cache_dir).exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def _make_large_repo(fixtures: TestFixtures, file_count: int) -> Path:
    """Create a temporary repository with many files."""
    template = "\n".join(
        [
            "def handler_{idx}(value: int) -> int:",
            "    return value + {idx}",
            "",
        ]
    )
    files = {}
    for idx in range(file_count):
        files[f"module_{idx}.py"] = template.format(idx=idx)
    return fixtures.create_temp_repo(files)


def _make_benchmark_fixture(fixtures: TestFixtures, file_count: int = 32) -> Tuple[Path, Dict[str, object]]:
    """Create a deterministic fixture repository for benchmark runs."""
    files: Dict[str, str] = {}
    for idx in range(file_count):
        files[f"pkg/module_{idx}.py"] = (
            f'"""Benchmark module {idx}."""\n\n'
            f"class BenchmarkWorker{idx}:\n"
            f'    """Process benchmark records for module {idx}."""\n'
            "    def __init__(self, seed: int) -> None:\n"
            "        self.seed = seed\n\n"
            "    def normalize(self, value: int) -> int:\n"
            '        """Normalize a benchmark value."""\n'
            "        return value + self.seed\n\n"
            "    def summarize(self, value: int) -> str:\n"
            '        """Summarize normalized benchmark output."""\n'
            "        return f'benchmark:{self.normalize(value)}'\n\n"
            f"def build_record_{idx}(value: int) -> dict[str, int]:\n"
            '    """Build a benchmark record."""\n'
            "    worker = BenchmarkWorker"
            f"{idx}(seed={idx})\n"
            "    return {'value': worker.normalize(value), 'module': worker.seed}\n\n"
            f"def describe_record_{idx}(value: int) -> str:\n"
            '    """Describe a benchmark record."""\n'
            "    return BenchmarkWorker"
            f"{idx}(seed={idx}).summarize(value)\n"
        )
    repo = fixtures.create_temp_repo(files, screen=False)
    fixture_info = {
        "kind": "generated",
        "file_count": file_count,
        "path": str(repo),
        "queries": ["normalize", "summarize", "benchmark", "record", "worker"],
    }
    return repo, fixture_info


def _parse_streaming_output(stdout: str) -> Tuple[bool, List[Dict[str, object]]]:
    """Parse line-delimited JSON output."""
    results: List[Dict[str, object]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return False, results
        if not isinstance(payload, dict):
            return False, results
        results.append(payload)
    return True, results


def _test_empty_repository(fixtures: TestFixtures) -> TestCaseResult:
    """Check indexing on an empty repository."""
    repo = fixtures.create_temp_repo({})
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir)
    try:
        output = runner.run_index(str(repo))
        schema = Checks.check_index_output(output)
        if not schema.ok:
            return TestCaseResult(name="Test 3.1: Empty Repository", status="failed", message=schema.message)
        indexed_files = int(output.get("indexed_files", 0))
        indexed_symbols = int(output.get("indexed_symbols", 0))
        if indexed_files != 0 or indexed_symbols != 0:
            details = {"indexed_files": indexed_files, "indexed_symbols": indexed_symbols}
            return TestCaseResult(
                name="Test 3.1: Empty Repository",
                status="failed",
                message="Expected zero indexed files/symbols",
                details=details,
            )
        return TestCaseResult(
            name="Test 3.1: Empty Repository",
            status="passed",
            message="Handled empty repository with zero indexed files",
        )
    except Exception as exc:
        return TestCaseResult(
            name="Test 3.1: Empty Repository",
            status="failed",
            message=f"Index command failed: {exc}",
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _test_unsupported_files(fixtures: TestFixtures) -> TestCaseResult:
    """Check indexing skips unsupported files."""
    repo = fixtures.create_temp_repo({"README.txt": "just text\n", "notes.md": "# Notes\n"})
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir)
    try:
        output = runner.run_index(str(repo))
        schema = Checks.check_index_output(output)
        if not schema.ok:
            return TestCaseResult(name="Test 3.2: Unsupported Files", status="failed", message=schema.message)
        indexed_files = int(output.get("indexed_files", 0))
        indexed_symbols = int(output.get("indexed_symbols", 0))
        if indexed_files != 0 or indexed_symbols != 0:
            details = {"indexed_files": indexed_files, "indexed_symbols": indexed_symbols}
            return TestCaseResult(
                name="Test 3.2: Unsupported Files",
                status="failed",
                message="Unsupported files should be skipped",
                details=details,
            )
        return TestCaseResult(
            name="Test 3.2: Unsupported Files",
            status="passed",
            message="Unsupported files skipped with zero indexed symbols",
        )
    except Exception as exc:
        return TestCaseResult(
            name="Test 3.2: Unsupported Files",
            status="failed",
            message=f"Index command failed: {exc}",
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _test_malformed_code(fixtures: TestFixtures) -> TestCaseResult:
    """Check indexing handles malformed code."""
    repo = fixtures.create_temp_repo({"broken.py": "def broken(:\n    pass\n"}, screen=False)
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir)
    try:
        output = runner.run_index(str(repo))
        schema = Checks.check_index_output(output)
        if not schema.ok:
            return TestCaseResult(name="Test 3.3: Malformed Code", status="failed", message=schema.message)
        return TestCaseResult(
            name="Test 3.3: Malformed Code",
            status="passed",
            message="Malformed source handled without crashing",
        )
    except Exception as exc:
        return TestCaseResult(
            name="Test 3.3: Malformed Code",
            status="failed",
            message=f"Index command failed: {exc}",
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _test_large_repository(fixtures: TestFixtures, file_count: int = 500) -> TestCaseResult:
    """Check indexing performance on a large repository."""
    repo = _make_large_repo(fixtures, file_count=file_count)
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir, timeout=300.0)
    try:
        output = runner.run_index(str(repo))
        schema = Checks.check_index_output(output)
        if not schema.ok:
            return TestCaseResult(name="Test 3.4: Large Repository", status="failed", message=schema.message)
        duration_ms = float(output.get("duration_ms", 0))
        indexed_files = int(output.get("indexed_files", 0))
        if indexed_files == 0:
            return TestCaseResult(
                name="Test 3.4: Large Repository",
                status="failed",
                message="Expected large repo to index files",
                details={"indexed_files": indexed_files},
            )
        if duration_ms > 120_000:
            return TestCaseResult(
                name="Test 3.4: Large Repository",
                status="failed",
                message="Indexing exceeded reasonable time (120s)",
                details={"duration_ms": duration_ms, "indexed_files": indexed_files},
            )
        message = f"Indexed {indexed_files} files in {_format_duration(duration_ms)}"
        return TestCaseResult(
            name="Test 3.4: Large Repository",
            status="passed",
            message=message,
            details={"duration_ms": duration_ms, "indexed_files": indexed_files},
        )
    except Exception as exc:
        return TestCaseResult(
            name="Test 3.4: Large Repository",
            status="failed",
            message=f"Index command failed: {exc}",
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _test_streaming_results(fixtures: TestFixtures) -> TestCaseResult:
    """Check search streaming output format."""
    repo = fixtures.create_temp_repo({"sample.py": fixtures.create_sample_python_file()})
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir)
    try:
        runner.run_index(str(repo))
        result = runner.run_command(
            ["search", "function", "--top-k", "10", "--stream", "--json"],
            capture_json=False,
        )
        if result.exit_code != 0:
            return TestCaseResult(
                name="Test 3.5: Streaming Results",
                status="failed",
                message=f"Search command failed: {result.stderr.strip()}",
            )
        ok, records = _parse_streaming_output(result.stdout)
        if not ok or not records:
            return TestCaseResult(
                name="Test 3.5: Streaming Results",
                status="failed",
                message="Streaming output was not line-delimited JSON",
            )
        return TestCaseResult(
            name="Test 3.5: Streaming Results",
            status="passed",
            message=f"Received {len(records)} streamed JSON results",
        )
    except Exception as exc:
        return TestCaseResult(
            name="Test 3.5: Streaming Results",
            status="failed",
            message=f"Streaming search failed: {exc}",
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _run_phase3(skip_large_repo: bool) -> PhaseReport:
    """Run phase 3 edge-case checks."""
    fixtures = TestFixtures()
    tests: List[TestCaseResult] = []
    start = time.perf_counter()
    try:
        tests.append(_test_empty_repository(fixtures))
        tests.append(_test_unsupported_files(fixtures))
        tests.append(_test_malformed_code(fixtures))
        if skip_large_repo:
            tests.append(
                TestCaseResult(
                    name="Test 3.4: Large Repository",
                    status="skipped",
                    message="Skipped by --skip-large-repo",
                )
            )
        else:
            tests.append(_test_large_repository(fixtures))
        tests.append(_test_streaming_results(fixtures))
    finally:
        fixtures.cleanup_temp_repos()
    duration_ms = (time.perf_counter() - start) * 1000
    return _build_phase_report(
        phase=3,
        title="Edge Cases",
        tests=tests,
        duration_ms=duration_ms,
    )


def _resolve_benchmark_repo(
    fixtures: TestFixtures,
    repo_path: Optional[Path],
) -> Tuple[Optional[Path], Dict[str, object], Optional[str]]:
    """Resolve the repository used for performance benchmarking."""
    if repo_path is not None:
        if not repo_path.exists() or not repo_path.is_dir():
            return None, {"kind": "custom", "path": str(repo_path)}, "performance_repo_missing"
        return repo_path, {"kind": "custom", "path": str(repo_path)}, None
    repo, fixture_info = _make_benchmark_fixture(fixtures)
    return repo, fixture_info, None


def _benchmark_indexing_speed(repo_path: Path) -> Tuple[TestCaseResult, BenchmarkMetric]:
    """Benchmark cold indexing speed and return a metric."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir, timeout=300.0)
    try:
        output = runner.run_index(str(repo_path))
        schema = Checks.check_index_output(output)
        if not schema.ok:
            return (
                TestCaseResult(name="Benchmark 4.1: Cold Index", status="failed", message=schema.message),
                BenchmarkMetric(name="cold_index_duration"),
            )
        indexed_files = int(output.get("indexed_files", 0))
        indexed_symbols = int(output.get("indexed_symbols", 0))
        duration_ms = float(output.get("duration_ms", 0))
        symbols_per_sec = 0.0 if duration_ms <= 0 else indexed_symbols / (duration_ms / 1000)
        return (
            TestCaseResult(
                name="Benchmark 4.1: Cold Index",
                status="passed",
                message=f"{indexed_symbols} symbols in {_format_duration(duration_ms)}",
                details={
                    "indexed_files": indexed_files,
                    "indexed_symbols": indexed_symbols,
                    "duration_ms": duration_ms,
                    "symbols_per_sec": symbols_per_sec,
                },
            ),
            BenchmarkMetric(
                name="cold_index_duration",
                duration_ms=duration_ms,
                details={
                    "indexed_files": indexed_files,
                    "indexed_symbols": indexed_symbols,
                },
            ),
        )
    except Exception as exc:
        return (
            TestCaseResult(
                name="Benchmark 4.1: Cold Index",
                status="failed",
                message=f"Index command failed: {exc}",
            ),
            BenchmarkMetric(name="cold_index_duration"),
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _benchmark_search_latency(repo_path: Path) -> Tuple[TestCaseResult, BenchmarkMetric]:
    """Benchmark search latency and return a metric."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir, timeout=300.0)
    queries = ["normalize", "summarize", "benchmark", "record", "worker"]
    try:
        index_output = runner.run_index(str(repo_path))
        indexed_symbols = int(index_output.get("indexed_symbols", 0))
        if indexed_symbols == 0:
            return (
                TestCaseResult(
                    name="Benchmark 4.2: Search Latency",
                    status="failed",
                    message="No symbols indexed for search benchmark",
                ),
                BenchmarkMetric(name="search_average_latency"),
            )
        timings: List[int] = []
        for query in queries:
            output = runner.run_search(query, top_k=5, debug_router=True)
            debug_payload = output.get("debug", {})
            timing = 0
            if isinstance(debug_payload, dict):
                timings_payload = debug_payload.get("timings", {})
                if isinstance(timings_payload, dict):
                    timing = int(timings_payload.get("total_ms", 0))
            timings.append(timing)
        avg_ms = sum(timings) / len(timings)
        return (
            TestCaseResult(
                name="Benchmark 4.2: Search Latency",
                status="passed",
                message=f"Average {avg_ms:.1f}ms across {len(timings)} queries",
                details={"average_ms": avg_ms, "samples": timings},
            ),
            BenchmarkMetric(
                name="search_average_latency",
                duration_ms=avg_ms,
                details={"queries": queries, "samples": timings},
            ),
        )
    except Exception as exc:
        return (
            TestCaseResult(
                name="Benchmark 4.2: Search Latency",
                status="failed",
                message=f"Search benchmark failed: {exc}",
            ),
            BenchmarkMetric(name="search_average_latency"),
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _benchmark_incremental_reindex(repo_path: Path) -> Tuple[TestCaseResult, BenchmarkMetric, BenchmarkMetric]:
    """Benchmark unchanged incremental reindex speed and throughput."""
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir, timeout=300.0)
    try:
        first = runner.run_index(str(repo_path))
        second = runner.run_index(str(repo_path))
        first_ms = float(first.get("duration_ms", 0))
        second_ms = float(second.get("duration_ms", 0))
        indexed_symbols = int(first.get("indexed_symbols", 0))
        symbols_per_sec = 0.0 if first_ms <= 0 else indexed_symbols / (first_ms / 1000)
        ratio = None if first_ms <= 0 else second_ms / first_ms
        return (
            TestCaseResult(
                name="Benchmark 4.3: Unchanged Incremental Re-index",
                status="passed",
                message=(
                    f"First {_format_duration(first_ms)}, second {_format_duration(second_ms)}"
                    if first_ms > 0
                    else "Completed unchanged incremental benchmark"
                ),
                details={
                    "first_ms": first_ms,
                    "second_ms": second_ms,
                    "ratio": ratio,
                    "symbols_per_sec": symbols_per_sec,
                },
            ),
            BenchmarkMetric(
                name="unchanged_incremental_duration",
                duration_ms=second_ms,
                details={"first_ms": first_ms, "ratio": ratio},
            ),
            BenchmarkMetric(
                name="index_throughput",
                throughput=symbols_per_sec,
                throughput_unit="symbols/s",
                details={"indexed_symbols": indexed_symbols, "cold_index_duration_ms": first_ms},
            ),
        )
    except Exception as exc:
        failure = TestCaseResult(
            name="Benchmark 4.3: Unchanged Incremental Re-index",
            status="failed",
            message=f"Incremental benchmark failed: {exc}",
        )
        return failure, BenchmarkMetric(name="unchanged_incremental_duration"), BenchmarkMetric(
            name="index_throughput"
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _phase4_performance_payload(metrics: List[BenchmarkMetric]) -> Dict[str, object]:
    """Build the phase performance payload used by render_json(report)."""
    payload: Dict[str, object] = {}
    for metric in metrics:
        if metric.name == "cold_index_duration":
            payload["cold_index_duration_ms"] = metric.duration_ms
        elif metric.name == "unchanged_incremental_duration":
            payload["unchanged_incremental_duration_ms"] = metric.duration_ms
            if metric.details:
                payload.update({k: v for k, v in metric.details.items() if k in {"first_ms", "ratio"}})
        elif metric.name == "search_average_latency":
            payload["search_average_latency_ms"] = metric.duration_ms
            if metric.details and "samples" in metric.details:
                payload["search_latency_samples_ms"] = metric.details["samples"]
        elif metric.name == "index_throughput":
            payload["index_symbols_per_second"] = metric.throughput
    return payload


def _apply_baseline_contract(
    metrics: List[BenchmarkMetric],
    baseline_payload: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Build a baseline-comparison payload using the shared Reporter machinery."""
    reporter = Reporter()
    reporter.add_section("Performance Benchmarks")

    if baseline_payload is not None:
        reporter.load_baseline_from_payload(baseline_payload)

    baseline_metrics = (
        baseline_payload.get("performance", {}).get("metrics", {})
        if isinstance(baseline_payload, dict)
        and isinstance(baseline_payload.get("performance"), dict)
        and isinstance(baseline_payload.get("performance", {}).get("metrics"), dict)
        else {}
    )

    def baseline_value(metric_name: str, field: str) -> Optional[float]:
        metric_payload = baseline_metrics.get(metric_name)
        if not isinstance(metric_payload, dict):
            return None
        value = metric_payload.get(field)
        return float(value) if isinstance(value, (int, float)) else None

    for metric in metrics:
        if metric.name == "cold_index_duration":
            max_duration = None
            baseline_duration = baseline_value(metric.name, "duration_ms")
            if baseline_duration is not None:
                max_duration = baseline_duration * 1.20
            reporter.add_performance_metric(
                metric.name,
                duration_ms=metric.duration_ms,
                thresholds=None,
            )
            if max_duration is not None:
                reporter.set_performance_thresholds(metric.name, max_duration_ms=max_duration)
        elif metric.name == "unchanged_incremental_duration":
            max_duration = None
            baseline_duration = baseline_value(metric.name, "duration_ms")
            if baseline_duration is not None:
                max_duration = baseline_duration * 1.25
            reporter.add_performance_metric(metric.name, duration_ms=metric.duration_ms)
            if max_duration is not None:
                reporter.set_performance_thresholds(metric.name, max_duration_ms=max_duration)
        elif metric.name == "search_average_latency":
            max_duration = None
            baseline_duration = baseline_value(metric.name, "duration_ms")
            if baseline_duration is not None:
                max_duration = baseline_duration * 1.20
            reporter.add_performance_metric(metric.name, duration_ms=metric.duration_ms)
            if max_duration is not None:
                reporter.set_performance_thresholds(metric.name, max_duration_ms=max_duration)
        elif metric.name == "index_throughput":
            min_throughput = None
            baseline_throughput = baseline_value(metric.name, "throughput")
            if baseline_throughput is not None:
                min_throughput = baseline_throughput * 0.85
            reporter.add_performance_metric(
                metric.name,
                throughput=metric.throughput,
                throughput_unit=metric.throughput_unit,
            )
            if min_throughput is not None:
                reporter.set_performance_thresholds(metric.name, min_throughput=min_throughput)

    reporter.add_baseline_trends()
    payload = reporter.generate_json()
    return payload.get("performance", {}) if isinstance(payload.get("performance"), dict) else {}


def _run_phase4(
    repo_path: Optional[Path],
) -> Tuple[PhaseReport, Dict[str, object], Dict[str, object], Optional[dict[str, object]]]:
    """Run performance benchmarks and return phase report, fixture info, baseline payload, and failure."""
    fixtures = TestFixtures()
    start = time.perf_counter()
    try:
        benchmark_repo, fixture_info, setup_failure = _resolve_benchmark_repo(fixtures, repo_path)
        if setup_failure is not None or benchmark_repo is None:
            tests = [
                TestCaseResult(
                    name="Benchmark setup",
                    status="failed",
                    message=f"Benchmark repository is unavailable: {fixture_info['path']}",
                )
            ]
            phase = _build_phase_report(
                phase=4,
                title="Performance Benchmarks",
                tests=tests,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            failure = {
                "code": setup_failure,
                "detail": f"Benchmark repository does not exist: {fixture_info['path']}",
            }
            return phase, fixture_info, {}, failure

        tests: List[TestCaseResult] = []
        metrics: List[BenchmarkMetric] = []

        indexing_result, cold_metric = _benchmark_indexing_speed(benchmark_repo)
        tests.append(indexing_result)
        metrics.append(cold_metric)

        search_result, search_metric = _benchmark_search_latency(benchmark_repo)
        tests.append(search_result)
        metrics.append(search_metric)

        incremental_result, unchanged_metric, throughput_metric = _benchmark_incremental_reindex(benchmark_repo)
        tests.append(incremental_result)
        metrics.extend([unchanged_metric, throughput_metric])

        phase = _build_phase_report(
            phase=4,
            title="Performance Benchmarks",
            tests=tests,
            duration_ms=(time.perf_counter() - start) * 1000,
            performance=_phase4_performance_payload(metrics),
        )
        baseline_payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fixture": fixture_info,
            "performance": {
                "metrics": {
                    metric.name: {
                        "name": metric.name,
                        "duration_ms": metric.duration_ms,
                        "throughput": metric.throughput,
                        "throughput_unit": metric.throughput_unit,
                    }
                    for metric in metrics
                }
            },
        }
        return phase, fixture_info, baseline_payload, None
    finally:
        fixtures.cleanup_temp_repos()


def run_edge_bench(
    *,
    skip_large_repo: bool,
    benchmark_only: bool,
    repo_path: Optional[Path],
) -> Tuple[VerificationReport, Dict[str, object], Dict[str, object], Optional[dict[str, object]]]:
    """Run phase 3 edge cases and/or phase 4 benchmarks."""
    logger.info(
        "Phase 3/4 run started (skip_large_repo=%s, benchmark_only=%s, repo_path=%s)",
        skip_large_repo,
        benchmark_only,
        repo_path,
    )
    phases: List[PhaseReport] = []
    fixture_info: Dict[str, object] = {}
    baseline_payload: Dict[str, object] = {}
    failure: Optional[dict[str, object]] = None

    if not benchmark_only:
        phases.append(_run_phase3(skip_large_repo))

    phase4, fixture_info, baseline_payload, failure = _run_phase4(repo_path)
    phases.append(phase4)
    report = build_verification_report(phases)
    logger.info("Phase 3/4 run completed")
    return report, fixture_info, baseline_payload, failure


def _read_json_file(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json_file(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")


def _build_output_payload(
    report: VerificationReport,
    *,
    mode: str,
    fixture_info: Dict[str, object],
    baseline_contract: Dict[str, object],
    baseline_path: Optional[Path],
    failure: Optional[dict[str, object]],
) -> Dict[str, object]:
    payload = render_json(report)
    payload["ok"] = report.summary.get("failed", 0) == 0 and failure is None
    payload["mode"] = mode
    payload["fixture"] = fixture_info
    payload["benchmark_contract"] = {
        "policy": BENCHMARK_THRESHOLD_POLICY,
        "baseline_file": str(baseline_path) if baseline_path is not None else None,
        "performance": baseline_contract,
    }
    if failure is not None:
        payload["failure"] = failure
    return payload


def main() -> int:
    """CLI entrypoint for phase 3/4 edge and bench runs."""
    parser = argparse.ArgumentParser(description="Run Phase 3 & 4 edge/bench checks for gloggur.")
    parser.add_argument("--skip-large-repo", action="store_true", help="Skip large repo edge case test.")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only Phase 4 benchmarks.")
    parser.add_argument("--baseline-file", type=Path, default=None, help="Compare benchmark results against a baseline JSON payload.")
    parser.add_argument("--write-baseline", action="store_true", help="Write the benchmark baseline payload to --baseline-file.")
    parser.add_argument("--repo", type=Path, default=None, help="Benchmark this repo instead of the deterministic generated fixture.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--log-level", type=str, default=None, help="Log level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--log-file", type=str, default=None, help="Write logs to file.")
    parser.add_argument("--trace-id", type=str, default=None, help="Trace ID for log correlation.")
    args = parser.parse_args()

    if args.write_baseline and args.baseline_file is None:
        parser.error("--write-baseline requires --baseline-file")

    configure_logging(
        debug=args.debug,
        log_level=args.log_level,
        log_file=args.log_file,
        trace_id=args.trace_id,
        stream="stderr" if args.format == "json" else None,
        force=True,
    )

    report, fixture_info, generated_baseline, failure = run_edge_bench(
        skip_large_repo=args.skip_large_repo,
        benchmark_only=args.benchmark_only,
        repo_path=args.repo,
    )

    baseline_payload: Optional[Dict[str, object]] = None
    if args.baseline_file is not None and args.baseline_file.exists():
        try:
            baseline_payload = _read_json_file(args.baseline_file)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failure = {
                "code": "performance_baseline_invalid",
                "detail": f"Unable to read baseline file {args.baseline_file}: {exc}",
            }

    if args.write_baseline and failure is None and report.summary.get("failed", 0) == 0:
        _write_json_file(args.baseline_file, generated_baseline)

    phase4_metrics = generated_baseline.get("performance", {})
    baseline_contract = _apply_baseline_contract(
        [
            BenchmarkMetric(name="cold_index_duration", duration_ms=phase4_metrics.get("metrics", {}).get("cold_index_duration", {}).get("duration_ms") if isinstance(phase4_metrics.get("metrics"), dict) else None),
            BenchmarkMetric(name="unchanged_incremental_duration", duration_ms=phase4_metrics.get("metrics", {}).get("unchanged_incremental_duration", {}).get("duration_ms") if isinstance(phase4_metrics.get("metrics"), dict) else None),
            BenchmarkMetric(name="search_average_latency", duration_ms=phase4_metrics.get("metrics", {}).get("search_average_latency", {}).get("duration_ms") if isinstance(phase4_metrics.get("metrics"), dict) else None),
            BenchmarkMetric(
                name="index_throughput",
                throughput=phase4_metrics.get("metrics", {}).get("index_throughput", {}).get("throughput") if isinstance(phase4_metrics.get("metrics"), dict) else None,
                throughput_unit=phase4_metrics.get("metrics", {}).get("index_throughput", {}).get("throughput_unit") if isinstance(phase4_metrics.get("metrics"), dict) else None,
            ),
        ],
        baseline_payload,
    )

    warnings = baseline_contract.get("warnings") if isinstance(baseline_contract, dict) else None
    if failure is None and isinstance(warnings, list) and warnings:
        failure = {
            "code": "performance_threshold_exceeded",
            "detail": "Benchmark metrics drifted beyond the allowed regression policy.",
            "warnings": warnings,
        }

    output_payload = _build_output_payload(
        report,
        mode="benchmark_only" if args.benchmark_only else "full",
        fixture_info=fixture_info,
        baseline_contract=baseline_contract,
        baseline_path=args.baseline_file,
        failure=failure,
    )

    if args.format == "json":
        print(json.dumps(output_payload, indent=2))
    else:
        print(render_markdown(report))
        if args.baseline_file is not None:
            print("\nBaseline contract:")
            print(json.dumps(output_payload["benchmark_contract"], indent=2))
        if failure is not None:
            print("\nFailure:")
            print(json.dumps(failure, indent=2))

    return 0 if output_payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
