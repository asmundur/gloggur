from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation import CommandRunner, TestFixtures, Validators
from scripts.validation.logging_utils import configure_logging
from scripts.validation.report_templates import (
    PhaseReport,
    TestCaseResult,
    ValidationReport,
    build_validation_report,
    render_json,
    render_markdown,
)

logger = logging.getLogger(__name__)


def _format_duration(duration_ms: float) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f}s"
    return f"{int(duration_ms)}ms"


def _summarize_tests(tests: List[TestCaseResult]) -> Dict[str, int]:
    total = len(tests)
    passed = sum(1 for test in tests if test.status == "passed")
    failed = sum(1 for test in tests if test.status == "failed")
    skipped = sum(1 for test in tests if test.status == "skipped")
    return {"total": total, "passed": passed, "failed": failed, "skipped": skipped}


def _status_from_summary(summary: Dict[str, int]) -> str:
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
    return CommandRunner(
        env={"GLOGGUR_CACHE_DIR": cache_dir, "GLOGGUR_LOCAL_FALLBACK": "1"},
        default_timeout=timeout,
    )


def _cleanup_cache_dir(cache_dir: str) -> None:
    if cache_dir and Path(cache_dir).exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def _make_large_repo(fixtures: TestFixtures, file_count: int) -> Path:
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


def _parse_streaming_output(stdout: str) -> Tuple[bool, List[Dict[str, object]]]:
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
    repo = fixtures.create_temp_repo({})
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir)
    try:
        output = runner.run_index(str(repo))
        schema = Validators.validate_index_output(output)
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
    repo = fixtures.create_temp_repo({"README.txt": "just text\n", "notes.md": "# Notes\n"})
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir)
    try:
        output = runner.run_index(str(repo))
        schema = Validators.validate_index_output(output)
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
    repo = fixtures.create_temp_repo({"broken.py": "def broken(:\n    pass\n"}, validate=False)
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir)
    try:
        output = runner.run_index(str(repo))
        schema = Validators.validate_index_output(output)
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
    repo = _make_large_repo(fixtures, file_count=file_count)
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir, timeout=300.0)
    try:
        output = runner.run_index(str(repo))
        schema = Validators.validate_index_output(output)
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


def _benchmark_indexing_speed(repo_path: Path) -> Tuple[TestCaseResult, Dict[str, object]]:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir, timeout=300.0)
    try:
        output = runner.run_index(str(repo_path))
        schema = Validators.validate_index_output(output)
        if not schema.ok:
            return (
                TestCaseResult(name="Benchmark 4.1: Indexing Speed", status="failed", message=schema.message),
                {},
            )
        indexed_files = int(output.get("indexed_files", 0))
        indexed_symbols = int(output.get("indexed_symbols", 0))
        duration_ms = float(output.get("duration_ms", 0))
        symbols_per_sec = 0.0 if duration_ms <= 0 else indexed_symbols / (duration_ms / 1000)
        threshold_ms = 10_000 if indexed_files <= 200 else 120_000 if indexed_files <= 1000 else 240_000
        meets_target = duration_ms > 0 and duration_ms <= threshold_ms
        message = (
            f"{indexed_symbols} symbols in {_format_duration(duration_ms)} "
            f"= {symbols_per_sec:.1f} symbols/sec"
        )
        details = {
            "indexed_files": indexed_files,
            "indexed_symbols": indexed_symbols,
            "duration_ms": duration_ms,
            "symbols_per_sec": symbols_per_sec,
            "target_ms": threshold_ms,
        }
        status = "passed" if meets_target else "failed"
        if not meets_target:
            message = f"{message} (target {_format_duration(threshold_ms)} not met)"
        return TestCaseResult(name="Benchmark 4.1: Indexing Speed", status=status, message=message, details=details), {
            "symbols_per_sec": round(symbols_per_sec, 2),
            "indexed_files": indexed_files,
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        return (
            TestCaseResult(
                name="Benchmark 4.1: Indexing Speed",
                status="failed",
                message=f"Index command failed: {exc}",
            ),
            {},
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _benchmark_search_latency(repo_path: Path) -> Tuple[TestCaseResult, Dict[str, object]]:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir, timeout=300.0)
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
                {},
            )
        queries = ["index", "class", "function", "cache", "search"]
        timings: List[int] = []
        for query in queries:
            output = runner.run_search(query, top_k=10)
            metadata = output.get("metadata", {})
            timing = int(metadata.get("search_time_ms", 0))
            timings.append(timing)
        if not timings:
            return (
                TestCaseResult(
                    name="Benchmark 4.2: Search Latency",
                    status="failed",
                    message="No search timings captured",
                ),
                {},
            )
        avg_ms = sum(timings) / len(timings)
        min_ms = min(timings)
        max_ms = max(timings)
        meets_target = avg_ms <= 100
        message = f"Average {avg_ms:.1f}ms (min {min_ms}ms, max {max_ms}ms)"
        status = "passed" if meets_target else "failed"
        if not meets_target:
            message = f"{message} (target < 100ms not met)"
        details = {"average_ms": avg_ms, "min_ms": min_ms, "max_ms": max_ms, "samples": timings}
        return (
            TestCaseResult(
                name="Benchmark 4.2: Search Latency",
                status=status,
                message=message,
                details=details,
            ),
            {"average_ms": round(avg_ms, 2), "min_ms": min_ms, "max_ms": max_ms},
        )
    except Exception as exc:
        return (
            TestCaseResult(
                name="Benchmark 4.2: Search Latency",
                status="failed",
                message=f"Search benchmark failed: {exc}",
            ),
            {},
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _benchmark_incremental_reindex(repo_path: Path) -> Tuple[TestCaseResult, Dict[str, object]]:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    runner = _new_runner(cache_dir, timeout=300.0)
    try:
        first = runner.run_index(str(repo_path))
        second = runner.run_index(str(repo_path))
        first_ms = float(first.get("duration_ms", 0))
        second_ms = float(second.get("duration_ms", 0))
        if first_ms <= 0:
            return (
                TestCaseResult(
                    name="Benchmark 4.3: Incremental Re-index",
                    status="failed",
                    message="First index duration was zero",
                ),
                {},
            )
        ratio = second_ms / first_ms
        meets_target = ratio <= 0.1
        speedup = 0.0 if second_ms == 0 else first_ms / second_ms
        message = f"First {_format_duration(first_ms)}, second {_format_duration(second_ms)}"
        status = "passed" if meets_target else "failed"
        if meets_target:
            message = f"{message} ({speedup:.1f}x speedup)"
        else:
            message = f"{message} (target < 10% of first run not met)"
        details = {
            "first_ms": first_ms,
            "second_ms": second_ms,
            "ratio": ratio,
            "speedup": speedup,
        }
        return (
            TestCaseResult(
                name="Benchmark 4.3: Incremental Re-index",
                status=status,
                message=message,
                details=details,
            ),
            {"first_ms": first_ms, "second_ms": second_ms, "speedup": round(speedup, 2)},
        )
    except Exception as exc:
        return (
            TestCaseResult(
                name="Benchmark 4.3: Incremental Re-index",
                status="failed",
                message=f"Incremental benchmark failed: {exc}",
            ),
            {},
        )
    finally:
        _cleanup_cache_dir(cache_dir)


def _run_phase4(repo_path: Path) -> PhaseReport:
    tests: List[TestCaseResult] = []
    performance: Dict[str, object] = {}
    start = time.perf_counter()

    indexing_result, indexing_perf = _benchmark_indexing_speed(repo_path)
    tests.append(indexing_result)
    performance.update(indexing_perf)

    search_result, search_perf = _benchmark_search_latency(repo_path)
    tests.append(search_result)
    performance.update(search_perf)

    incremental_result, incremental_perf = _benchmark_incremental_reindex(repo_path)
    tests.append(incremental_result)
    performance.update(incremental_perf)

    duration_ms = (time.perf_counter() - start) * 1000
    return _build_phase_report(
        phase=4,
        title="Performance Benchmarks",
        tests=tests,
        duration_ms=duration_ms,
        performance=performance,
    )


def run_validation(skip_large_repo: bool, benchmark_only: bool) -> ValidationReport:
    logger.info("Phase 3/4 validation started (skip_large_repo=%s, benchmark_only=%s)", skip_large_repo, benchmark_only)
    phases: List[PhaseReport] = []
    if not benchmark_only:
        phases.append(_run_phase3(skip_large_repo))
    phases.append(_run_phase4(PROJECT_ROOT))
    report = build_validation_report(phases)
    logger.info("Phase 3/4 validation completed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 & 4 validation tests for gloggur.")
    parser.add_argument("--skip-large-repo", action="store_true", help="Skip large repo edge case test.")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only Phase 4 benchmarks.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--log-level", type=str, default=None, help="Log level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--log-file", type=str, default=None, help="Write logs to file.")
    parser.add_argument("--trace-id", type=str, default=None, help="Trace ID for log correlation.")
    args = parser.parse_args()

    configure_logging(
        debug=args.debug,
        log_level=args.log_level,
        log_file=args.log_file,
        trace_id=args.trace_id,
        stream="stderr" if args.format == "json" else None,
        force=True,
    )

    report = run_validation(skip_large_repo=args.skip_large_repo, benchmark_only=args.benchmark_only)
    if args.format == "json":
        print(json.dumps(render_json(report), indent=2))
    else:
        print(render_markdown(report))

    return 0 if report.summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
