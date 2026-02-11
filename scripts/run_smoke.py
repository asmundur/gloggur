from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gloggur.indexer.cache import CacheConfig, CacheManager
from scripts.verification import CommandRunner, Reporter, RetryConfig, TestFixtures, TestResult, Checks
from scripts.verification.logging_utils import configure_logging

logger = logging.getLogger(__name__)


@dataclass
class Phase1Result:
    """Result wrapper for phase 1 tests."""
    name: str
    result: TestResult


def _format_duration(duration_ms: float | int) -> str:
    """Format a duration in milliseconds."""
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f}s"
    return f"{int(duration_ms)}ms"


def _count_symbols(db_path: str) -> int:
    """Count symbols in the index database."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
        return int(row[0]) if row else 0


def _load_baseline_payload(path: Optional[str]) -> Optional[Dict[str, object]]:
    """Load a baseline JSON payload from disk."""
    if not path:
        return None
    baseline_path = Path(path)
    if not baseline_path.exists():
        return None
    try:
        return json.loads(baseline_path.read_text(encoding="utf8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Read a float from an environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from an environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _check_required_fields(results: List[Dict[str, object]]) -> Optional[str]:
    """Check search results contain required fields."""
    required = {"symbol", "kind", "file", "line", "signature", "similarity_score"}
    for idx, item in enumerate(results):
        missing = [field for field in required if field not in item]
        if missing:
            return f"Result {idx} missing fields: {', '.join(missing)}"
    return None


def _create_search_fixture() -> Tuple[Path, Path, str]:
    """Create a temporary file for search tests."""
    fixture_dir = Path(tempfile.mkdtemp(prefix="gloggur-search-"))
    fixture_file = fixture_dir / "search_fixture.py"
    content = textwrap.dedent(
        """\
        \"\"\"Temporary search fixture to exercise filters.\"\"\"

        def search_fixture_function() -> str:
            \"\"\"Return a fixture-specific payload for search.\"\"\"
            return "search fixture function"

        def helper_function() -> int:
            \"\"\"Helper that increases symbol count.\"\"\"
            return 42
        """
    ).strip() + "\n"
    fixture_file.write_text(content, encoding="utf8")
    query = "search fixture function"
    return fixture_dir, fixture_file, query


def _create_incremental_target_fixture() -> Path:
    """Create an ephemeral fixture file for incremental indexing checks."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf8",
        suffix=".py",
        prefix=".gloggur-phase1-",
        dir=str(PROJECT_ROOT),
        delete=False,
    )
    with handle:
        handle.write(
            textwrap.dedent(
                """\
                \"\"\"Ephemeral fixture for phase 1 incremental indexing checks.\"\"\"

                def phase1_incremental_target() -> int:
                    \"\"\"Return a stable integer value.\"\"\"
                    return 7
                """
            ).strip()
            + "\n"
        )
    return Path(handle.name)


def test_basic_indexing(runner: CommandRunner, cache_dir: str) -> Tuple[TestResult, Dict[str, object]]:
    """Smoke test for indexing a repository."""
    try:
        output = runner.run_index(".", timeout=300.0)
    except Exception as exc:
        return TestResult(passed=False, message=f"Index command failed: {exc}"), {}

    schema = Checks.check_index_output(output)
    if not schema.ok:
        return TestResult(passed=False, message=schema.message, details=schema.details), output

    indexed_files = int(output.get("indexed_files", 0))
    indexed_symbols = int(output.get("indexed_symbols", 0))
    duration_ms = float(output.get("duration_ms", 0))

    if indexed_files <= 0:
        return TestResult(passed=False, message="No files indexed", details=output), output
    if indexed_symbols <= 0:
        return TestResult(passed=False, message="No symbols indexed", details=output), output

    cache_check = Checks.check_cache_exists(cache_dir)
    if not cache_check.ok:
        return TestResult(passed=False, message=cache_check.message, details=cache_check.details), output

    db_path = os.path.join(cache_dir, "index.db")
    db_check = Checks.check_database_symbols(db_path, 1)
    if not db_check.ok:
        return TestResult(passed=False, message=db_check.message, details=db_check.details), output

    db_count = _count_symbols(db_path)
    if db_count != indexed_symbols:
        return (
            TestResult(
                passed=False,
                message="Database symbol count does not match output",
                details={"db_count": db_count, "output": output},
            ),
            output,
        )

    message = f"Indexed {indexed_files} files, {indexed_symbols} symbols in {_format_duration(duration_ms)}"
    details = {"indexed_files": indexed_files, "indexed_symbols": indexed_symbols, "duration_ms": duration_ms}
    return TestResult(passed=True, message=message, details=details), output


def test_incremental_indexing(
    runner: CommandRunner,
    first_run: Optional[Dict[str, object]],
    target_file: Path,
) -> TestResult:
    """Smoke test for incremental indexing behavior."""
    try:
        baseline = first_run or runner.run_index(".", timeout=300.0)
        second_run = runner.run_index(".", timeout=300.0)
    except Exception as exc:
        return TestResult(passed=False, message=f"Index command failed: {exc}")

    first_indexed = int(baseline.get("indexed_files", 0))
    first_skipped = int(baseline.get("skipped_files", 0))
    first_duration = float(baseline.get("duration_ms", 0))
    second_indexed = int(second_run.get("indexed_files", 0))
    second_skipped = int(second_run.get("skipped_files", 0))
    second_symbols = int(second_run.get("indexed_symbols", 0))
    second_duration = float(second_run.get("duration_ms", 0))

    total_files = first_indexed + first_skipped
    skipped_match = second_skipped in {first_indexed, total_files}
    if second_indexed != 0 or second_symbols != 0 or not skipped_match:
        return TestResult(
            passed=False,
            message="Second run did not skip all files",
            details={"first": baseline, "second": second_run},
        )

    speedup = 0.0
    if first_duration > 0:
        speedup = 1.0 - (second_duration / first_duration)
    if first_duration > 0 and speedup < 0.8:
        return TestResult(
            passed=False,
            message="Incremental run was not at least 80% faster",
            details={"first_ms": first_duration, "second_ms": second_duration, "speedup": speedup},
        )

    original = target_file.read_text(encoding="utf8")
    modified = original + "\n# phase1 smoke test change\n"
    try:
        target_file.write_text(modified, encoding="utf8")
        modified_run = runner.run_index(".", timeout=300.0)
    except Exception as exc:
        return TestResult(passed=False, message=f"Re-index after change failed: {exc}")
    finally:
        target_file.write_text(original, encoding="utf8")

    modified_indexed = int(modified_run.get("indexed_files", 0))
    if modified_indexed != 1:
        return TestResult(
            passed=False,
            message="Modified run did not re-index exactly one file",
            details={"modified": modified_run},
        )

    message = (
        f"Skipped {second_skipped}/{total_files} files; speedup {speedup * 100:.1f}% "
        f"({_format_duration(first_duration)} -> {_format_duration(second_duration)})"
    )
    details = {
        "first": baseline,
        "second": second_run,
        "modified": modified_run,
        "speedup": speedup,
    }
    return TestResult(passed=True, message=message, details=details)


def test_search_functionality(runner: CommandRunner, target_file: Path, cache_dir: str) -> TestResult:
    """Smoke test for search filters and scores."""
    try:
        output = runner.run_search("index repository", top_k=5)
    except Exception as exc:
        return TestResult(passed=False, message=f"Search command failed: {exc}")

    schema = Checks.check_search_output(output)
    if not schema.ok:
        return TestResult(passed=False, message=schema.message, details=schema.details)

    results = output.get("results", [])
    if not results:
        return TestResult(passed=False, message="Search returned no results", details=output)

    missing_fields = _check_required_fields(results)
    if missing_fields:
        return TestResult(passed=False, message=missing_fields, details=output)

    fixture_dir, fixture_file, fixture_query = _create_search_fixture()
    try:
        runner.run_index(str(fixture_dir), timeout=300.0)
        cache = CacheManager(CacheConfig(cache_dir))
        fixture_symbols = cache.list_symbols_for_file(str(fixture_file))
        function_symbol = next((symbol for symbol in fixture_symbols if symbol.kind == "function"), None)
        if not function_symbol:
            return TestResult(
                passed=False,
                message="Search fixture did not expose any function symbols",
                details={"file": str(fixture_file)},
            )
        try:
            kind_output = runner.run_search(
                fixture_query,
                top_k=50,
                kind="function",
                file_path=function_symbol.file_path,
            )
        except Exception as exc:
            return TestResult(passed=False, message=f"Search (kind filter) failed: {exc}")
        kind_results = kind_output.get("results", [])
        if not kind_results:
            return TestResult(
                passed=False,
                message="Kind filter returned no results",
                details={"query": fixture_query, "file": function_symbol.file_path},
            )
        if any(item.get("kind") != "function" for item in kind_results):
            return TestResult(
                passed=False,
                message="Kind filter returned invalid results",
                details={"results": kind_results},
            )
        if any(item.get("file") != function_symbol.file_path for item in kind_results):
            return TestResult(
                passed=False,
                message="Kind filter returned unexpected files",
                details={"results": kind_results, "expected_file": function_symbol.file_path},
            )
    finally:
        shutil.rmtree(fixture_dir, ignore_errors=True)

    sample_file = str(target_file)
    if results and isinstance(results[0], dict) and results[0].get("file"):
        sample_file = str(results[0]["file"])
    try:
        file_output = runner.run_search("index", top_k=5, file_path=sample_file)
    except Exception as exc:
        return TestResult(passed=False, message=f"Search (file filter) failed: {exc}")
    file_results = file_output.get("results", [])
    if not file_results or any(item.get("file") != sample_file for item in file_results):
        return TestResult(
            passed=False,
            message="File filter returned invalid results",
            details={"results": file_results, "file": sample_file},
        )

    search_time_ms = None
    metadata = output.get("metadata")
    if isinstance(metadata, dict) and metadata.get("search_time_ms") is not None:
        try:
            search_time_ms = float(metadata["search_time_ms"])
        except (TypeError, ValueError):
            search_time_ms = None

    message = f"Search returned {len(results)} results with valid scores and filters"
    details = {"total_results": len(results)}
    if search_time_ms is not None:
        details["search_time_ms"] = search_time_ms
    return TestResult(passed=True, message=message, details=details)


def _create_docstring_fixture(path: Path) -> None:
    """Create a fixture file with missing docstrings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "\n".join(
            [
                '"""Docstring audit fixture."""',
                "",
                "class MissingDocstring:",
                "    def run(self) -> None:",
                "        pass",
                "",
                "",
                "def greet(name: str, count: int) -> str:",
                "    \"\"\"Greet someone.",
                "",
                "    Args:",
                "        name (str): Name to greet.",
                "    \"\"\"",
                "    return (f\"Hello, {name}!\" * count)",
            ]
        )
        + "\n"
    )
    path.write_text(content, encoding="utf8")


def test_docstring_audit(runner: CommandRunner, fixture_path: Path) -> TestResult:
    """Smoke test for docstring audit warnings."""
    try:
        _create_docstring_fixture(fixture_path)
        output = runner.run_inspect(".")
    except Exception as exc:
        return TestResult(passed=False, message=f"Inspect command failed: {exc}")
    finally:
        if fixture_path.exists():
            fixture_path.unlink()

    warnings = output.get("warnings", [])
    if not isinstance(warnings, list) or not warnings:
        return TestResult(passed=False, message="Inspection returned no warnings", details=output)

    missing_doc = False
    for report in warnings:
        for warning in report.get("warnings", []) if isinstance(report, dict) else []:
            if "Missing docstring" in warning:
                missing_doc = True
    if not missing_doc:
        return TestResult(passed=False, message="Missing docstring warnings not detected")

    reports = output.get("reports", [])
    if not isinstance(reports, list) or not reports:
        return TestResult(passed=False, message="Inspection returned no reports", details=output)
    if not any(report.get("semantic_score") is not None for report in reports if isinstance(report, dict)):
        return TestResult(passed=False, message="No semantic scores reported", details=output)

    message = f"Inspection returned {len(warnings)} warning entries"
    details = {"total_warnings": len(warnings)}
    return TestResult(passed=True, message=message, details=details)


def test_status_and_cache(runner: CommandRunner) -> TestResult:
    """Smoke test for status and cache clear commands."""
    try:
        status_before = runner.run_status(timeout=10.0)
        total_before = int(status_before.get("total_symbols", 0))
        if total_before <= 0:
            return TestResult(passed=False, message="Status showed zero symbols before clear", details=status_before)
        cleared = runner.run_clear_cache()
        if not cleared.get("cleared"):
            return TestResult(passed=False, message="Clear-cache did not report cleared", details=cleared)
        status_after = runner.run_status(timeout=10.0)
        total_after = int(status_after.get("total_symbols", 0))
        if total_after != 0:
            return TestResult(passed=False, message="Status after clear-cache was not zero", details=status_after)
    except Exception as exc:
        return TestResult(passed=False, message=f"Status/cache command failed: {exc}")

    message = f"Status reported {total_before} symbols; cache cleared successfully"
    details = {"total_symbols_before": total_before}
    return TestResult(passed=True, message=message, details=details)


def render_report(results: List[Phase1Result]) -> str:
    """Render a markdown report for phase 1."""
    total = len(results)
    passed = sum(1 for item in results if item.result.passed)
    failed = total - passed

    lines = [
        "# Phase 1: Smoke Tests",
        "",
        "## Summary",
        f"- Total: {total}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        "",
    ]
    for item in results:
        status = "PASS" if item.result.passed else "FAIL"
        lines.append(f"## {item.name} - {status}")
        lines.append("")
        lines.append(f"- {item.result.message}")
        if item.result.details:
            details = json.dumps(item.result.details, indent=2)
            lines.append("- Details:")
            lines.append("```json")
            lines.append(details)
            lines.append("```")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_phase1(
    verbose: bool = False,
    emit_summary: bool = True,
    baseline_path: Optional[str] = None,
    skip_docstring_audit: bool = False,
) -> Tuple[int, str, Dict[str, object]]:
    """Run phase 1 smoke tests."""
    start = time.perf_counter()
    logger.info("Phase 1 started")
    reporter = Reporter()
    reporter.add_section("Phase 1 Smoke Tests")
    baseline_payload = _load_baseline_payload(baseline_path)
    if baseline_payload:
        reporter.load_baseline_from_payload(baseline_payload)

    results: List[Phase1Result] = []
    cache_dir = os.getenv("GLOGGUR_CACHE_DIR") or str(Path(".gloggur-cache").resolve())
    # Configure retry for transient failures (e.g., network hiccups or rate limits).
    retry_config = RetryConfig(
        max_attempts=3,
        initial_backoff_ms=1000.0,
        retryable_exceptions=(ConnectionError, TimeoutError),
    )
    runner = CommandRunner(
        cwd=str(PROJECT_ROOT),
        env={"GLOGGUR_CACHE_DIR": cache_dir},
        default_timeout=120.0,
        retry_config=retry_config,
    )

    search_target_file = Path("gloggur/cli/main.py")
    fixture_path = Path("tests/fixtures/phase1_docstring_fixture.py")

    with TestFixtures(cache_dir=cache_dir) as fixtures:
        backup_path: Optional[Path] = None
        incremental_target_file: Optional[Path] = None
        try:
            backup_path = fixtures.backup_cache()
            fixtures.cleanup_cache()
            incremental_target_file = _create_incremental_target_fixture()

            test_result, first_run = test_basic_indexing(runner, cache_dir)
            results.append(Phase1Result("Test 1.1: Basic Indexing", test_result))
            reporter.add_test_result("Test 1.1: Basic Indexing", test_result)
            if isinstance(test_result.details, dict):
                duration_ms = test_result.details.get("duration_ms")
                indexed_files = test_result.details.get("indexed_files")
                if isinstance(duration_ms, (int, float)) and duration_ms > 0:
                    throughput = None
                    if isinstance(indexed_files, (int, float)):
                        throughput = indexed_files / (duration_ms / 1000)
                    reporter.add_performance_metric(
                        "Indexing",
                        duration_ms=float(duration_ms),
                        throughput=throughput,
                        throughput_unit="files/s",
                    )
                    reporter.set_performance_thresholds(
                        "Indexing",
                        max_duration_ms=_env_float("GLOGGUR_PERF_INDEX_MAX_MS", 30000.0),
                    )
            if verbose and test_result.details:
                print(json.dumps({"test": "basic_indexing", "details": test_result.details}, indent=2))

            if incremental_target_file is None:
                test_result = TestResult(passed=False, message="Incremental fixture setup failed")
            else:
                test_result = test_incremental_indexing(runner, first_run, incremental_target_file)
            results.append(Phase1Result("Test 1.2: Incremental Indexing", test_result))
            reporter.add_test_result("Test 1.2: Incremental Indexing", test_result)
            if isinstance(test_result.details, dict):
                second_run = test_result.details.get("second")
                first_run_details = test_result.details.get("first")
                if isinstance(second_run, dict):
                    duration_ms = second_run.get("duration_ms")
                    total_files = None
                    if isinstance(first_run_details, dict):
                        first_indexed = first_run_details.get("indexed_files", 0)
                        first_skipped = first_run_details.get("skipped_files", 0)
                        if isinstance(first_indexed, (int, float)) and isinstance(first_skipped, (int, float)):
                            total_files = first_indexed + first_skipped
                    if isinstance(duration_ms, (int, float)) and duration_ms > 0:
                        throughput = None
                        if isinstance(total_files, (int, float)) and total_files > 0:
                            throughput = total_files / (duration_ms / 1000)
                        reporter.add_performance_metric(
                            "Incremental Indexing",
                            duration_ms=float(duration_ms),
                            throughput=throughput,
                            throughput_unit="files/s",
                        )
                        reporter.set_performance_thresholds(
                            "Incremental Indexing",
                            max_duration_ms=_env_float("GLOGGUR_PERF_INCREMENTAL_MAX_MS", 10000.0),
                        )
            if verbose and test_result.details:
                print(json.dumps({"test": "incremental_indexing", "details": test_result.details}, indent=2))

            test_result = test_search_functionality(runner, search_target_file, cache_dir)
            results.append(Phase1Result("Test 1.3: Search Functionality", test_result))
            reporter.add_test_result("Test 1.3: Search Functionality", test_result)
            if isinstance(test_result.details, dict):
                search_time_ms = test_result.details.get("search_time_ms")
                total_results = test_result.details.get("total_results")
                if isinstance(search_time_ms, (int, float)) and search_time_ms > 0:
                    throughput = None
                    if isinstance(total_results, (int, float)):
                        throughput = total_results / (search_time_ms / 1000)
                    reporter.add_performance_metric(
                        "Search",
                        duration_ms=float(search_time_ms),
                        throughput=throughput,
                        throughput_unit="results/s",
                    )
                    reporter.set_performance_thresholds(
                        "Search",
                        max_duration_ms=_env_float("GLOGGUR_PERF_SEARCH_MAX_MS", 5000.0),
                    )
            if verbose and test_result.details:
                print(json.dumps({"test": "search_functionality", "details": test_result.details}, indent=2))

            if skip_docstring_audit:
                test_result = TestResult(
                    passed=True,
                    message="Docstring audit skipped for this run",
                    details={"skipped": True},
                )
            else:
                test_result = test_docstring_audit(runner, fixture_path)
            results.append(Phase1Result("Test 1.4: Docstring Audit", test_result))
            reporter.add_test_result("Test 1.4: Docstring Audit", test_result)
            if verbose and test_result.details:
                print(json.dumps({"test": "docstring_audit", "details": test_result.details}, indent=2))

            test_result = test_status_and_cache(runner)
            results.append(Phase1Result("Test 1.5: Status & Cache Management", test_result))
            reporter.add_test_result("Test 1.5: Status & Cache Management", test_result)
            if verbose and test_result.details:
                print(json.dumps({"test": "status_and_cache", "details": test_result.details}, indent=2))

        finally:
            if backup_path is not None:
                fixtures.restore_cache(backup_path)
                shutil.rmtree(backup_path, ignore_errors=True)
            else:
                fixtures.cleanup_cache()
            if fixture_path.exists():
                fixture_path.unlink()
            if incremental_target_file is not None and incremental_target_file.exists():
                incremental_target_file.unlink()

    if emit_summary:
        reporter.print_summary()
    exit_code = 0 if all(item.result.passed for item in results) else 1
    duration_ms = (time.perf_counter() - start) * 1000
    reporter.add_performance_metric(
        "Phase 1 Total",
        duration_ms=duration_ms,
        throughput=len(results) / (duration_ms / 1000) if duration_ms > 0 else None,
        throughput_unit="tests/s",
    )
    if baseline_payload:
        reporter.add_baseline_trends()
    markdown = render_report(results)
    performance_section = reporter.render_performance_markdown()
    if performance_section:
        markdown = f"{markdown.rstrip()}\n\n{performance_section}\n"
    payload = reporter.generate_json()
    summary = payload.get("summary", {})
    if isinstance(summary, dict) and "skipped" not in summary:
        summary["skipped"] = 0
    payload.update(
        {
            "phase": 1,
            "title": "Smoke Tests",
            "duration_ms": duration_ms,
            "summary": summary,
        }
    )
    logger.info("Phase 1 completed in %.2f ms", duration_ms)
    return exit_code, markdown, payload


def main() -> None:
    """CLI entrypoint for phase 1 smoke run."""
    parser = argparse.ArgumentParser(description="Run Phase 1 smoke tests for gloggur.")
    parser.add_argument("--output", type=str, default=None, help="Write markdown report to a file")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--verbose", action="store_true", help="Print verbose test details")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to JSON report payload for baseline performance comparisons",
    )
    parser.add_argument(
        "--skip-docstring-audit",
        action="store_true",
        help="Skip the docstring audit smoke test.",
    )
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

    emit_summary = args.format != "json"
    verbose = args.verbose and args.format != "json"
    skip_docstring_audit = args.skip_docstring_audit or _env_flag("GLOGGUR_SKIP_DOCSTRING_AUDIT")
    exit_code, markdown, payload = run_phase1(
        verbose=verbose,
        emit_summary=emit_summary,
        baseline_path=args.baseline,
        skip_docstring_audit=skip_docstring_audit,
    )
    output = markdown if args.format == "markdown" else json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf8")
    else:
        print(output)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
