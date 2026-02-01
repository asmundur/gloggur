from __future__ import annotations

import argparse
import json
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
from scripts.validation import CommandRunner, Reporter, TestFixtures, TestResult, Validators


@dataclass
class Phase1Result:
    name: str
    result: TestResult


def _format_duration(duration_ms: float | int) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f}s"
    return f"{int(duration_ms)}ms"


def _count_symbols(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
        return int(row[0]) if row else 0


def _validate_required_fields(results: List[Dict[str, object]]) -> Optional[str]:
    required = {"symbol", "kind", "file", "line", "signature", "similarity_score"}
    for idx, item in enumerate(results):
        missing = [field for field in required if field not in item]
        if missing:
            return f"Result {idx} missing fields: {', '.join(missing)}"
    return None


def _create_search_fixture() -> Tuple[Path, Path, str]:
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


def test_basic_indexing(runner: CommandRunner, cache_dir: str) -> Tuple[TestResult, Dict[str, object]]:
    try:
        output = runner.run_index(".")
    except Exception as exc:
        return TestResult(passed=False, message=f"Index command failed: {exc}"), {}

    schema = Validators.validate_index_output(output)
    if not schema.ok:
        return TestResult(passed=False, message=schema.message, details=schema.details), output

    indexed_files = int(output.get("indexed_files", 0))
    indexed_symbols = int(output.get("indexed_symbols", 0))
    duration_ms = float(output.get("duration_ms", 0))

    if indexed_files <= 0:
        return TestResult(passed=False, message="No files indexed", details=output), output
    if indexed_symbols <= 0:
        return TestResult(passed=False, message="No symbols indexed", details=output), output

    cache_check = Validators.check_cache_exists(cache_dir)
    if not cache_check.ok:
        return TestResult(passed=False, message=cache_check.message, details=cache_check.details), output

    db_path = os.path.join(cache_dir, "index.db")
    db_check = Validators.check_database_symbols(db_path, 1)
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
    try:
        baseline = first_run or runner.run_index(".")
        second_run = runner.run_index(".")
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
        modified_run = runner.run_index(".")
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
    try:
        output = runner.run_search("index repository", top_k=5)
    except Exception as exc:
        return TestResult(passed=False, message=f"Search command failed: {exc}")

    schema = Validators.validate_search_output(output)
    if not schema.ok:
        return TestResult(passed=False, message=schema.message, details=schema.details)

    results = output.get("results", [])
    if not results:
        return TestResult(passed=False, message="Search returned no results", details=output)

    missing_fields = _validate_required_fields(results)
    if missing_fields:
        return TestResult(passed=False, message=missing_fields, details=output)

    fixture_dir, fixture_file, fixture_query = _create_search_fixture()
    try:
        runner.run_index(str(fixture_dir))
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

    message = f"Search returned {len(results)} results with valid scores and filters"
    details = {"total_results": len(results)}
    return TestResult(passed=True, message=message, details=details)


def _create_docstring_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "\n".join(
            [
                '"""Docstring validation fixture."""',
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


def test_docstring_validation(runner: CommandRunner, fixture_path: Path) -> TestResult:
    try:
        _create_docstring_fixture(fixture_path)
        output = runner.run_validate(".")
    except Exception as exc:
        return TestResult(passed=False, message=f"Validate command failed: {exc}")
    finally:
        if fixture_path.exists():
            fixture_path.unlink()

    warnings = output.get("warnings", [])
    if not isinstance(warnings, list) or not warnings:
        return TestResult(passed=False, message="Validation returned no warnings", details=output)

    missing_doc = False
    missing_params = False
    for report in warnings:
        for warning in report.get("warnings", []) if isinstance(report, dict) else []:
            if "Missing docstring" in warning:
                missing_doc = True
            if "Missing docstring params" in warning:
                missing_params = True
    if not missing_doc:
        return TestResult(passed=False, message="Missing docstring warnings not detected")
    if not missing_params:
        return TestResult(passed=False, message="Missing parameter doc warnings not detected")

    message = f"Validation returned {len(warnings)} warning entries"
    details = {"total_warnings": len(warnings)}
    return TestResult(passed=True, message=message, details=details)


def test_status_and_cache(runner: CommandRunner) -> TestResult:
    try:
        status_before = runner.run_status()
        total_before = int(status_before.get("total_symbols", 0))
        if total_before <= 0:
            return TestResult(passed=False, message="Status showed zero symbols before clear", details=status_before)
        cleared = runner.run_clear_cache()
        if not cleared.get("cleared"):
            return TestResult(passed=False, message="Clear-cache did not report cleared", details=cleared)
        status_after = runner.run_status()
        total_after = int(status_after.get("total_symbols", 0))
        if total_after != 0:
            return TestResult(passed=False, message="Status after clear-cache was not zero", details=status_after)
    except Exception as exc:
        return TestResult(passed=False, message=f"Status/cache command failed: {exc}")

    message = f"Status reported {total_before} symbols; cache cleared successfully"
    details = {"total_symbols_before": total_before}
    return TestResult(passed=True, message=message, details=details)


def render_report(results: List[Phase1Result]) -> str:
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


def run_phase1(verbose: bool = False, emit_summary: bool = True) -> Tuple[int, str, Dict[str, object]]:
    start = time.perf_counter()
    reporter = Reporter()
    reporter.add_section("Phase 1 Smoke Tests")

    results: List[Phase1Result] = []
    cache_dir = str(Path(".gloggur-cache").resolve())
    runner = CommandRunner(
        cwd=str(PROJECT_ROOT),
        env={"GLOGGUR_CACHE_DIR": cache_dir},
        default_timeout=300.0,
    )

    target_file = Path("gloggur/cli/main.py")
    fixture_path = Path("tests/fixtures/phase1_docstring_fixture.py")

    with TestFixtures(cache_dir=cache_dir) as fixtures:
        backup_path: Optional[Path] = None
        try:
            backup_path = fixtures.backup_cache()
            fixtures.cleanup_cache()

            test_result, first_run = test_basic_indexing(runner, cache_dir)
            results.append(Phase1Result("Test 1.1: Basic Indexing", test_result))
            reporter.add_test_result("Test 1.1: Basic Indexing", test_result)
            if verbose and test_result.details:
                print(json.dumps({"test": "basic_indexing", "details": test_result.details}, indent=2))

            test_result = test_incremental_indexing(runner, first_run, target_file)
            results.append(Phase1Result("Test 1.2: Incremental Indexing", test_result))
            reporter.add_test_result("Test 1.2: Incremental Indexing", test_result)
            if verbose and test_result.details:
                print(json.dumps({"test": "incremental_indexing", "details": test_result.details}, indent=2))

            test_result = test_search_functionality(runner, target_file, cache_dir)
            results.append(Phase1Result("Test 1.3: Search Functionality", test_result))
            reporter.add_test_result("Test 1.3: Search Functionality", test_result)
            if verbose and test_result.details:
                print(json.dumps({"test": "search_functionality", "details": test_result.details}, indent=2))

            test_result = test_docstring_validation(runner, fixture_path)
            results.append(Phase1Result("Test 1.4: Docstring Validation", test_result))
            reporter.add_test_result("Test 1.4: Docstring Validation", test_result)
            if verbose and test_result.details:
                print(json.dumps({"test": "docstring_validation", "details": test_result.details}, indent=2))

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

    markdown = render_report(results)
    if emit_summary:
        reporter.print_summary()
    exit_code = 0 if all(item.result.passed for item in results) else 1
    duration_ms = (time.perf_counter() - start) * 1000
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
    return exit_code, markdown, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 smoke tests for gloggur.")
    parser.add_argument("--output", type=str, default=None, help="Write markdown report to a file")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--verbose", action="store_true", help="Print verbose test details")
    args = parser.parse_args()

    emit_summary = args.format != "json"
    verbose = args.verbose and args.format != "json"
    exit_code, markdown, payload = run_phase1(verbose=verbose, emit_summary=emit_summary)
    output = markdown if args.format == "markdown" else json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf8")
    else:
        print(output)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
