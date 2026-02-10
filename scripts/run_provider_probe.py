from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gloggur.config import GloggurConfig
from scripts.verification import CommandRunner, Checks
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


def _vector_store_stats(cache_dir: str) -> Tuple[bool, str, Dict[str, object]]:
    """Check vector store files and return status/details."""
    index_path = os.path.join(cache_dir, "vectors.index")
    id_map_path = os.path.join(cache_dir, "vectors.json")
    if not os.path.exists(index_path):
        return False, "Vector index file missing", {"index_path": index_path}
    if not os.path.exists(id_map_path):
        return False, "Vector ID map file missing", {"id_map_path": id_map_path}
    try:
        with open(id_map_path, "r", encoding="utf8") as handle:
            ids = json.load(handle)
    except json.JSONDecodeError as exc:
        return False, "Vector ID map invalid JSON", {"error": str(exc), "id_map_path": id_map_path}
    if not isinstance(ids, list) or not ids:
        return False, "Vector ID map empty", {"id_count": len(ids) if isinstance(ids, list) else 0}
    return True, "Vector store populated", {"id_count": len(ids)}


def _check_index_output(output: Dict[str, object]) -> Optional[Tuple[str, Dict[str, object]]]:
    """Check index output and return error details if any."""
    schema = Checks.check_index_output(output)
    if not schema.ok:
        return schema.message, schema.details or {}
    indexed_files = int(output.get("indexed_files", 0))
    indexed_symbols = int(output.get("indexed_symbols", 0))
    if indexed_files <= 0:
        return "Indexing returned zero files", {"output": output}
    if indexed_symbols <= 0:
        return "Indexing returned zero symbols", {"output": output}
    return None


def _check_search_output(output: Dict[str, object]) -> Optional[Tuple[str, Dict[str, object]]]:
    """Check search output and return error details if any."""
    schema = Checks.check_search_output(output)
    if not schema.ok:
        return schema.message, schema.details or {}
    results = output.get("results", [])
    if not results:
        return "Search returned no results", {"output": output}
    return None


def _run_provider_test(
    name: str,
    provider: str,
    query: str,
    cache_dir: str,
    runner: CommandRunner,
    model_name: Optional[str] = None,
) -> TestCaseResult:
    """Run a single embedding provider probe."""
    try:
        index_output = runner.run_index(".", embedding_provider=provider)
    except Exception as exc:
        return TestCaseResult(name=name, status="failed", message=f"Index command failed: {exc}")

    index_error = _check_index_output(index_output)
    if index_error:
        message, details = index_error
        return TestCaseResult(name=name, status="failed", message=message, details=details)

    cache_check = Checks.check_cache_exists(cache_dir)
    if not cache_check.ok:
        return TestCaseResult(name=name, status="failed", message=cache_check.message, details=cache_check.details)

    vectors_ok, vector_message, vector_details = _vector_store_stats(cache_dir)
    if not vectors_ok:
        return TestCaseResult(name=name, status="failed", message=vector_message, details=vector_details)

    try:
        search_output = runner.run_search(query, top_k=3)
    except Exception as exc:
        return TestCaseResult(name=name, status="failed", message=f"Search command failed: {exc}")

    search_error = _check_search_output(search_output)
    if search_error:
        message, details = search_error
        return TestCaseResult(name=name, status="failed", message=message, details=details)

    indexed_files = int(index_output.get("indexed_files", 0))
    indexed_symbols = int(index_output.get("indexed_symbols", 0))
    details = {
        "indexed_files": indexed_files,
        "indexed_symbols": indexed_symbols,
        "vector_count": vector_details.get("id_count"),
    }
    if model_name:
        details["model"] = model_name
    return TestCaseResult(
        name=name,
        status="passed",
        message=f"Indexed {indexed_files} files and returned {len(search_output.get('results', []))} results",
        details=details,
    )


def _load_repo_config(repo_root: Path) -> GloggurConfig:
    """Load repo configuration for embedding tests."""
    for candidate in (".gloggur.yaml", ".gloggur.yml", ".gloggur.json"):
        config_path = repo_root / candidate
        if config_path.exists():
            return GloggurConfig.load(path=str(config_path))
    return GloggurConfig.load()


def test_local_embeddings(repo_root: Path, config: GloggurConfig, verbose: bool = False) -> TestCaseResult:
    """Check local embedding provider."""
    model_name = config.local_embedding_model
    with tempfile.TemporaryDirectory(prefix="gloggur-phase2-local-") as cache_dir:
        runner = CommandRunner(
            cwd=str(repo_root),
            env={"GLOGGUR_CACHE_DIR": cache_dir},
            default_timeout=300.0,
        )
        result = _run_provider_test(
            name="Test 2.1: Local Embeddings",
            provider="local",
            query="parser",
            cache_dir=cache_dir,
            runner=runner,
            model_name=model_name,
        )
        if verbose and result.details:
            print(json.dumps({"test": "local_embeddings", "details": result.details}, indent=2))
        return result


def _missing_api_key_result(name: str, env_key: str) -> TestCaseResult:
    """Return a skipped test result when API key is missing."""
    return TestCaseResult(
        name=name,
        status="skipped",
        message=f"Skipped: {env_key} not set",
        details={"missing_env": env_key},
    )


def test_openai_embeddings(repo_root: Path, config: GloggurConfig, verbose: bool = False) -> TestCaseResult:
    """Check OpenAI embedding provider."""
    if not os.getenv("OPENAI_API_KEY"):
        return _missing_api_key_result("Test 2.2: OpenAI Embeddings", "OPENAI_API_KEY")

    model_name = config.openai_embedding_model
    with tempfile.TemporaryDirectory(prefix="gloggur-phase2-openai-") as cache_dir:
        runner = CommandRunner(
            cwd=str(repo_root),
            env={"GLOGGUR_CACHE_DIR": cache_dir},
            default_timeout=300.0,
        )
        result = _run_provider_test(
            name="Test 2.2: OpenAI Embeddings",
            provider="openai",
            query="vector store",
            cache_dir=cache_dir,
            runner=runner,
            model_name=model_name,
        )
        if verbose and result.details:
            print(json.dumps({"test": "openai_embeddings", "details": result.details}, indent=2))
        return result


def test_gemini_embeddings(repo_root: Path, config: GloggurConfig, verbose: bool = False) -> TestCaseResult:
    """Check Gemini embedding provider."""
    env_key = None
    for candidate in ("GLOGGUR_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.getenv(candidate):
            env_key = candidate
            break
    if env_key is None:
        return _missing_api_key_result("Test 2.3: Gemini Embeddings", "GLOGGUR_GEMINI_API_KEY or GEMINI_API_KEY")

    model_name = config.gemini_embedding_model
    with tempfile.TemporaryDirectory(prefix="gloggur-phase2-gemini-") as cache_dir:
        runner = CommandRunner(
            cwd=str(repo_root),
            env={"GLOGGUR_CACHE_DIR": cache_dir},
            default_timeout=300.0,
        )
        result = _run_provider_test(
            name="Test 2.3: Gemini Embeddings",
            provider="gemini",
            query="hybrid search",
            cache_dir=cache_dir,
            runner=runner,
            model_name=model_name,
        )
        if verbose and result.details:
            print(json.dumps({"test": "gemini_embeddings", "details": result.details}, indent=2))
        return result


def _summarize_tests(tests: List[TestCaseResult]) -> Dict[str, int]:
    """Summarize test results by status."""
    total = len(tests)
    passed = sum(1 for test in tests if test.status == "passed")
    failed = sum(1 for test in tests if test.status == "failed")
    skipped = sum(1 for test in tests if test.status == "skipped")
    return {"total": total, "passed": passed, "failed": failed, "skipped": skipped}


def _phase_status(summary: Dict[str, int]) -> str:
    """Derive phase status from summary counts."""
    if summary.get("failed"):
        return "failed"
    if summary.get("skipped"):
        return "passed_with_skips"
    return "passed"


def run_phase2(verbose: bool = False) -> VerificationReport:
    """Run phase 2 provider probes."""
    repo_root = Path(__file__).resolve().parents[1]
    config = _load_repo_config(repo_root)
    start = time.perf_counter()
    logger.info("Phase 2 started")

    tests = [
        test_local_embeddings(repo_root, config, verbose=verbose),
        test_openai_embeddings(repo_root, config, verbose=verbose),
        test_gemini_embeddings(repo_root, config, verbose=verbose),
    ]

    summary = _summarize_tests(tests)
    phase_report = PhaseReport(
        phase=2,
        title="Embedding Providers",
        status=_phase_status(summary),
        summary=summary,
        tests=tests,
        duration_ms=(time.perf_counter() - start) * 1000,
    )
    report = build_verification_report([phase_report])
    logger.info("Phase 2 completed in %.2f ms", phase_report.duration_ms)
    return report


def main() -> int:
    """CLI entrypoint for phase 2 provider probes."""
    parser = argparse.ArgumentParser(description="Run Phase 2 embedding provider tests for gloggur.")
    parser.add_argument("--output", type=str, default=None, help="Write report to a file.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--verbose", action="store_true", help="Print verbose test details.")
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

    report = run_phase2(verbose=args.verbose)
    output = render_markdown(report) if args.format == "markdown" else json.dumps(render_json(report), indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf8")
    else:
        print(output)

    return 0 if report.summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
