from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.models import Symbol
from scripts.verification import CommandRunner, Reporter, TestFixtures, TestResult, Checks
from scripts.verification.reporter import PerformanceMetric


def test_command_runner_status_json() -> None:
    """Ensure status command returns expected JSON fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = CommandRunner(env={"GLOGGUR_CACHE_DIR": tmpdir})
        payload = runner.run_status()
        assert payload["cache_dir"] == tmpdir
        assert "total_symbols" in payload


def test_checks_schema_and_scores() -> None:
    """Check schema and similarity score ranges."""
    output = {
        "query": "hello",
        "results": [{"similarity_score": 0.5}],
        "metadata": {"total_results": 1, "search_time_ms": 10},
    }
    result = Checks.check_search_output(output)
    assert result.ok, result.message


def test_database_symbols_check() -> None:
    """Check database symbol count bounds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(CacheConfig(tmpdir))
        symbol = Symbol(
            id="sym-1",
            name="demo",
            kind="function",
            file_path="demo.py",
            start_line=1,
            end_line=2,
            signature="demo()",
            docstring="demo",
            body_hash="hash",
            embedding_vector=None,
            language="python",
        )
        cache.upsert_symbols([symbol])
        result = Checks.check_database_symbols(os.path.join(tmpdir, "index.db"), 1)
        assert result.ok, result.message


def test_reporter_outputs() -> None:
    """Ensure reporter emits markdown and JSON summaries."""
    reporter = Reporter()
    reporter.add_section("Smoke")
    reporter.add_test_result("status", TestResult(passed=True, message="ok"))
    reporter.add_test_result("search", TestResult(passed=False, message="fail"))
    markdown = reporter.generate_markdown()
    payload = reporter.generate_json()
    assert "Verification Report" in markdown
    assert payload["summary"]["failed"] == 1


def test_reporter_performance_markdown_with_baseline_trends() -> None:
    """Ensure performance markdown includes baseline comparisons."""
    reporter = Reporter()
    reporter.add_performance_metric("Phase 1 Total", duration_ms=1500.0, throughput=2.0, throughput_unit="tests/s")
    reporter.set_baseline_metrics(
        {"Phase 1 Total": PerformanceMetric(name="Phase 1 Total", duration_ms=1200.0, throughput=2.5)}
    )
    reporter.add_baseline_trends()
    markdown = reporter.render_performance_markdown()
    assert "Phase 1 Total" in markdown
    assert "Comparison to baseline:" in markdown
    assert "Performance Trends" in markdown
    assert "```mermaid" in markdown


def test_fixtures_create_and_cleanup() -> None:
    """Ensure fixtures create and cleanup temp repos and caches."""
    fixtures = TestFixtures(cache_dir=tempfile.mkdtemp(prefix="gloggur-cache-"))
    repo = fixtures.create_temp_repo({"sample.py": fixtures.create_sample_python_file()})
    assert (repo / "sample.py").exists()
    fixtures.cleanup_cache()
    assert not os.path.isdir(fixtures.cache_dir)


def test_fixtures_backup_restore() -> None:
    """Ensure fixture cache backup and restore works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()
        (cache_dir / "index.db").write_text("data", encoding="utf8")
        fixtures = TestFixtures(cache_dir=str(cache_dir))
        backup = fixtures.backup_cache()
        assert backup is not None
        (cache_dir / "index.db").write_text("new", encoding="utf8")
        fixtures.restore_cache(backup)
        assert (cache_dir / "index.db").read_text(encoding="utf8") == "data"
        if backup.exists():
            shutil.rmtree(backup)
