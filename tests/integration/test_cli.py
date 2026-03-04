from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.error
from pathlib import Path

import pytest
from click.testing import CliRunner

from gloggur.cli import main as cli_main
from gloggur.cli.main import cli
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.search import attach_legacy_search_contract
from scripts.verification.fixtures import TestFixtures


def _write_fallback_marker(cache_dir: str) -> None:
    """Create the local embedding fallback marker file."""
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _parse_json_output(output: str) -> dict[str, object]:
    """Parse JSON output from CLI command output."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    if isinstance(payload, dict):
        return attach_legacy_search_contract(payload)
    return payload


def _tamper_vector_id_map_drop_one_symbol(cache_dir: str) -> None:
    """Drop one symbol id from vectors.json to simulate vector/cache drift."""
    id_map_path = Path(cache_dir) / "vectors.json"
    payload = json.loads(id_map_path.read_text(encoding="utf8"))
    symbol_map = payload.get("symbol_to_vector_id")
    if not isinstance(symbol_map, dict) or not symbol_map:
        raise AssertionError("expected non-empty symbol_to_vector_id map")
    drop_id = sorted(symbol_map)[0]
    symbol_map.pop(drop_id, None)
    payload["symbol_to_vector_id"] = symbol_map
    fallback_order = payload.get("fallback_order")
    if isinstance(fallback_order, list):
        payload["fallback_order"] = [
            str(symbol_id) for symbol_id in fallback_order if str(symbol_id) in symbol_map
        ]
    id_map_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")


def _sha256_for_file(path: Path) -> str:
    """Return sha256 digest for a file path."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_index_search_status_and_clear_cache() -> None:
    """End-to-end CLI test for index/search/status/clear-cache."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0
        index_payload = _parse_json_output(index_result.output)
        assert index_payload["indexed_files"] == 1
        assert index_payload["indexed_symbols"] > 0

        status_result = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_result.exit_code == 0
        status_payload = _parse_json_output(status_result.output)
        assert status_payload["total_symbols"] > 0

        search_result = runner.invoke(cli, ["search", "add", "--json", "--top-k", "3"], env=env)
        assert search_result.exit_code == 0
        search_payload = _parse_json_output(search_result.output)
        assert search_payload["metadata"]["total_results"] > 0

        clear_result = runner.invoke(cli, ["clear-cache", "--json"], env=env)
        assert clear_result.exit_code == 0
        clear_payload = _parse_json_output(clear_result.output)
        assert clear_payload["cleared"] is True


def test_cli_search_json_emits_contextpack_v2_without_legacy_keys() -> None:
    """search --json should emit ContextPack v2 and exclude removed v1 payload keys."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(cli, ["search", "add", "--json", "--top-k", "3"], env=env)
        assert result.exit_code == 0, result.output
        start = result.output.find("{")
        assert start >= 0
        payload = json.loads(result.output[start:])
        assert payload["schema_version"] == 2
        assert isinstance(payload.get("summary"), dict)
        assert isinstance(payload.get("hits"), list)
        assert "results" not in payload
        assert "metadata" not in payload


def test_cli_search_low_signal_reports_bounded_retry_metadata() -> None:
    """Low-signal search should return empty bounded evidence with deterministic metadata."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0

        search_result = runner.invoke(
            cli,
            [
                "search",
                "unlikely-token-for-low-signal",
                "--json",
                "--file",
                "missing.py",
                "--top-k",
                "2",
                "--confidence-threshold",
                "0.95",
                "--max-requery-attempts",
                "1",
            ],
            env=env,
        )
        assert search_result.exit_code == 0, search_result.output
        payload = _parse_json_output(search_result.output)
        metadata = payload["metadata"]
        assert isinstance(metadata, dict)
        assert int(metadata["total_results"]) == 0
        assert metadata["initial_confidence"] == pytest.approx(0.0)
        assert metadata["final_confidence"] == pytest.approx(0.0)
        assert metadata["retry_enabled"] is True
        assert metadata["retry_performed"] is False
        assert metadata["retry_attempts"] == 0
        assert metadata["initial_top_k"] == 2
        assert metadata["final_top_k"] == 2
        assert metadata["low_confidence"] is True


def test_cli_search_file_filter_prefix_boundary_and_context_radius() -> None:
    """File filter should support exact+prefix matching with boundary safety and context metadata."""
    runner = CliRunner()
    source_src = (
        "# prelude\n"
        "def src_target() -> str:\n"
        "    token = 'src-token'\n"
        "    token = token + '-a'\n"
        "    token = token + '-b'\n"
        "    token = token + '-c'\n"
        "    token = token + '-d'\n"
        "    token = token + '-e'\n"
        "    return token\n"
    )
    source_src2 = (
        "def src2_target() -> str:\n"
        "    return 'src2-token'\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "src/feature.py": source_src,
                "src2/feature.py": source_src2,
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}
        previous_cwd = os.getcwd()
        os.chdir(repo)
        try:
            index_result = runner.invoke(cli, ["index", ".", "--json"], env=env)
            assert index_result.exit_code == 0, index_result.output

            src_filter_result = runner.invoke(
                cli,
                ["search", "token", "--json", "--file", "src", "--top-k", "10"],
                env=env,
            )
            assert src_filter_result.exit_code == 0, src_filter_result.output
            src_payload = _parse_json_output(src_filter_result.output)
            src_results = src_payload["results"]
            assert isinstance(src_results, list) and src_results
            assert all("/src/" in str(item["file"]).replace("\\", "/") for item in src_results)
            assert all("/src2/" not in str(item["file"]).replace("\\", "/") for item in src_results)
            src_metadata = src_payload["metadata"]
            assert src_metadata["file_filter"] == "src"
            assert src_metadata["file_filter_match_mode"] == "exact_or_prefix"
            assert src_metadata["file_filter_warning_codes"] == []

            missing_filter_result = runner.invoke(
                cli,
                ["search", "token", "--json", "--file", "missing/", "--top-k", "5"],
                env=env,
            )
            assert missing_filter_result.exit_code == 0, missing_filter_result.output
            missing_payload = _parse_json_output(missing_filter_result.output)
            missing_metadata = missing_payload["metadata"]
            assert int(missing_metadata["total_results"]) == 0
            assert missing_metadata["file_filter"] == "missing/"
            assert missing_metadata["file_filter_warning_codes"] == ["file_filter_no_match"]

            default_radius_result = runner.invoke(
                cli,
                ["search", "token", "--json", "--file", "src", "--top-k", "1"],
                env=env,
            )
            assert default_radius_result.exit_code == 0, default_radius_result.output
            default_payload = _parse_json_output(default_radius_result.output)
            default_metadata = default_payload["metadata"]
            assert default_metadata["context_radius"] == 12
            default_context_lines = str(default_payload["results"][0]["context"]).splitlines()

            custom_radius_result = runner.invoke(
                cli,
                [
                    "search",
                    "token",
                    "--json",
                    "--file",
                    "src",
                    "--top-k",
                    "1",
                    "--context-radius",
                    "15",
                ],
                env=env,
            )
            assert custom_radius_result.exit_code == 0, custom_radius_result.output
            custom_payload = _parse_json_output(custom_radius_result.output)
            custom_metadata = custom_payload["metadata"]
            assert custom_metadata["context_radius"] == 15
            custom_context_lines = str(custom_payload["results"][0]["context"]).splitlines()
            assert len(custom_context_lines) >= len(default_context_lines)
        finally:
            os.chdir(previous_cwd)


def test_cli_search_rejects_legacy_grounding_contract_flags() -> None:
    """Legacy evidence/grounding flags should fail closed under ContextPack v2."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0

        search_result = runner.invoke(
            cli,
            [
                "search",
                "add",
                "--json",
                "--with-evidence-trace",
                "--validate-grounding",
                "--evidence-min-confidence",
                "0.0",
                "--evidence-min-items",
                "1",
            ],
            env=env,
        )
        assert search_result.exit_code == 1, search_result.output
        payload = _parse_json_output(search_result.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["code"] == "search_contract_v1_removed"
        assert payload["failure_codes"] == ["search_contract_v1_removed"]


def test_cli_search_fail_on_ungrounded_returns_nonzero_with_validation_payload() -> None:
    """Legacy fail-on-ungrounded path should fail closed with v2 migration code."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0

        search_result = runner.invoke(
            cli,
            [
                "search",
                "needle",
                "--json",
                "--file",
                "missing.py",
                "--validate-grounding",
                "--fail-on-ungrounded",
            ],
            env=env,
        )
        assert search_result.exit_code == 1
        payload = _parse_json_output(search_result.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["code"] == "search_contract_v1_removed"
        assert payload["failure_codes"] == ["search_contract_v1_removed"]


def test_cli_guidance_uses_ingested_coverage_for_constraining_tests() -> None:
    """Coverage ingest should feed guidance test-to-code linkage and untested warnings."""
    runner = CliRunner()
    source = """
def covered_target(x):
    return x + 1

def untested_with_invariant(x):
    assert x > 0
    return x
"""
    test_source = """
from service import covered_target

def test_calls_covered_target():
    assert covered_target(1) == 2
"""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"service.py": source, "test_service.py": test_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        service_path = str(repo / "service.py")
        cache = CacheManager(CacheConfig(cache_dir))
        service_symbols = {
            symbol.name: symbol for symbol in cache.list_symbols_for_file(service_path)
        }
        covered_symbol = service_symbols["covered_target"]
        untested_symbol = service_symbols["untested_with_invariant"]
        covered_symbol_id = covered_symbol.id
        untested_symbol_id = untested_symbol.id
        covered_line = int(covered_symbol.start_line)

        test_path = str(repo / "test_service.py")
        test_symbols = {
            symbol.name: symbol for symbol in cache.list_symbols_for_file(test_path)
        }
        test_symbol_id = test_symbols["test_calls_covered_target"].id

        coverage_payload = {test_symbol_id: {service_path: [covered_line]}}
        coverage_file = repo / "gloggur-coverage.json"
        coverage_file.write_text(json.dumps(coverage_payload), encoding="utf8")

        ingest_result = runner.invoke(
            cli,
            ["coverage", "ingest", str(coverage_file), "--json"],
            env=env,
        )
        assert ingest_result.exit_code == 0, ingest_result.output
        ingest_payload = _parse_json_output(ingest_result.output)
        assert ingest_payload["tests_processed"] == 1
        assert ingest_payload["files_affected"] == 1
        assert int(ingest_payload["symbols_updated"]) >= 1

        guidance_result = runner.invoke(
            cli,
            ["guidance", covered_symbol_id, "--json"],
            env=env,
        )
        assert guidance_result.exit_code == 0, guidance_result.output
        guidance_payload = _parse_json_output(guidance_result.output)
        constraining_tests = guidance_payload["constraining_tests"]
        assert isinstance(constraining_tests, list)
        linkage = next(t for t in constraining_tests if t["test_symbol_id"] == test_symbol_id)
        assert linkage["constraint_strength"] == "strong"
        assert guidance_payload["untested_behaviors"] == []

        untested_guidance_result = runner.invoke(
            cli,
            ["guidance", untested_symbol_id, "--json"],
            env=env,
        )
        assert untested_guidance_result.exit_code == 0, untested_guidance_result.output
        untested_guidance_payload = _parse_json_output(untested_guidance_result.output)
        untested_behaviors = untested_guidance_payload["untested_behaviors"]
        assert isinstance(untested_behaviors, list)
        assert any("no dynamic test coverage" in warning for warning in untested_behaviors)
        assert any("strict invariants" in warning for warning in untested_behaviors)


def test_cli_detects_model_change_and_rebuilds_on_index() -> None:
    """Status/search should require reindex on model change; index should self-rebuild."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)

        env_model_a = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
            "GLOGGUR_LOCAL_MODEL": "model-a",
        }
        env_model_b = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
            "GLOGGUR_LOCAL_MODEL": "model-b",
        }

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env_model_a)
        assert first_index.exit_code == 0

        status_before = runner.invoke(cli, ["status", "--json"], env=env_model_a)
        assert status_before.exit_code == 0
        status_before_payload = _parse_json_output(status_before.output)
        assert status_before_payload["needs_reindex"] is False
        assert status_before_payload["expected_index_profile"] == "test:model-a"
        assert status_before_payload["cached_index_profile"] == "test:model-a"
        assert status_before_payload["resume_decision"] == "resume_ok"
        assert status_before_payload["resume_reason_codes"] == []
        assert status_before_payload["resume_fingerprint_match"] is True
        assert status_before_payload["last_success_resume_fingerprint"] == status_before_payload[
            "expected_resume_fingerprint"
        ]
        assert status_before_payload["last_success_resume_fingerprint_match"] is True
        assert isinstance(status_before_payload["last_success_resume_at"], str)
        assert isinstance(status_before_payload["tool_version"], str)
        assert status_before_payload["last_success_tool_version"] == status_before_payload[
            "tool_version"
        ]
        assert status_before_payload["last_success_tool_version_match"] is True
        first_success_fingerprint = status_before_payload["last_success_resume_fingerprint"]
        assert isinstance(first_success_fingerprint, str)

        status_changed = runner.invoke(cli, ["status", "--json"], env=env_model_b)
        assert status_changed.exit_code == 0
        status_changed_payload = _parse_json_output(status_changed.output)
        assert status_changed_payload["needs_reindex"] is True
        assert status_changed_payload["expected_index_profile"] == "test:model-b"
        assert status_changed_payload["cached_index_profile"] == "test:model-a"
        assert "embedding profile changed" in str(status_changed_payload["reindex_reason"])
        assert status_changed_payload["resume_decision"] == "reindex_required"
        assert status_changed_payload["resume_reason_codes"] == ["embedding_profile_changed"]
        assert status_changed_payload["resume_fingerprint_match"] is False
        assert (
            status_changed_payload["last_success_resume_fingerprint"]
            == first_success_fingerprint
        )
        assert status_changed_payload["last_success_resume_fingerprint_match"] is False
        assert status_changed_payload["last_success_tool_version_match"] is True

        search_changed = runner.invoke(cli, ["search", "add", "--json"], env=env_model_b)
        assert search_changed.exit_code == 0
        search_changed_payload = _parse_json_output(search_changed.output)
        assert search_changed_payload["results"] == []
        assert search_changed_payload["metadata"]["needs_reindex"] is True
        assert "embedding profile changed" in str(
            search_changed_payload["metadata"]["reindex_reason"]
        )
        assert search_changed_payload["metadata"]["resume_decision"] == "reindex_required"
        assert search_changed_payload["metadata"]["resume_reason_codes"] == [
            "embedding_profile_changed"
        ]
        assert search_changed_payload["metadata"]["resume_fingerprint_match"] is False
        assert (
            search_changed_payload["metadata"]["last_success_resume_fingerprint"]
            == first_success_fingerprint
        )
        assert search_changed_payload["metadata"]["last_success_resume_fingerprint_match"] is False
        assert search_changed_payload["metadata"]["last_success_tool_version_match"] is True

        rebuild_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env_model_b)
        assert rebuild_index.exit_code == 0
        rebuild_payload = _parse_json_output(rebuild_index.output)
        assert rebuild_payload["indexed_files"] == 1
        assert rebuild_payload["indexed_symbols"] > 0

        status_after = runner.invoke(cli, ["status", "--json"], env=env_model_b)
        assert status_after.exit_code == 0
        status_after_payload = _parse_json_output(status_after.output)
        assert status_after_payload["needs_reindex"] is False
        assert status_after_payload["expected_index_profile"] == "test:model-b"
        assert status_after_payload["cached_index_profile"] == "test:model-b"
        assert status_after_payload["resume_decision"] == "resume_ok"
        assert status_after_payload["resume_reason_codes"] == []
        assert status_after_payload["resume_fingerprint_match"] is True
        assert status_after_payload["last_success_resume_fingerprint"] == status_after_payload[
            "expected_resume_fingerprint"
        ]
        assert status_after_payload["last_success_resume_fingerprint_match"] is True
        assert (
            status_after_payload["last_success_tool_version"]
            == status_after_payload["tool_version"]
        )
        assert status_after_payload["last_success_tool_version_match"] is True
        assert status_after_payload["last_success_resume_fingerprint"] != first_success_fingerprint


def test_cli_index_reports_incremental_observability_and_prunes_deleted_files() -> None:
    """Index payload should expose incremental counters and prune deleted-file symbols."""
    runner = CliRunner()
    source = (
        "def keep_me(value: int) -> int:\n"
        "    return value + 1\n"
    )
    delete_source = (
        "def remove_me(value: int) -> int:\n"
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": source, "remove.py": delete_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0
        first_payload = _parse_json_output(first_index.output)
        for field in (
            "files_scanned",
            "files_changed",
            "files_removed",
            "symbols_added",
            "symbols_updated",
            "symbols_removed",
        ):
            assert field in first_payload
        assert int(first_payload["files_changed"]) == 2
        assert int(first_payload["files_removed"]) == 0
        assert int(first_payload["symbols_added"]) > 0
        initial_symbols = int(first_payload["indexed_symbols"])
        assert initial_symbols > 0

        status_before = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_before.exit_code == 0
        status_before_payload = _parse_json_output(status_before.output)
        total_before = int(status_before_payload["total_symbols"])
        assert total_before == initial_symbols

        (repo / "remove.py").unlink()
        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 0
        second_payload = _parse_json_output(second_index.output)
        assert int(second_payload["indexed_files"]) == 0
        assert int(second_payload["skipped_files"]) == 1
        assert int(second_payload["files_removed"]) == 1
        assert int(second_payload["symbols_removed"]) > 0
        assert int(second_payload["files_changed"]) == 0

        status_after = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_after.exit_code == 0
        status_after_payload = _parse_json_output(status_after.output)
        total_after = int(status_after_payload["total_symbols"])
        assert total_after < total_before


def test_cli_index_rename_does_not_leave_ghost_symbols() -> None:
    """Rename flows should not report success while retaining stale old-path symbols."""
    runner = CliRunner()
    old_source = (
        'def legacy_symbol(value: int) -> int:\n'
        '    """legacy rename ghost token old-only"""\n'
        "    return value + 1\n"
    )
    new_source = (
        'def modern_symbol(value: int) -> int:\n'
        '    """fresh rename token new-only"""\n'
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"module_old.py": old_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0

        old_path = repo / "module_old.py"
        new_path = repo / "module_new.py"
        old_path.rename(new_path)
        new_path.write_text(new_source, encoding="utf8")

        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 0, second_index.output
        second_payload = _parse_json_output(second_index.output)
        assert int(second_payload["indexed_files"]) == 1
        assert int(second_payload["files_removed"]) == 1
        assert int(second_payload["symbols_removed"]) > 0
        assert int(second_payload["symbols_added"]) > 0

        cache = CacheManager(CacheConfig(cache_dir))
        assert cache.get_file_metadata(str(old_path)) is None
        assert cache.list_symbols_for_file(str(old_path)) == []
        assert cache.get_file_metadata(str(new_path)) is not None


def test_cli_index_docstring_only_change_is_not_skipped() -> None:
    """Docstring-only edits should trigger reindex and refresh cached symbol docstrings."""
    runner = CliRunner()
    original_source = (
        "def summarize(value: int) -> int:\n"
        '    """doc-old-token-for-regression"""\n'
        "    return value + 1\n"
    )
    updated_source = (
        "def summarize(value: int) -> int:\n"
        '    """doc-new-token-for-regression"""\n'
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": original_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output

        file_path = repo / "sample.py"
        file_path.write_text(updated_source, encoding="utf8")
        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 0, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["indexed_files"]) == 1
        assert int(payload["files_changed"]) == 1
        assert int(payload["skipped_files"]) == 0
        assert int(payload["symbols_updated"]) >= 1

        cache = CacheManager(CacheConfig(cache_dir))
        symbols = cache.list_symbols_for_file(str(file_path))
        assert len(symbols) == 1
        docstring = symbols[0].docstring
        assert isinstance(docstring, str)
        assert "doc-new-token-for-regression" in docstring
        assert "doc-old-token-for-regression" not in docstring


def test_cli_index_reports_vector_metadata_mismatch_on_tampered_vector_map() -> None:
    """Index should fail closed with deterministic JSON when vectors.json drifts from cache symbols."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output
        _tamper_vector_id_map_drop_one_symbol(cache_dir)

        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 1, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["failed"]) == 1
        reasons = payload["failed_reasons"]
        assert isinstance(reasons, dict)
        assert reasons == {"vector_metadata_mismatch": 1}
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["vector_metadata_mismatch"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "vector_metadata_mismatch" in guidance
        assert isinstance(guidance["vector_metadata_mismatch"], list)
        assert guidance["vector_metadata_mismatch"]
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["type"] == "index_failure"
        assert error["code"] == "vector_metadata_mismatch"
        assert "Indexing completed with file-level failures." in str(error["detail"])


def test_cli_index_transient_vector_upsert_failure_is_not_sticky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient vector upsert failures should not leave sticky mismatch state across reruns."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        monkeypatch.setattr(cli_main.VectorStore, "_check_faiss", staticmethod(lambda: False))
        original_upsert = cli_main.VectorStore.upsert_vectors
        fail_state = {"triggered": False}

        def _fail_once(self: object, symbols: list[object]) -> None:
            if not fail_state["triggered"]:
                fail_state["triggered"] = True
                raise RuntimeError("simulated transient vector upsert failure")
            original_upsert(self, symbols)

        monkeypatch.setattr(cli_main.VectorStore, "upsert_vectors", _fail_once)

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 1, first_index.output
        first_payload = _parse_json_output(first_index.output)
        assert int(first_payload["indexed_symbols"]) == 0
        reasons = first_payload["failed_reasons"]
        assert isinstance(reasons, dict)
        assert reasons == {"storage_error": 1}
        assert "vector_metadata_mismatch" not in reasons

        cache = CacheManager(CacheConfig(cache_dir))
        sample_path = str(repo / "sample.py")
        assert cache.get_file_metadata(sample_path) is None
        assert cache.list_symbols_for_file(sample_path) == []

        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 0, second_index.output
        second_payload = _parse_json_output(second_index.output)
        assert int(second_payload["failed"]) == 0
        assert int(second_payload["indexed_symbols"]) > 0

        status_result = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_result.exit_code == 0, status_result.output
        status_payload = _parse_json_output(status_result.output)
        assert status_payload["resume_decision"] == "resume_ok"


def test_cli_single_file_index_fails_closed_on_tampered_vector_map() -> None:
    """Single-file index should not report success when vectors.json drifts from cache symbols."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output
        _tamper_vector_id_map_drop_one_symbol(cache_dir)

        second_index = runner.invoke(cli, ["index", str(repo / "sample.py"), "--json"], env=env)
        assert second_index.exit_code == 1, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["files_considered"]) == 1
        assert int(payload["failed"]) == 1
        reasons = payload["failed_reasons"]
        assert isinstance(reasons, dict)
        assert reasons == {"vector_metadata_mismatch": 1}
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["vector_metadata_mismatch"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "vector_metadata_mismatch" in guidance
        assert isinstance(guidance["vector_metadata_mismatch"], list)
        assert guidance["vector_metadata_mismatch"]


def test_cli_single_file_index_rename_prunes_missing_old_path_entries() -> None:
    """Single-file indexing should prune stale old-path rows after rename flows."""
    runner = CliRunner()
    old_source = (
        'def legacy_symbol(value: int) -> int:\n'
        '    """legacy single-file rename token old-only"""\n'
        "    return value + 1\n"
    )
    new_source = (
        'def modern_symbol(value: int) -> int:\n'
        '    """single-file rename token new-only"""\n'
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"module_old.py": old_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output

        old_path = repo / "module_old.py"
        new_path = repo / "module_new.py"
        old_path.rename(new_path)
        new_path.write_text(new_source, encoding="utf8")

        second_index = runner.invoke(cli, ["index", str(new_path), "--json"], env=env)
        assert second_index.exit_code == 0, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["files_considered"]) == 1
        assert int(payload["indexed_files"]) == 1
        assert int(payload["files_changed"]) == 1
        assert int(payload["files_removed"]) == 1
        assert int(payload["symbols_removed"]) > 0

        cache = CacheManager(CacheConfig(cache_dir))
        assert cache.get_file_metadata(str(old_path)) is None
        assert cache.list_symbols_for_file(str(old_path)) == []
        assert cache.get_file_metadata(str(new_path)) is not None


def test_cli_single_file_index_surfaces_stale_cleanup_error_with_failure_contract(
    monkeypatch,
) -> None:
    """Single-file index should fail closed with deterministic stale cleanup codes/remediation."""
    runner = CliRunner()
    keep_source = (
        "def keep_me(value: int) -> int:\n"
        "    return value + 1\n"
    )
    drop_source = (
        "def drop_me(value: int) -> int:\n"
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": keep_source, "drop.py": drop_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output

        stale_path = str(repo / "drop.py")
        (repo / "drop.py").unlink()
        original_delete = CacheManager.delete_file_metadata

        def _raise_on_stale_delete(self, path: str) -> None:
            if path == stale_path:
                raise OSError("simulated stale cleanup failure")
            original_delete(self, path)

        monkeypatch.setattr(CacheManager, "delete_file_metadata", _raise_on_stale_delete)
        second_index = runner.invoke(cli, ["index", str(repo / "keep.py"), "--json"], env=env)

        assert second_index.exit_code == 1, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["files_considered"]) == 1
        assert int(payload["failed"]) == 1
        reasons = payload["failed_reasons"]
        assert isinstance(reasons, dict)
        assert reasons == {"stale_cleanup_error": 1}
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["stale_cleanup_error"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "stale_cleanup_error" in guidance
        assert isinstance(guidance["stale_cleanup_error"], list)
        assert guidance["stale_cleanup_error"]


def test_cli_status_and_index_self_heal_corrupted_cache_idempotently() -> None:
    """Status/index should self-heal a corrupted DB once and remain stable afterwards."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        db_path = Path(cache_dir) / "index.db"
        db_path.write_bytes(b"broken sqlite bytes")
        Path(f"{db_path}-wal").write_bytes(b"broken wal")
        Path(f"{db_path}-shm").write_bytes(b"broken shm")

        first_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert first_status.exit_code == 0
        assert "Cache corruption detected at" in first_status.output
        first_payload = _parse_json_output(first_status.output)
        assert first_payload["needs_reindex"] is True
        assert int(first_payload["total_symbols"]) == 0
        quarantined_after_first = sorted(
            path.name for path in Path(cache_dir).iterdir() if ".corrupt." in path.name
        )
        assert quarantined_after_first
        assert not Path(f"{db_path}-wal").exists()
        assert not Path(f"{db_path}-shm").exists()

        second_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert second_status.exit_code == 0
        assert "Cache corruption detected at" not in second_status.output
        second_payload = _parse_json_output(second_status.output)
        assert second_payload["needs_reindex"] is True
        quarantined_after_second = sorted(
            path.name for path in Path(cache_dir).iterdir() if ".corrupt." in path.name
        )
        assert quarantined_after_second == quarantined_after_first

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0
        index_payload = _parse_json_output(index_result.output)
        assert index_payload["indexed_files"] == 1
        assert index_payload["indexed_symbols"] > 0

        final_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert final_status.exit_code == 0
        final_payload = _parse_json_output(final_status.output)
        assert final_payload["needs_reindex"] is False
        assert int(final_payload["total_symbols"]) > 0


def test_cli_search_self_heals_corrupted_cache() -> None:
    """Search should recover from cache corruption and return reindex metadata, not crash."""
    runner = CliRunner()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {"GLOGGUR_CACHE_DIR": cache_dir}
    db_path = Path(cache_dir) / "index.db"
    db_path.write_bytes(b"broken sqlite bytes")

    first_search = runner.invoke(cli, ["search", "add", "--json"], env=env)
    assert first_search.exit_code == 0
    assert "Cache corruption detected at" in first_search.output
    first_payload = _parse_json_output(first_search.output)
    assert first_payload["results"] == []
    metadata = first_payload["metadata"]
    assert isinstance(metadata, dict)
    assert int(metadata["total_results"]) == 0

    second_search = runner.invoke(cli, ["search", "add", "--json"], env=env)
    assert second_search.exit_code == 0
    assert "Cache corruption detected at" not in second_search.output


def test_cli_clear_cache_self_heals_corrupted_cache() -> None:
    """clear-cache should recover from corruption instead of failing with a DB error."""
    runner = CliRunner()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {"GLOGGUR_CACHE_DIR": cache_dir}
    db_path = Path(cache_dir) / "index.db"
    db_path.write_bytes(b"broken sqlite bytes")

    result = runner.invoke(cli, ["clear-cache", "--json"], env=env)
    assert result.exit_code == 0
    assert "Cache corruption detected at" in result.output
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True


def test_cli_artifact_publish_packages_cache_with_manifest() -> None:
    """artifact publish should emit deterministic checksums and embed manifest in archive."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        artifacts_dir = Path(tempfile.mkdtemp(prefix="gloggur-artifacts-"))
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        publish_result = runner.invoke(
            cli,
            ["artifact", "publish", "--json", "--destination", str(artifacts_dir)],
            env=env,
        )
        assert publish_result.exit_code == 0, publish_result.output
        payload = _parse_json_output(publish_result.output)
        assert payload["published"] is True
        artifact_path = Path(str(payload["artifact_path"]))
        assert artifact_path.exists()
        assert payload["artifact_uri"] == artifact_path.resolve().as_uri()
        assert payload["archive_sha256"] == _sha256_for_file(artifact_path)
        assert int(payload["archive_bytes"]) == artifact_path.stat().st_size
        manifest = payload["manifest"]
        assert isinstance(manifest, dict)
        assert manifest["manifest_schema_version"] == "1"
        assert int(manifest["files_total"]) > 0
        manifest_payload = json.dumps(manifest, sort_keys=True, indent=2).encode("utf8") + b"\n"
        assert payload["manifest_sha256"] == hashlib.sha256(manifest_payload).hexdigest()

        with tarfile.open(artifact_path, "r:gz") as archive:
            names = sorted(archive.getnames())
            assert "manifest.json" in names
            assert any(name.startswith("cache/") for name in names)
            manifest_member = archive.extractfile("manifest.json")
            assert manifest_member is not None
            archived_manifest = json.loads(manifest_member.read().decode("utf8"))
            assert archived_manifest == manifest


def test_cli_artifact_publish_fails_closed_for_unsupported_destination_scheme() -> None:
    """artifact publish should return structured failure codes for unsupported destination schemes."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(
            cli,
            ["artifact", "publish", "--json", "--destination", "ftp://example.com/upload"],
            env=env,
        )
        assert result.exit_code == 1, result.output
        payload = _parse_json_output(result.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["type"] == "cli_contract_error"
        assert error["code"] == "artifact_destination_unsupported"
        assert payload["failure_codes"] == ["artifact_destination_unsupported"]


def test_cli_artifact_publish_fails_closed_for_uninitialized_source_cache() -> None:
    """artifact publish should fail closed when source cache has no index metadata."""
    runner = CliRunner()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {"GLOGGUR_CACHE_DIR": cache_dir}
    destination = Path(tempfile.mkdtemp(prefix="gloggur-artifacts-")) / "artifact.tar.gz"

    result = runner.invoke(
        cli,
        ["artifact", "publish", "--json", "--destination", str(destination)],
        env=env,
    )
    assert result.exit_code == 1, result.output
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "cli_contract_error"
    assert error["code"] == "artifact_source_uninitialized"
    assert payload["failure_codes"] == ["artifact_source_uninitialized"]


def test_cli_artifact_publish_fails_closed_when_destination_exists_without_overwrite() -> None:
    """artifact publish should not silently overwrite existing destination artifacts."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}
        destination_path = Path(tempfile.mkdtemp(prefix="gloggur-artifacts-")) / "cache-artifact.tgz"

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        first_publish = runner.invoke(
            cli,
            [
                "artifact",
                "publish",
                "--json",
                "--destination",
                str(destination_path),
            ],
            env=env,
        )
        assert first_publish.exit_code == 0, first_publish.output

        second_publish = runner.invoke(
            cli,
            [
                "artifact",
                "publish",
                "--json",
                "--destination",
                str(destination_path),
            ],
            env=env,
        )
        assert second_publish.exit_code == 1, second_publish.output
        payload = _parse_json_output(second_publish.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["type"] == "cli_contract_error"
        assert error["code"] == "artifact_destination_exists"
        assert payload["failure_codes"] == ["artifact_destination_exists"]


def test_cli_artifact_publish_supports_uploader_command_template() -> None:
    """artifact publish should support external uploader-command transport with stable metadata."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        uploader_dir = Path(tempfile.mkdtemp(prefix="gloggur-uploader-"))
        uploader_script = uploader_dir / "copy_uploader.py"
        uploader_script.write_text(
            "from pathlib import Path\n"
            "import shutil\n"
            "import sys\n"
            "src = Path(sys.argv[1])\n"
            "dest = Path(sys.argv[2])\n"
            "dest.parent.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copyfile(src, dest)\n",
            encoding="utf8",
        )
        uploaded_artifact = uploader_dir / "uploaded" / "cache-artifact.tgz"
        uploader_template = (
            f"{sys.executable} {uploader_script} "
            "{artifact_path} {destination}"
        )
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        publish_result = runner.invoke(
            cli,
            [
                "artifact",
                "publish",
                "--json",
                "--destination",
                str(uploaded_artifact),
                "--uploader-command",
                uploader_template,
            ],
            env=env,
        )
        assert publish_result.exit_code == 0, publish_result.output
        payload = _parse_json_output(publish_result.output)
        assert payload["published"] is True
        assert payload["transport"] == "uploader_command"
        assert payload["artifact_destination"] == str(uploaded_artifact)
        assert payload["artifact_path"] is None
        assert payload["artifact_uri"] == str(uploaded_artifact)
        assert uploaded_artifact.exists()
        assert payload["archive_sha256"] == _sha256_for_file(uploaded_artifact)
        assert int(payload["archive_bytes"]) == uploaded_artifact.stat().st_size
        uploader = payload["uploader"]
        assert isinstance(uploader, dict)
        assert uploader["mode"] == "uploader_command"
        assert uploader["destination"] == str(uploaded_artifact)
        validate_result = runner.invoke(
            cli,
            ["artifact", "validate", "--json", "--artifact", str(uploaded_artifact)],
        )
        assert validate_result.exit_code == 0, validate_result.output


def test_cli_artifact_publish_fails_closed_for_uploader_command_failure() -> None:
    """artifact publish should surface structured failure codes for uploader-command errors."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        uploader_dir = Path(tempfile.mkdtemp(prefix="gloggur-uploader-"))
        uploader_script = uploader_dir / "fail_uploader.py"
        uploader_script.write_text(
            "import sys\n"
            "sys.stderr.write('simulated uploader failure\\n')\n"
            "raise SystemExit(7)\n",
            encoding="utf8",
        )
        uploader_template = (
            f"{sys.executable} {uploader_script} "
            "{artifact_path} {destination}"
        )
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(
            cli,
            [
                "artifact",
                "publish",
                "--json",
                "--destination",
                "https://example.com/upload",
                "--uploader-command",
                uploader_template,
            ],
            env=env,
        )
        assert result.exit_code == 1, result.output
        payload = _parse_json_output(result.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["type"] == "cli_contract_error"
        assert error["code"] == "artifact_uploader_failed"
        assert payload["failure_codes"] == ["artifact_uploader_failed"]


def test_cli_artifact_publish_supports_direct_http_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """artifact publish should support direct HTTP PUT upload with stable metadata and headers."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    destination_url = "https://artifacts.example.test/upload/cache-artifact.tgz"
    captured: dict[str, object] = {}

    class _FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.headers = {
                "Content-Type": "application/json",
                "ETag": "test-etag",
            }

        def read(self) -> bytes:
            return b'{"uploaded": true}'

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def _fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        assert isinstance(request, cli_main.urllib_request.Request)
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        return _FakeResponse()

    monkeypatch.setattr(cli_main.urllib_request, "urlopen", _fake_urlopen)

    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        publish_result = runner.invoke(
            cli,
            [
                "artifact",
                "publish",
                "--json",
                "--destination",
                destination_url,
            ],
            env=env,
        )
        assert publish_result.exit_code == 0, publish_result.output
        payload = _parse_json_output(publish_result.output)
        assert payload["published"] is True
        assert payload["transport"] == "http_put"
        assert payload["artifact_destination"] == destination_url
        assert payload["artifact_path"] is None
        assert payload["artifact_uri"] == destination_url
        http_upload = payload["http_upload"]
        assert isinstance(http_upload, dict)
        assert http_upload["mode"] == "http_put"
        assert http_upload["destination"] == destination_url
        assert int(http_upload["status_code"]) == 200
        uploaded_body = captured["body"]
        assert isinstance(uploaded_body, bytes)
        assert hashlib.sha256(uploaded_body).hexdigest() == payload["archive_sha256"]
        assert len(uploaded_body) == int(payload["archive_bytes"])
        headers = captured["headers"]
        assert isinstance(headers, dict)
        assert headers["X-gloggur-archive-sha256"] == payload["archive_sha256"]
        assert headers["X-gloggur-archive-bytes"] == str(payload["archive_bytes"])
        assert headers["X-gloggur-manifest-sha256"] == payload["manifest_sha256"]
        uploaded_path = Path(tempfile.mkdtemp(prefix="gloggur-http-upload-")) / "uploaded-artifact.tgz"
        uploaded_path.write_bytes(uploaded_body)
        validate_result = runner.invoke(
            cli,
            ["artifact", "validate", "--json", "--artifact", str(uploaded_path)],
        )
        assert validate_result.exit_code == 0, validate_result.output


def test_cli_artifact_publish_fails_closed_for_http_upload_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """artifact publish should emit a stable contract for non-2xx HTTP upload responses."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    destination_url = "https://artifacts.example.test/upload/cache-artifact.tgz"

    def _failing_urlopen(request: object, timeout: float) -> object:
        assert isinstance(request, cli_main.urllib_request.Request)
        raise urllib.error.HTTPError(
            destination_url,
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(b"forbidden"),
        )

    monkeypatch.setattr(cli_main.urllib_request, "urlopen", _failing_urlopen)

    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        result = runner.invoke(
            cli,
            [
                "artifact",
                "publish",
                "--json",
                "--destination",
                destination_url,
            ],
            env=env,
        )
        assert result.exit_code == 1, result.output
        payload = _parse_json_output(result.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["type"] == "cli_contract_error"
        assert error["code"] == "artifact_http_upload_failed"
        assert payload["failure_codes"] == ["artifact_http_upload_failed"]


def test_cli_artifact_validate_reports_verified_metadata_after_publish() -> None:
    """artifact validate should verify published archive integrity and manifest metadata."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        artifacts_dir = Path(tempfile.mkdtemp(prefix="gloggur-artifacts-"))
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output

        publish_result = runner.invoke(
            cli,
            ["artifact", "publish", "--json", "--destination", str(artifacts_dir)],
            env=env,
        )
        assert publish_result.exit_code == 0, publish_result.output
        publish_payload = _parse_json_output(publish_result.output)
        artifact_path = Path(str(publish_payload["artifact_path"]))

        validate_result = runner.invoke(
            cli,
            ["artifact", "validate", "--json", "--artifact", str(artifact_path)],
        )
        assert validate_result.exit_code == 0, validate_result.output
        payload = _parse_json_output(validate_result.output)
        assert payload["valid"] is True
        assert payload["artifact_uri"] == artifact_path.resolve().as_uri()
        assert payload["archive_sha256"] == _sha256_for_file(artifact_path)
        assert int(payload["archive_bytes"]) == artifact_path.stat().st_size
        assert payload["manifest"] == publish_payload["manifest"]
        verification = payload["file_hash_verification"]
        assert verification["enabled"] is True
        assert verification["checked_files"] == payload["manifest"]["files_total"]
        assert verification["checked_bytes"] == payload["manifest"]["bytes_total"]


def test_cli_artifact_validate_fails_closed_for_missing_artifact_path() -> None:
    """artifact validate should emit a stable contract when the requested archive is missing."""
    runner = CliRunner()
    missing_artifact = Path(tempfile.mkdtemp(prefix="gloggur-artifacts-")) / "missing.tar.gz"

    result = runner.invoke(
        cli,
        ["artifact", "validate", "--json", "--artifact", str(missing_artifact)],
    )
    assert result.exit_code == 1, result.output
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "cli_contract_error"
    assert error["code"] == "artifact_path_missing"
    assert payload["failure_codes"] == ["artifact_path_missing"]


def test_cli_artifact_restore_restores_cache_for_downstream_search() -> None:
    """artifact restore should recreate a reusable cache directory for downstream status/search."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        source_cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-source-")
        _write_fallback_marker(source_cache_dir)
        artifacts_dir = Path(tempfile.mkdtemp(prefix="gloggur-artifacts-"))
        source_env = {"GLOGGUR_CACHE_DIR": source_cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=source_env)
        assert index_result.exit_code == 0, index_result.output

        publish_result = runner.invoke(
            cli,
            ["artifact", "publish", "--json", "--destination", str(artifacts_dir)],
            env=source_env,
        )
        assert publish_result.exit_code == 0, publish_result.output
        publish_payload = _parse_json_output(publish_result.output)
        artifact_path = Path(str(publish_payload["artifact_path"]))

        restore_root = Path(tempfile.mkdtemp(prefix="gloggur-restored-parent-"))
        restore_dir = restore_root / "restored-cache"
        restore_result = runner.invoke(
            cli,
            [
                "artifact",
                "restore",
                "--json",
                "--artifact",
                str(artifact_path),
                "--destination",
                str(restore_dir),
            ],
        )
        assert restore_result.exit_code == 0, restore_result.output
        restore_payload = _parse_json_output(restore_result.output)
        assert restore_payload["restored"] is True
        assert restore_payload["destination_cache_dir"] == str(restore_dir)
        assert restore_payload["restored_files"] == publish_payload["manifest"]["files_total"]
        assert restore_payload["restored_bytes"] == publish_payload["manifest"]["bytes_total"]
        assert (restore_dir / "index.db").exists()

        restored_env = {"GLOGGUR_CACHE_DIR": str(restore_dir)}
        status_result = runner.invoke(cli, ["status", "--json"], env=restored_env)
        assert status_result.exit_code == 0, status_result.output
        status_payload = _parse_json_output(status_result.output)
        assert int(status_payload["total_symbols"]) > 0

        search_result = runner.invoke(
            cli,
            ["search", "add", "--json", "--top-k", "3"],
            env=restored_env,
        )
        assert search_result.exit_code == 0, search_result.output
        search_payload = _parse_json_output(search_result.output)
        assert int(search_payload["metadata"]["total_results"]) > 0


def test_cli_artifact_restore_fails_closed_when_destination_exists_without_overwrite() -> None:
    """artifact restore should not silently replace an existing cache directory."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        source_cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-source-")
        _write_fallback_marker(source_cache_dir)
        artifacts_dir = Path(tempfile.mkdtemp(prefix="gloggur-artifacts-"))
        source_env = {"GLOGGUR_CACHE_DIR": source_cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=source_env)
        assert index_result.exit_code == 0, index_result.output

        publish_result = runner.invoke(
            cli,
            ["artifact", "publish", "--json", "--destination", str(artifacts_dir)],
            env=source_env,
        )
        assert publish_result.exit_code == 0, publish_result.output
        artifact_path = Path(str(_parse_json_output(publish_result.output)["artifact_path"]))

        restore_dir = Path(tempfile.mkdtemp(prefix="gloggur-restored-cache-"))
        result = runner.invoke(
            cli,
            [
                "artifact",
                "restore",
                "--json",
                "--artifact",
                str(artifact_path),
                "--destination",
                str(restore_dir),
            ],
        )
        assert result.exit_code == 1, result.output
        payload = _parse_json_output(result.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["type"] == "cli_contract_error"
        assert error["code"] == "artifact_restore_destination_exists"
        assert payload["failure_codes"] == ["artifact_restore_destination_exists"]


def test_cli_inspect_self_heals_corrupted_cache() -> None:
    """Inspect command should run after automatic corruption recovery."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}
        db_path = Path(cache_dir) / "index.db"
        db_path.write_bytes(b"broken sqlite bytes")

        result = runner.invoke(cli, ["inspect", str(repo), "--json"], env=env)
        assert result.exit_code == 0
        assert "Cache corruption detected at" in result.output
        payload = _parse_json_output(result.output)
        assert int(payload["reports_total"]) >= 0


def test_cli_inspect_defaults_to_source_focus_with_opt_in_full_audit() -> None:
    """Inspect should default to src focus and include tests/scripts only when explicitly requested."""
    runner = CliRunner()
    source_no_docstring = (
        "def sample(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "src/main.py": source_no_docstring,
                "tests/test_main.py": source_no_docstring,
                "scripts/tool.py": source_no_docstring,
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        default_result = runner.invoke(cli, ["inspect", str(repo), "--json", "--force"], env=env)
        assert default_result.exit_code == 0, default_result.output
        default_payload = _parse_json_output(default_result.output)
        default_scope = default_payload["inspect_scope"]
        assert isinstance(default_scope, dict)
        assert default_scope["include_tests"] is False
        assert default_scope["include_scripts"] is False
        default_summary = default_payload["warning_summary"]
        assert isinstance(default_summary, dict)
        default_by_class = default_summary["by_path_class"]
        assert isinstance(default_by_class, dict)
        assert int(default_by_class["src"]) > 0
        assert int(default_by_class["tests"]) == 0
        assert int(default_by_class["scripts"]) == 0
        default_warnings = default_payload["warnings"]
        assert isinstance(default_warnings, list)
        assert all(item["path_class"] == "src" for item in default_summary["top_files"])

        full_result = runner.invoke(
            cli,
            [
                "inspect",
                str(repo),
                "--json",
                "--force",
                "--include-tests",
                "--include-scripts",
            ],
            env=env,
        )
        assert full_result.exit_code == 0, full_result.output
        full_payload = _parse_json_output(full_result.output)
        full_scope = full_payload["inspect_scope"]
        assert isinstance(full_scope, dict)
        assert full_scope["include_tests"] is True
        assert full_scope["include_scripts"] is True
        full_summary = full_payload["warning_summary"]
        assert isinstance(full_summary, dict)
        full_by_class = full_summary["by_path_class"]
        assert isinstance(full_by_class, dict)
        assert int(full_by_class["src"]) > 0
        assert int(full_by_class["tests"]) > 0
        assert int(full_by_class["scripts"]) > 0


def test_cli_inspect_explicit_tests_path_remains_included_without_flags() -> None:
    """Explicit tests-path inspect should not be filtered out by default src-focus behavior."""
    runner = CliRunner()
    source_no_docstring = (
        "def sample(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"tests/test_main.py": source_no_docstring})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        result = runner.invoke(
            cli,
            ["inspect", str(repo / "tests"), "--json", "--force"],
            env=env,
        )
        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        summary = payload["warning_summary"]
        assert isinstance(summary, dict)
        by_class = summary["by_path_class"]
        assert isinstance(by_class, dict)
        assert int(by_class["tests"]) > 0
        assert int(by_class["src"]) == 0


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/gloggur/config.py",
        "src/gloggur/cli/main.py",
        "src/gloggur/embeddings/gemini.py",
        "src/gloggur/indexer/concurrency.py",
        "src/gloggur/indexer/indexer.py",
        "src/gloggur/watch/service.py",
    ],
)
def test_cli_inspect_reports_no_missing_docstrings_for_recent_f11_hotspots(
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    """Recently cleaned hotspot files should stay free of Missing docstring warnings."""
    runner = CliRunner()
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda config: None,
    )
    env = {
        "GLOGGUR_CACHE_DIR": tempfile.mkdtemp(prefix="gloggur-cache-"),
        "GLOGGUR_EMBEDDING_PROVIDER": "",
    }

    result = runner.invoke(cli, ["inspect", relative_path, "--json", "--force"], env=env)

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    missing = [
        item["symbol_id"]
        for item in payload["warnings"]
        if "Missing docstring" in item.get("warnings", [])
    ]
    assert missing == []


def test_cli_inspect_reuses_cached_warning_reports_for_unchanged_files() -> None:
    """Second inspect without --force should preserve actionable warnings from cache."""
    runner = CliRunner()
    source_no_docstring = (
        "def sample(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"src/main.py": source_no_docstring})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first = runner.invoke(cli, ["inspect", str(repo), "--json", "--force"], env=env)
        assert first.exit_code == 0, first.output
        first_payload = _parse_json_output(first.output)
        assert int(first_payload["total"]) > 0
        assert int(first_payload["cached_files_reused"]) == 0
        first_warning_ids = sorted(str(item["symbol_id"]) for item in first_payload["warnings"])

        second = runner.invoke(cli, ["inspect", str(repo), "--json"], env=env)
        assert second.exit_code == 0, second.output
        second_payload = _parse_json_output(second.output)
        assert int(second_payload["skipped_files"]) == 1
        assert int(second_payload["cached_files_reused"]) == 1
        assert int(second_payload["cached_warning_reports_reused"]) == int(first_payload["total"])
        second_warning_ids = sorted(str(item["symbol_id"]) for item in second_payload["warnings"])
        assert second_warning_ids == first_warning_ids
        assert int(second_payload["total"]) == int(first_payload["total"])
        assert second_payload["failure_codes"] == []


def test_cli_inspect_reuses_cached_clean_reports_for_unchanged_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second inspect without --force should preserve clean semantic reports from cache."""
    runner = CliRunner()
    source = (
        "def sample(value: int) -> int:\n"
        '    """Increment the value."""\n'
        "    return value + 1\n"
    )

    class FakeEmbedding(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [1.0, 0.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            _ = texts
            return [[1.0, 0.0], [1.0, 0.0]]

        def get_dimension(self) -> int:
            return 2

    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda config: FakeEmbedding(),
    )

    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"src/main.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config_path = Path(repo) / "gloggur.yaml"
        config_path.write_text(
            "docstring_semantic_threshold: 0.2\n"
            "docstring_semantic_min_chars: 0\n",
            encoding="utf8",
        )
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first = runner.invoke(
            cli,
            ["inspect", str(repo), "--json", "--force", "--config", str(config_path)],
            env=env,
        )
        assert first.exit_code == 0, first.output
        first_payload = _parse_json_output(first.output)
        assert int(first_payload["total"]) == 0
        assert int(first_payload["reports_total"]) == 1
        assert int(first_payload["cached_reports_reused"]) == 0

        second = runner.invoke(
            cli,
            ["inspect", str(repo), "--json", "--config", str(config_path)],
            env=env,
        )
        assert second.exit_code == 0, second.output
        second_payload = _parse_json_output(second.output)
        assert int(second_payload["skipped_files"]) == 1
        assert int(second_payload["cached_files_reused"]) == 1
        assert int(second_payload["cached_reports_reused"]) == 1
        assert int(second_payload["cached_warning_reports_reused"]) == 0
        assert int(second_payload["total"]) == 0
        assert int(second_payload["reports_total"]) == 1
        reports = second_payload["reports"]
        assert isinstance(reports, list)
        assert len(reports) == 1
        score_metadata = reports[0]["score_metadata"]
        assert isinstance(score_metadata, dict)
        assert score_metadata["cached_report_reuse"] is True


def test_cli_inspect_clears_stale_cached_reports_after_file_is_fixed() -> None:
    """A fixed file should not resurrect stale cached warnings on the next unchanged inspect."""
    runner = CliRunner()
    source_no_docstring = (
        "def sample(value: int) -> int:\n"
        "    return value + 1\n"
    )
    source_with_docstring = (
        "def sample(value: int) -> int:\n"
        '    """Increment the value."""\n'
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"src/main.py": source_no_docstring})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first = runner.invoke(cli, ["inspect", str(repo), "--json", "--force"], env=env)
        assert first.exit_code == 0, first.output
        first_payload = _parse_json_output(first.output)
        assert int(first_payload["total"]) == 1

        (Path(repo) / "src" / "main.py").write_text(source_with_docstring, encoding="utf8")
        second = runner.invoke(cli, ["inspect", str(repo), "--json", "--force"], env=env)
        assert second.exit_code == 0, second.output
        second_payload = _parse_json_output(second.output)
        assert int(second_payload["total"]) == 0

        third = runner.invoke(cli, ["inspect", str(repo), "--json"], env=env)
        assert third.exit_code == 0, third.output
        third_payload = _parse_json_output(third.output)
        assert int(third_payload["skipped_files"]) == 1
        assert int(third_payload["cached_files_reused"]) == 1
        assert int(third_payload["cached_warning_reports_reused"]) == 0
        assert int(third_payload["total"]) == 0
        assert int(third_payload["reports_total"]) == int(second_payload["reports_total"])


def test_cli_inspect_allow_partial_emits_failure_contract_for_decode_errors() -> None:
    """Inspect should report deterministic failure codes/guidance when partial failures are allowed."""
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "src/main.py": (
                    "def sample(value: int) -> int:\n"
                    "    return value + 1\n"
                )
            }
        )
        (repo / "src" / "broken.py").write_bytes(b"\xff\xfe\x00\x00")
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        result = runner.invoke(
            cli,
            ["inspect", str(repo), "--json", "--force", "--allow-partial"],
            env=env,
        )
        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        assert int(payload["failed"]) == 1
        assert payload["allow_partial"] is True
        assert payload["allow_partial_applied"] is True
        reasons = payload["failed_reasons"]
        assert isinstance(reasons, dict)
        assert reasons == {"decode_error": 1}
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["decode_error"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "decode_error" in guidance
        assert isinstance(guidance["decode_error"], list)
        assert guidance["decode_error"]
        samples = payload["failed_samples"]
        assert isinstance(samples, list)
        assert samples


def test_cli_inspect_fails_closed_without_allow_partial_on_decode_errors() -> None:
    """Inspect should exit non-zero when file failures occur without explicit partial override."""
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "src/main.py": (
                    "def sample(value: int) -> int:\n"
                    "    return value + 1\n"
                )
            }
        )
        (repo / "src" / "broken.py").write_bytes(b"\xff\xfe\x00\x00")
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        result = runner.invoke(
            cli,
            ["inspect", str(repo), "--json", "--force"],
            env=env,
        )
        assert result.exit_code == 1, result.output
        payload = _parse_json_output(result.output)
        assert int(payload["failed"]) == 1
        assert payload["allow_partial"] is False
        assert payload["allow_partial_applied"] is False
        assert payload["failure_codes"] == ["decode_error"]
        error = payload["error"]
        assert isinstance(error, dict)
        assert error["type"] == "inspect_failure"
        assert error["code"] == "decode_error"
        assert "Inspect completed with file-level failures." in str(error["detail"])


def test_cli_inspect_warning_summary_payload_schema_is_stable() -> None:
    """Inspect payload should keep stable summary fields and value types for automations."""
    runner = CliRunner()
    source_no_docstring = (
        "def sample(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"src/main.py": source_no_docstring})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        result = runner.invoke(cli, ["inspect", str(repo), "--json", "--force"], env=env)
        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)

        for legacy_field in (
            "warnings",
            "total",
            "reports",
            "reports_total",
            "files_considered",
            "inspected_files",
            "skipped_files",
        ):
            assert legacy_field in payload

        assert "inspect_payload_schema_version" in payload
        assert payload["inspect_payload_schema_version"] == "1"
        schema_policy = payload["inspect_payload_schema_policy"]
        assert isinstance(schema_policy, dict)
        assert set(schema_policy.keys()) == {
            "policy_version",
            "bump_required_for",
            "bump_not_required_for",
        }
        assert schema_policy["policy_version"] == "1"
        assert isinstance(schema_policy["bump_required_for"], list)
        assert isinstance(schema_policy["bump_not_required_for"], list)

        scope = payload["inspect_scope"]
        assert isinstance(scope, dict)
        assert set(scope.keys()) == {"default_src_focus", "include_tests", "include_scripts"}
        assert isinstance(scope["default_src_focus"], bool)
        assert isinstance(scope["include_tests"], bool)
        assert isinstance(scope["include_scripts"], bool)
        assert payload["allow_partial"] is False
        assert payload["allow_partial_applied"] is False

        summary = payload["warning_summary"]
        assert isinstance(summary, dict)
        assert set(summary.keys()) == {
            "total_warnings",
            "by_warning_type",
            "by_path_class",
            "reports_by_path_class",
            "top_files",
        }
        assert isinstance(summary["total_warnings"], int)
        assert isinstance(summary["by_warning_type"], dict)
        assert isinstance(summary["by_path_class"], dict)
        assert isinstance(summary["reports_by_path_class"], dict)
        assert isinstance(summary["top_files"], list)

        by_path_class = summary["by_path_class"]
        assert set(by_path_class.keys()) == {"src", "tests", "scripts", "other"}
        assert all(isinstance(value, int) for value in by_path_class.values())

        reports_by_path_class = summary["reports_by_path_class"]
        assert set(reports_by_path_class.keys()) == {"src", "tests", "scripts", "other"}
        assert all(isinstance(value, int) for value in reports_by_path_class.values())

        assert summary["top_files"], "expected at least one top file entry"
        for item in summary["top_files"]:
            assert isinstance(item, dict)
            assert set(item.keys()) == {"file", "warnings", "path_class"}
            assert isinstance(item["file"], str)
            assert isinstance(item["warnings"], int)
            assert item["path_class"] in {"src", "tests", "scripts", "other"}

        assert payload["failure_codes"] == []
        assert payload["failure_guidance"] == {}


def test_cli_inspect_calibrated_threshold_reduces_low_semantic_warning_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibrated threshold should reduce low-semantic warnings versus legacy 0.2."""
    runner = CliRunner()

    class DeterministicInspectEmbeddingProvider(EmbeddingProvider):
        """Deterministic provider to stabilize inspect warning-count regression checks."""

        @staticmethod
        def _vector_for_score(score: float) -> list[float]:
            safe_score = max(0.0, min(1.0, score))
            tail = (1.0 - (safe_score * safe_score)) ** 0.5
            return [safe_score, tail]

        def embed_text(self, text: str) -> list[float]:
            lowered = text.lower()
            if "doc_hi" in lowered:
                return [1.0, 0.0]
            if "doc_mid_a" in lowered:
                return [1.0, 0.0]
            if "doc_mid_b" in lowered:
                return [1.0, 0.0]
            if "doc_low" in lowered:
                return [1.0, 0.0]
            if "code_hi" in lowered:
                return self._vector_for_score(0.90)
            if "code_mid_a" in lowered:
                return self._vector_for_score(0.15)
            if "code_mid_b" in lowered:
                return self._vector_for_score(0.16)
            if "code_low" in lowered:
                return self._vector_for_score(0.02)
            return [1.0, 0.0]

        def embed_batch(self, texts) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda _config, **_kwargs: DeterministicInspectEmbeddingProvider(),
    )

    source = (
        "def high_signal() -> int:\n"
        "    \"\"\"DOC_HI descriptive text.\"\"\"\n"
        "    value = 1 + 1  # CODE_HI marker with stable semantic alignment padding text.\n"
        "    return value\n\n"
        "def medium_signal_a() -> int:\n"
        "    \"\"\"DOC_MID_A descriptive text.\"\"\"\n"
        "    value = 2 + 2  # CODE_MID_A marker with deterministic medium score padding.\n"
        "    return value\n\n"
        "def medium_signal_b() -> int:\n"
        "    \"\"\"DOC_MID_B descriptive text.\"\"\"\n"
        "    value = 3 + 3  # CODE_MID_B marker with deterministic medium score padding.\n"
        "    return value\n\n"
        "def low_signal() -> int:\n"
        "    \"\"\"DOC_LOW descriptive text.\"\"\"\n"
        "    value = 4 + 4  # CODE_LOW marker with deterministic low score padding text.\n"
        "    return value\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"src/sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}
        legacy_config = repo / "legacy-threshold.yaml"
        legacy_config.write_text(
            "docstring_semantic_threshold: 0.2\n",
            encoding="utf8",
        )

        calibrated_result = runner.invoke(
            cli,
            ["inspect", str(repo), "--json", "--force"],
            env=env,
        )
        assert calibrated_result.exit_code == 0, calibrated_result.output
        calibrated_payload = _parse_json_output(calibrated_result.output)
        calibrated_summary = calibrated_payload["warning_summary"]
        assert isinstance(calibrated_summary, dict)
        calibrated_by_type = calibrated_summary["by_warning_type"]
        assert isinstance(calibrated_by_type, dict)
        calibrated_low = int(calibrated_by_type.get("Low semantic similarity", 0))

        legacy_result = runner.invoke(
            cli,
            ["inspect", str(repo), "--json", "--force", "--config", str(legacy_config)],
            env=env,
        )
        assert legacy_result.exit_code == 0, legacy_result.output
        legacy_payload = _parse_json_output(legacy_result.output)
        legacy_summary = legacy_payload["warning_summary"]
        assert isinstance(legacy_summary, dict)
        legacy_by_type = legacy_summary["by_warning_type"]
        assert isinstance(legacy_by_type, dict)
        legacy_low = int(legacy_by_type.get("Low semantic similarity", 0))

        assert legacy_low == 3
        assert calibrated_low == 1
        assert calibrated_low <= int(legacy_low * 0.6)


def test_cli_index_unchanged_run_skips_all_files_and_is_faster() -> None:
    """F6 performance regression: unchanged-workspace re-index must skip all files.

    Validates two guarantees:
    1. Behavioral contract: every file is hash-matched and skipped (files_changed == 0,
       skipped_files == total files from the full build). This is the mechanism that
       produces the speedup -- no re-parsing, no re-embedding.
    2. Timing contract: the unchanged run's ``duration_ms`` is strictly less than the
       full-build run. We use a generous upper-bound ratio (80 %) so the assertion
       survives slow CI runners while still catching regressions where unchanged files
       are accidentally re-indexed.
    """
    runner = CliRunner()

    def _make_module(name: str) -> str:
        return (
            f'"""Module {name} for performance regression fixture."""\n'
            "\n"
            f"class {name.capitalize()}Service:\n"
            f'    """Service class for {name}."""\n'
            "\n"
            "    def __init__(self, value: int) -> None:\n"
            "        self.value = value\n"
            "\n"
            "    def process(self) -> int:\n"
            '        """Return processed value."""\n'
            "        return self.value * 2\n"
            "\n"
            "\n"
            f"def compute_{name}(x: int, y: int) -> int:\n"
            f'    """Compute result for {name}."""\n'
            "    return x + y\n"
        )

    # Build a multi-file repo large enough to make the skip contrast visible.
    module_names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    files = {f"{name}.py": _make_module(name) for name in module_names}

    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(files)
        cache_dir = tempfile.mkdtemp(prefix="gloggur-perf-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        # --- Full build ---
        full_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert full_result.exit_code == 0, full_result.output
        full_payload = _parse_json_output(full_result.output)

        full_scanned: int = int(full_payload["files_scanned"])
        full_changed: int = int(full_payload["files_changed"])
        full_duration_ms: int = int(full_payload["duration_ms"])

        # All files should have been processed in the first run.
        assert full_scanned == len(module_names), (
            f"expected {len(module_names)} files scanned, got {full_scanned}"
        )
        assert full_changed == len(module_names), (
            f"expected all {len(module_names)} files changed on first run, got {full_changed}"
        )

        # --- Unchanged re-index ---
        unchanged_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert unchanged_result.exit_code == 0, unchanged_result.output
        unchanged_payload = _parse_json_output(unchanged_result.output)

        unchanged_scanned: int = int(unchanged_payload["files_scanned"])
        unchanged_changed: int = int(unchanged_payload["files_changed"])
        unchanged_skipped: int = int(unchanged_payload["skipped_files"])
        unchanged_duration_ms: int = int(unchanged_payload["duration_ms"])

        # Behavioral contract: every file must be hash-matched and skipped.
        assert unchanged_scanned == full_scanned, (
            f"unchanged run scanned {unchanged_scanned} files, expected {full_scanned}"
        )
        assert unchanged_changed == 0, (
            f"unchanged run should have files_changed=0, got {unchanged_changed}"
        )
        assert unchanged_skipped == full_scanned, (
            f"unchanged run should skip all {full_scanned} files, got {unchanged_skipped}"
        )

        # Timing contract: unchanged run must be faster than the full build.
        # The 80 % ceiling is generous enough to survive slow CI but catches
        # regressions where unchanged files are accidentally re-processed.
        if full_duration_ms > 0:
            ratio = unchanged_duration_ms / full_duration_ms
            assert ratio < 0.80, (
                f"unchanged re-index took {unchanged_duration_ms} ms "
                f"({ratio:.0%} of full-build {full_duration_ms} ms); "
                "expected < 80 % -- files may be re-indexed instead of skipped"
            )
