from __future__ import annotations

import errno
import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

import gloggur.indexer.cache as cache_module
import gloggur.storage.metadata_store as metadata_store_module
import gloggur.storage.vector_store as vector_store_module
from gloggur.cli import main as cli_main
from gloggur.cli.main import (
    CLI_FAILURE_REMEDIATION,
    INSPECT_PAYLOAD_SCHEMA_BUMP_POLICY,
    _build_inspect_failure_contract,
    _build_inspect_warning_summary,
    _build_resume_contract,
    _compute_retrieval_confidence,
    _inspect_path_class,
    _load_cached_inspect_reports,
    _metadata_reindex_reason,
    _next_retry_top_k,
    _persist_last_success_resume_state,
    _profile_reindex_reason,
    _should_include_inspect_path,
    _stable_fingerprint,
    _tool_version_reindex_reason,
)
from gloggur.indexer.cache import CacheConfig, CacheManager, CacheRecoveryError
from gloggur.io_failures import StorageIOError
from gloggur.models import IndexMetadata, Symbol
from gloggur.search import attach_legacy_search_contract


@pytest.fixture(autouse=True)
def _disable_faiss_for_cli_unit_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep CLI unit tests deterministic and offline by forcing non-FAISS mode."""
    monkeypatch.setattr(
        vector_store_module.VectorStore,
        "_check_faiss",
        staticmethod(lambda: False),
    )


def test_profile_reindex_reason_no_metadata_and_no_profile() -> None:
    """No index metadata/profile should not force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=False,
        cached_profile=None,
        expected_profile="local:model-a",
    )
    assert reason is None


def test_profile_reindex_reason_unknown_profile_with_metadata() -> None:
    """Index metadata without cached profile should force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile=None,
        expected_profile="local:model-a",
    )
    assert reason == "cached embedding profile is unknown"


def test_profile_reindex_reason_profile_changed() -> None:
    """Mismatched cached/expected profile should force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:model-a",
        expected_profile="local:model-b",
    )
    assert reason == "embedding profile changed (cached=local:model-a, current=local:model-b)"


def test_profile_reindex_reason_profile_matches() -> None:
    """Matching profile should not force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:model-a",
        expected_profile="local:model-a",
    )
    assert reason is None


def test_profile_reindex_reason_hf_snapshot_alias_matches_short_model() -> None:
    """Local/test HuggingFace snapshot aliases should not trigger false profile drift."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile=(
            "local:/Users/example/.cache/huggingface/hub/"
            "models--microsoft--codebert-base/snapshots/abc123"
        ),
        expected_profile="local:microsoft/codebert-base",
    )
    assert reason is None


def test_profile_reindex_reason_non_hf_local_path_stays_strict() -> None:
    """Non-HuggingFace local model paths should continue to require exact profile matches."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:/tmp/custom-model-path",
        expected_profile="local:microsoft/codebert-base",
    )
    assert reason is not None
    assert "embedding profile changed" in reason


def test_metadata_reindex_reason_missing_metadata() -> None:
    """Missing index metadata should report an explicit rebuild reason."""
    reason = _metadata_reindex_reason(metadata_present=False)
    assert reason is not None
    assert "index metadata missing" in reason


def test_metadata_reindex_reason_present_metadata() -> None:
    """Existing metadata should not add metadata-specific reindex reason."""
    reason = _metadata_reindex_reason(metadata_present=True)
    assert reason is None


def test_tool_version_reindex_reason_is_legacy_safe() -> None:
    """Missing legacy tool-version marker should not force a reindex."""
    reason = _tool_version_reindex_reason(
        last_success_tool_version=None,
        current_tool_version="0.2.0",
    )
    assert reason is None


def test_tool_version_reindex_reason_detects_version_drift() -> None:
    """Mismatched tool-version markers should force deterministic rebuild signaling."""
    reason = _tool_version_reindex_reason(
        last_success_tool_version="0.1.0",
        current_tool_version="0.2.0",
    )
    assert reason == "tool version changed (cached=0.1.0, current=0.2.0)"


def test_compute_retrieval_confidence_empty_results_is_zero() -> None:
    """Empty result sets should report explicit zero confidence."""
    assert _compute_retrieval_confidence([]) == 0.0


def test_compute_retrieval_confidence_weights_top_and_top3_average() -> None:
    """Confidence should use deterministic weighted top-score + top3 average formula."""
    confidence = _compute_retrieval_confidence(
        [
            {"similarity_score": 0.2},
            {"similarity_score": 0.6},
            {"similarity_score": 0.8},
            {"similarity_score": 0.4},
        ]
    )
    # top=0.8, top3avg=(0.8+0.6+0.4)/3=0.6 -> 0.7*0.8 + 0.3*0.6 = 0.74
    assert confidence == pytest.approx(0.74)


def test_compute_retrieval_confidence_rejects_non_numeric_scores() -> None:
    """Malformed similarity scores should fail loud to avoid silent confidence drift."""
    with pytest.raises(ValueError, match="non-numeric similarity_score"):
        _compute_retrieval_confidence([{"similarity_score": "not-a-float"}])


def test_next_retry_top_k_expands_and_caps_deterministically() -> None:
    """Retry strategy should expand top-k and clamp at configured max."""
    assert _next_retry_top_k(4) == 8
    assert _next_retry_top_k(33, max_top_k=64) == 64
    assert _next_retry_top_k(64, max_top_k=64) == 64


def test_next_retry_top_k_rejects_non_positive_inputs() -> None:
    """Invalid retry strategy bounds should fail loudly."""
    with pytest.raises(ValueError, match="current_top_k must be >= 1"):
        _next_retry_top_k(0)
    with pytest.raises(ValueError, match="max_top_k must be >= 1"):
        _next_retry_top_k(1, max_top_k=0)


def test_stable_fingerprint_is_order_independent() -> None:
    """Fingerprint helper should be deterministic regardless of dict key ordering."""
    payload_a = {
        "b": 2,
        "a": {
            "z": 1,
            "y": ["x", "w"],
        },
    }
    payload_b = {
        "a": {
            "y": ["x", "w"],
            "z": 1,
        },
        "b": 2,
    }
    assert _stable_fingerprint(payload_a) == _stable_fingerprint(payload_b)


def test_inspect_path_class_and_default_filter_scope() -> None:
    """Inspect defaults should include src/other and exclude tests/scripts noise."""
    src_path = "/tmp/repo/src/module.py"
    test_path = "/tmp/repo/tests/test_module.py"
    script_path = "/tmp/repo/scripts/tool.py"
    other_path = "/tmp/repo/examples/demo.py"

    assert _inspect_path_class(src_path) == "src"
    assert _inspect_path_class(test_path) == "tests"
    assert _inspect_path_class(script_path) == "scripts"
    assert _inspect_path_class(other_path) == "other"

    assert _should_include_inspect_path(
        src_path,
        include_tests=False,
        include_scripts=False,
    )
    assert _should_include_inspect_path(
        other_path,
        include_tests=False,
        include_scripts=False,
    )
    assert not _should_include_inspect_path(
        test_path,
        include_tests=False,
        include_scripts=False,
    )
    assert not _should_include_inspect_path(
        script_path,
        include_tests=False,
        include_scripts=False,
    )


def test_inspect_warning_summary_groups_by_type_path_class_and_top_files() -> None:
    """Inspect warning summary should be deterministic and machine-readable."""
    warning_reports = [
        {
            "symbol_id": "/tmp/repo/src/a.py:1:a",
            "warnings": [
                "Missing docstring",
                "Low semantic similarity (score=0.100, threshold=0.200)",
            ],
        },
        {
            "symbol_id": "/tmp/repo/tests/test_a.py:1:test_a",
            "warnings": ["Missing docstring"],
        },
        {
            "symbol_id": "/tmp/repo/scripts/tool.py:1:tool",
            "warnings": ["Missing docstring"],
        },
    ]
    summary = _build_inspect_warning_summary(
        warning_reports,
        symbol_file_paths={
            "/tmp/repo/src/a.py:1:a": "/tmp/repo/src/a.py",
            "/tmp/repo/tests/test_a.py:1:test_a": "/tmp/repo/tests/test_a.py",
            "/tmp/repo/scripts/tool.py:1:tool": "/tmp/repo/scripts/tool.py",
        },
    )

    assert summary["total_warnings"] == 4
    assert summary["by_warning_type"] == {
        "Low semantic similarity": 1,
        "Missing docstring": 3,
    }
    assert summary["by_path_class"] == {
        "src": 2,
        "tests": 1,
        "scripts": 1,
        "other": 0,
    }
    assert summary["reports_by_path_class"] == {
        "src": 1,
        "tests": 1,
        "scripts": 1,
        "other": 0,
    }
    top_files = summary["top_files"]
    assert isinstance(top_files, list)
    assert len(top_files) == 3
    assert top_files[0]["file"] == "/tmp/repo/src/a.py"
    assert top_files[0]["warnings"] == 2
    assert top_files[0]["path_class"] == "src"


def test_inspect_failure_contract_is_machine_readable_and_normalized() -> None:
    """Inspect failure contract should expose stable codes/guidance and ignore invalid counters."""
    contract = _build_inspect_failure_contract(
        {
            "decode_error": 2,
            "parse_error": 1,
            "bogus_zero": 0,
            "bogus_negative": -1,
        }
    )
    assert contract["failure_codes"] == ["decode_error", "parse_error"]
    guidance = contract["failure_guidance"]
    assert isinstance(guidance, dict)
    assert "decode_error" in guidance
    assert isinstance(guidance["decode_error"], list)
    assert guidance["decode_error"]
    assert "parse_error" in guidance
    assert isinstance(guidance["parse_error"], list)
    assert guidance["parse_error"]


def test_attach_primary_error_from_failure_contract_uses_primary_failure_code() -> None:
    """Aggregate failure payloads should expose a stable top-level error block."""
    payload: dict[str, object] = {
        "failure_codes": ["decode_error", "parse_error"],
        "failure_guidance": {
            "decode_error": ["normalize file encoding"],
            "parse_error": ["fix file syntax"],
        },
    }

    cli_main._attach_primary_error_from_failure_contract(
        payload,
        error_type="inspect_failure",
        detail="Inspect completed with file-level failures.",
        probable_cause="One or more files could not be inspected.",
        default_remediation="default remediation",
    )

    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "inspect_failure"
    assert error["code"] == "decode_error"
    assert error["detail"] == "Inspect completed with file-level failures."
    assert error["probable_cause"] == "One or more files could not be inspected."
    assert error["remediation"] == ["normalize file encoding"]


def test_inspect_payload_schema_policy_declares_breaking_change_rules() -> None:
    """Inspect schema policy should define explicit bump/no-bump conditions."""
    assert INSPECT_PAYLOAD_SCHEMA_BUMP_POLICY["policy_version"] == "1"
    bump_required = INSPECT_PAYLOAD_SCHEMA_BUMP_POLICY["bump_required_for"]
    assert isinstance(bump_required, list)
    assert "remove_or_rename_existing_field" in bump_required
    assert "change_existing_field_type" in bump_required
    no_bump = INSPECT_PAYLOAD_SCHEMA_BUMP_POLICY["bump_not_required_for"]
    assert isinstance(no_bump, list)
    assert "add_new_optional_field" in no_bump


def test_load_cached_inspect_reports_rehydrates_cached_reports(tmp_path: Path) -> None:
    """Inspect should reuse cached warning and score metadata for unchanged files."""
    cache_dir = tmp_path / "cache"
    cache = CacheManager(CacheConfig(str(cache_dir)))
    symbol = Symbol(
        id=str(tmp_path / "src" / "main.py") + ":1:sample",
        name="sample",
        kind="function",
        file_path=str(tmp_path / "src" / "main.py"),
        start_line=1,
        end_line=2,
        signature="def sample():",
        body_hash="abc",
    )
    cache.set_audit_report(
        symbol.id,
        warnings=[],
        semantic_score=0.91,
        score_metadata={"threshold_applied": 0.2, "scored": True},
    )
    cache.upsert_symbols([symbol])

    reports = _load_cached_inspect_reports(
        cache,
        symbol.file_path,
        symbol_ids=(),
    )

    assert len(reports) == 1
    assert reports[0].symbol_id == symbol.id
    assert reports[0].warnings == []
    assert reports[0].semantic_score == pytest.approx(0.91)
    assert reports[0].score_metadata == {
        "threshold_applied": 0.2,
        "scored": True,
        "cached_report_reuse": True,
    }


def test_cli_failure_catalog_includes_watch_preflight_codes() -> None:
    """Watch preflight failure codes should stay documented in the CLI failure catalog."""
    for code in (
        "watch_mode_conflict",
        "watch_path_missing",
        "watch_mode_invalid",
        "local_fallback_env_unsupported",
    ):
        assert code in CLI_FAILURE_REMEDIATION
        guidance = CLI_FAILURE_REMEDIATION[code]
        assert isinstance(guidance, list)
        assert guidance


def test_cli_failure_catalog_includes_artifact_publish_codes() -> None:
    """Artifact publish failure codes should stay documented in the CLI failure catalog."""
    for code in (
        "artifact_source_missing",
        "artifact_source_not_directory",
        "artifact_source_uninitialized",
        "artifact_destination_unsupported",
        "artifact_destination_exists",
        "artifact_destination_inside_source",
    ):
        assert code in CLI_FAILURE_REMEDIATION
        guidance = CLI_FAILURE_REMEDIATION[code]
        assert isinstance(guidance, list)
        assert guidance


def test_cli_failure_catalog_includes_artifact_restore_codes() -> None:
    """Artifact restore/validate failure codes should stay documented in the CLI failure catalog."""
    for code in (
        "artifact_path_missing",
        "artifact_path_not_file",
        "artifact_archive_invalid",
        "artifact_manifest_missing",
        "artifact_manifest_invalid",
        "artifact_manifest_schema_unsupported",
        "artifact_manifest_file_mismatch",
        "artifact_manifest_totals_mismatch",
        "artifact_restore_destination_exists",
        "artifact_restore_destination_not_directory",
    ):
        assert code in CLI_FAILURE_REMEDIATION
        guidance = CLI_FAILURE_REMEDIATION[code]
        assert isinstance(guidance, list)
        assert guidance


def test_cli_failure_catalog_includes_artifact_uploader_codes() -> None:
    """Artifact uploader failure codes should stay documented in the CLI failure catalog."""
    for code in (
        "artifact_uploader_command_invalid",
        "artifact_uploader_failed",
        "artifact_uploader_timeout",
        "artifact_http_upload_failed",
        "artifact_http_upload_timeout",
    ):
        assert code in CLI_FAILURE_REMEDIATION
        guidance = CLI_FAILURE_REMEDIATION[code]
        assert isinstance(guidance, list)
        assert guidance


def test_coverage_ingest_invalid_json_returns_coverage_file_invalid(tmp_path: Path) -> None:
    """Malformed coverage JSON should fail closed with stable coverage_file_invalid code."""
    runner = CliRunner()
    coverage_file = tmp_path / "broken-coverage.json"
    coverage_file.write_text('{"test_func": ', encoding="utf8")
    env = {"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")}

    result = runner.invoke(
        cli_main.cli,
        ["coverage", "ingest", str(coverage_file), "--json"],
        env=env,
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "coverage_file_invalid"


def test_coverage_import_python_missing_line_data_returns_coverage_sqlite_invalid(
    tmp_path: Path,
) -> None:
    """coverage import-python should fail closed when required sqlite tables are missing."""
    runner = CliRunner()
    coverage_db = tmp_path / "coverage.db"
    output_file = tmp_path / "gloggur-coverage.json"
    env = {"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")}

    conn = sqlite3.connect(coverage_db)
    try:
        conn.execute("CREATE TABLE context (id INTEGER PRIMARY KEY, context TEXT)")
        conn.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
        conn.commit()
    finally:
        conn.close()

    result = runner.invoke(
        cli_main.cli,
        [
            "coverage",
            "import-python",
            str(coverage_db),
            "--output",
            str(output_file),
            "--json",
        ],
        env=env,
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "coverage_sqlite_invalid"


def test_coverage_import_json_adapter_round_trips_context_map(tmp_path: Path) -> None:
    """coverage import should support generic JSON context maps via adapter registry."""
    runner = CliRunner()
    source_file = tmp_path / "contexts.json"
    output_file = tmp_path / "gloggur-coverage.json"
    source_payload = {
        "test_alpha": {"a.py": [1, 2, 3]},
        "test_beta": {"b.py": [10]},
    }
    source_file.write_text(json.dumps(source_payload), encoding="utf8")
    env = {"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")}

    result = runner.invoke(
        cli_main.cli,
        [
            "coverage",
            "import",
            str(source_file),
            "--importer",
            "json",
            "--output",
            str(output_file),
            "--json",
        ],
        env=env,
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["tests_extracted"] == 2
    assert output_file.exists()
    written = json.loads(output_file.read_text(encoding="utf8"))
    assert written == source_payload


def test_adapters_list_reports_discoverable_categories(tmp_path: Path) -> None:
    """adapters list should return machine-readable adapter categories and active defaults."""
    runner = CliRunner()
    env = {"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")}
    result = runner.invoke(
        cli_main.cli,
        ["adapters", "list", "--json"],
        env=env,
    )
    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["active"]["embedding_provider"] == "local"
    available = payload["available"]
    assert "parsers" in available
    assert "coverage_importers" in available
    assert "embedding_providers" in available
    assert "storage_backends" in available
    assert "runtime_hosts" in available


def test_resolve_artifact_destination_rejects_unsupported_scheme() -> None:
    """Artifact destination parser should fail closed for non-file URI schemes."""
    with pytest.raises(cli_main.CLIContractError) as exc_info:
        cli_main._resolve_artifact_destination(
            "https://example.com/upload",
            default_filename="artifact.tar.gz",
        )
    error = exc_info.value
    assert error.error_code == "artifact_destination_unsupported"


def test_resolve_artifact_destination_supports_file_directory_uri(tmp_path: Path) -> None:
    """file:// URI destination should resolve to default filename under directory path."""
    destination_dir = tmp_path / "artifacts"
    destination_dir.mkdir()
    resolved = cli_main._resolve_artifact_destination(
        destination_dir.resolve().as_uri() + "/",
        default_filename="artifact.tar.gz",
    )
    assert resolved == str(destination_dir / "artifact.tar.gz")


def test_resolve_artifact_restore_path_rejects_parent_traversal(tmp_path: Path) -> None:
    """Restore path resolution should reject manifest entries that escape the restore root."""
    with pytest.raises(cli_main.CLIContractError) as exc_info:
        cli_main._resolve_artifact_restore_path(str(tmp_path), "../escape.db")
    error = exc_info.value
    assert error.error_code == "artifact_manifest_invalid"


def test_render_artifact_uploader_command_formats_supported_placeholders() -> None:
    """Uploader command template should expand supported placeholders deterministically."""
    argv = cli_main._render_artifact_uploader_command(
        "tool --src {artifact_path} --dest {destination} --sha {archive_sha256}",
        artifact_path="/tmp/cache.tgz",
        destination="https://example.com/upload",
        artifact_name="cache.tgz",
        archive_sha256="abc123",
        archive_bytes=42,
        manifest_sha256="def456",
    )
    assert argv == [
        "tool",
        "--src",
        "/tmp/cache.tgz",
        "--dest",
        "https://example.com/upload",
        "--sha",
        "abc123",
    ]


def test_render_artifact_uploader_command_rejects_unknown_placeholder() -> None:
    """Uploader command template should fail closed on unsupported placeholders."""
    with pytest.raises(cli_main.CLIContractError) as exc_info:
        cli_main._render_artifact_uploader_command(
            "tool {artifact_path} {unknown}",
            artifact_path="/tmp/cache.tgz",
            destination="https://example.com/upload",
            artifact_name="cache.tgz",
            archive_sha256="abc123",
            archive_bytes=42,
            manifest_sha256="def456",
        )
    error = exc_info.value
    assert error.error_code == "artifact_uploader_command_invalid"


def test_resume_contract_profile_change_is_machine_readable() -> None:
    """Profile drift should produce stable machine-readable resume metadata signals."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-b",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
    )

    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["embedding_profile_changed"]
    assert payload["resume_fingerprint_match"] is False


def test_resume_contract_hf_snapshot_alias_is_resume_compatible() -> None:
    """HuggingFace snapshot-path aliases should not force false reindex decisions."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:microsoft/codebert-base",
        cached_profile=(
            "local:/Users/example/.cache/huggingface/hub/"
            "models--microsoft--codebert-base/snapshots/abc123"
        ),
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
    )

    assert payload["resume_decision"] == "resume_ok"
    assert payload["resume_reason_codes"] == []
    assert payload["resume_fingerprint_match"] is True


def test_resume_contract_missing_metadata_has_machine_reason_code() -> None:
    """Missing index metadata should surface explicit stable reason codes for agents."""
    payload = _build_resume_contract(
        metadata=None,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile=None,
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
    )

    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["missing_index_metadata"]
    remediation = payload["resume_remediation"]
    assert isinstance(remediation, dict)
    assert "missing_index_metadata" in remediation
    assert isinstance(remediation["missing_index_metadata"], list)
    assert remediation["missing_index_metadata"]


def test_resume_contract_interrupted_index_has_machine_reason_and_remediation() -> None:
    """Interrupted runs should emit explicit interruption reason code and deterministic guidance."""
    payload = _build_resume_contract(
        metadata=None,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint="last-success-fingerprint",
        last_success_resume_at="2026-02-26T00:00:00+00:00",
    )

    assert payload["resume_decision"] == "reindex_required"
    codes = payload["resume_reason_codes"]
    assert isinstance(codes, list)
    assert "index_interrupted" in codes
    assert "missing_index_metadata" in codes
    remediation = payload["resume_remediation"]
    assert isinstance(remediation, dict)
    assert "index_interrupted" in remediation
    assert isinstance(remediation["index_interrupted"], list)
    assert remediation["index_interrupted"]


def test_resume_contract_reports_last_success_fingerprint_match_signal() -> None:
    """Resume contract should expose whether last-success fingerprint still matches expected."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    expected = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
    )["expected_resume_fingerprint"]
    assert isinstance(expected, str)

    match_payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=expected,
        last_success_resume_at="2026-02-26T00:00:00+00:00",
    )
    mismatch_payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint="stale-fingerprint",
        last_success_resume_at="2026-02-26T00:00:00+00:00",
    )

    assert match_payload["last_success_resume_fingerprint_match"] is True
    assert mismatch_payload["last_success_resume_fingerprint_match"] is False


def test_resume_contract_detects_tool_version_drift_since_last_success() -> None:
    """Tool-version drift should be detectable without mislabeling profile drift."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    old_fingerprint = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        tool_version="0.1.0",
    )["expected_resume_fingerprint"]
    assert isinstance(old_fingerprint, str)
    payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=old_fingerprint,
        last_success_resume_at="2026-02-26T00:00:00+00:00",
        tool_version="0.2.0",
        last_success_tool_version="0.1.0",
    )
    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["tool_version_changed"]
    assert payload["resume_fingerprint_match"] is False
    assert payload["last_success_tool_version_match"] is False
    assert payload["last_success_resume_fingerprint_match"] is False


def test_resume_contract_tool_version_override_is_explicit_and_resume_ok() -> None:
    """Explicit tool-version drift override should be machine-readable and not silent."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        tool_version="0.2.0",
        last_success_tool_version="0.1.0",
        allow_tool_version_drift=True,
    )

    assert payload["resume_decision"] == "resume_ok"
    assert payload["resume_reason_codes"] == ["tool_version_changed_override"]
    assert payload["tool_version_drift_detected"] is True
    assert payload["allow_tool_version_drift"] is True
    assert payload["tool_version_drift_override_applied"] is True
    remediation = payload["resume_remediation"]
    assert isinstance(remediation, dict)
    assert "tool_version_changed_override" in remediation
    assert isinstance(remediation["tool_version_changed_override"], list)
    assert remediation["tool_version_changed_override"]


def test_resume_contract_tool_version_override_does_not_bypass_missing_metadata() -> None:
    """Tool-version override must not mask true missing-metadata reindex requirements."""
    payload = _build_resume_contract(
        metadata=None,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint="last-success",
        last_success_resume_at="2026-02-27T00:00:00+00:00",
        tool_version="0.2.0",
        last_success_tool_version="0.1.0",
        allow_tool_version_drift=True,
    )

    assert payload["resume_decision"] == "reindex_required"
    assert payload["allow_tool_version_drift"] is True
    assert payload["tool_version_drift_override_applied"] is True
    codes = set(payload["resume_reason_codes"])
    assert "tool_version_changed_override" in codes
    assert "missing_index_metadata" in codes
    assert "index_interrupted" in codes


def test_resume_contract_missing_tool_version_marker_remains_resume_ok() -> None:
    """Legacy caches without tool-version markers should keep resume compatibility logic stable."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    payload = _build_resume_contract(
        metadata=metadata,
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile="local:model-a",
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        tool_version="0.2.0",
        last_success_tool_version=None,
    )

    assert payload["resume_decision"] == "resume_ok"
    assert payload["resume_reason_codes"] == []
    assert payload["resume_fingerprint_match"] is True
    assert payload["last_success_tool_version_match"] is None


def test_persist_last_success_resume_state_no_change_reindex_is_idempotent() -> None:
    """An unchanged re-index must not advance last_success_resume_at or rewrite the fingerprint.

    Regression test for bd-l7d: before the fix, _persist_last_success_resume_state wrote a
    fresh metadata.last_updated timestamp to last_success_resume_at on every successful index
    run — including runs where no indexed content changed.  This caused last_success_resume_at
    to drift even when the resume fingerprint was stable.
    """

    writes: list[tuple[str, str]] = []

    class FakeCache:
        last_reset_reason = None
        _fingerprint: str | None = None
        _at: str | None = None
        _tool_version: str | None = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=5, indexed_files=2)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> str | None:
            return self._fingerprint

        def set_last_success_resume_fingerprint(self, fp: str) -> None:
            writes.append(("fingerprint", fp))
            self._fingerprint = fp

        def get_last_success_resume_at(self) -> str | None:
            return self._at

        def set_last_success_resume_at(self, ts: str) -> None:
            writes.append(("at", ts))
            self._at = ts

        def get_last_success_tool_version(self) -> str | None:
            return self._tool_version

        def set_last_success_tool_version(self, v: str) -> None:
            writes.append(("tool_version", v))
            self._tool_version = v

        def count_symbols(self) -> int:
            return 5

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir="/tmp/cache",
    )
    cache = FakeCache()

    # First call: cold state, fingerprint must be written.
    _persist_last_success_resume_state(config, cache)
    assert len(writes) == 3, f"First index should write all three fields, got: {writes}"
    first_fp = cache._fingerprint
    first_at = cache._at
    assert first_fp is not None
    assert first_at is not None

    writes.clear()

    # Second call: identical state (no content change).  Nothing must be rewritten.
    _persist_last_success_resume_state(config, cache)
    assert (
        writes == []
    ), f"Unchanged re-index must not rewrite any stored state, got writes: {writes}"
    assert cache._fingerprint == first_fp, "expected_resume_fingerprint must be stable"
    assert cache._at == first_at, "last_success_resume_at must not advance on unchanged re-index"


def test_build_status_payload_requires_reindex_on_tool_version_drift() -> None:
    """Status payload should fail closed when last-success tool version drifts."""
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir="/tmp/cache",
    )

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=3, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def list_symbols(self) -> list[object]:
            return [object(), object(), object()]

    payload = cli_main._build_status_payload(config, FakeCache())

    assert payload["needs_reindex"] is True
    assert "tool version changed" in str(payload["reindex_reason"])
    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["tool_version_changed"]
    assert payload["resume_fingerprint_match"] is False


def test_build_status_payload_allows_explicit_tool_version_drift_override() -> None:
    """Status payload should allow explicit tool-version drift override with explicit metadata."""
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir="/tmp/cache",
    )

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=3, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def list_symbols(self) -> list[object]:
            return [object(), object(), object()]

    payload = cli_main._build_status_payload(
        config,
        FakeCache(),
        allow_tool_version_drift=True,
    )

    assert payload["needs_reindex"] is False
    assert payload["reindex_reason"] is None
    assert payload["resume_decision"] == "resume_ok"
    assert payload["resume_reason_codes"] == ["tool_version_changed_override"]
    assert payload["tool_version_drift_detected"] is True
    assert payload["allow_tool_version_drift"] is True
    assert payload["tool_version_drift_override_applied"] is True


def test_resolve_allow_tool_version_drift_combines_cli_flag_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI flag should OR with strict env override values."""
    monkeypatch.setattr(cli_main.GloggurConfig, "_load_dotenv", lambda _path=".env": {})
    monkeypatch.delenv("GLOGGUR_ALLOW_TOOL_VERSION_DRIFT", raising=False)
    assert cli_main._resolve_allow_tool_version_drift(cli_flag_enabled=False) is False
    assert cli_main._resolve_allow_tool_version_drift(cli_flag_enabled=True) is True

    monkeypatch.setenv("GLOGGUR_ALLOW_TOOL_VERSION_DRIFT", "true")
    assert cli_main._resolve_allow_tool_version_drift(cli_flag_enabled=False) is True
    assert cli_main._resolve_allow_tool_version_drift(cli_flag_enabled=True) is True

    monkeypatch.setenv("GLOGGUR_ALLOW_TOOL_VERSION_DRIFT", "false")
    assert cli_main._resolve_allow_tool_version_drift(cli_flag_enabled=False) is False
    assert cli_main._resolve_allow_tool_version_drift(cli_flag_enabled=True) is True


def test_resolve_allow_tool_version_drift_rejects_invalid_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid env override values should fail loudly with stable code."""
    monkeypatch.setattr(cli_main.GloggurConfig, "_load_dotenv", lambda _path=".env": {})
    monkeypatch.setenv("GLOGGUR_ALLOW_TOOL_VERSION_DRIFT", "sometimes")

    with pytest.raises(cli_main.CLIContractError) as error:
        cli_main._resolve_allow_tool_version_drift(cli_flag_enabled=False)

    assert error.value.error_code == "allow_tool_version_drift_env_invalid"


def test_validate_legacy_local_fallback_env_rejects_non_empty_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy fallback env should fail loud with a stable contract code."""
    monkeypatch.setattr(cli_main.GloggurConfig, "_load_dotenv", lambda _path=".env": {})
    monkeypatch.setenv("GLOGGUR_LOCAL_FALLBACK", "1")

    with pytest.raises(cli_main.CLIContractError) as error:
        cli_main._validate_legacy_local_fallback_env()

    assert error.value.error_code == "local_fallback_env_unsupported"


def test_validate_legacy_local_fallback_env_allows_blank_or_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset or blank legacy fallback env should not trigger contract failure."""
    monkeypatch.setattr(cli_main.GloggurConfig, "_load_dotenv", lambda _path=".env": {})
    monkeypatch.delenv("GLOGGUR_LOCAL_FALLBACK", raising=False)
    cli_main._validate_legacy_local_fallback_env()
    monkeypatch.setenv("GLOGGUR_LOCAL_FALLBACK", " ")
    cli_main._validate_legacy_local_fallback_env()


def test_status_json_rejects_legacy_local_fallback_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """status --json should fail closed when legacy fallback env is configured."""
    runner = CliRunner()
    monkeypatch.setattr(cli_main.GloggurConfig, "_load_dotenv", lambda _path=".env": {})
    result = runner.invoke(
        cli_main.cli,
        ["status", "--json"],
        env={"GLOGGUR_LOCAL_FALLBACK": "1"},
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "local_fallback_env_unsupported"
    assert payload["failure_codes"] == ["local_fallback_env_unsupported"]


def test_adapters_list_json_rejects_legacy_local_fallback_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """adapters list --json should use structured contract handling for legacy fallback env."""
    runner = CliRunner()
    monkeypatch.setattr(cli_main.GloggurConfig, "_load_dotenv", lambda _path=".env": {})
    result = runner.invoke(
        cli_main.cli,
        ["adapters", "list", "--json"],
        env={"GLOGGUR_LOCAL_FALLBACK": "1"},
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "local_fallback_env_unsupported"
    assert payload["failure_codes"] == ["local_fallback_env_unsupported"]


def test_status_json_rejects_invalid_tool_version_drift_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """status --json should fail closed on malformed drift override env values."""
    runner = CliRunner()
    monkeypatch.setattr(cli_main.GloggurConfig, "_load_dotenv", lambda _path=".env": {})
    result = runner.invoke(
        cli_main.cli,
        ["status", "--json"],
        env={"GLOGGUR_ALLOW_TOOL_VERSION_DRIFT": "sometimes"},
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "allow_tool_version_drift_env_invalid"
    assert payload["failure_codes"] == ["allow_tool_version_drift_env_invalid"]


def test_status_supports_tilde_expanded_config_path(tmp_path: Path) -> None:
    """status should expand `~` in --config paths before loading files."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    config_path = fake_home / ".gloggur.yaml"
    config_path.write_text(
        f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )

    result = runner.invoke(
        cli_main.cli,
        ["status", "--json", "--config", "~/.gloggur.yaml"],
        env={"HOME": str(fake_home), "GLOGGUR_CACHE_DIR": ""},
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["cache_dir"] == str(cache_dir)


def test_create_runtime_applies_embedding_override_without_reloading_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime override should not trigger an unwrapped second config-file load."""

    base_config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            return None

        def get_index_profile(self) -> None:
            return None

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: base_config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _cache_dir: FakeCache())
    monkeypatch.setattr(cli_main, "VectorStore", lambda _cfg: object())

    def _unexpected_reload(*_args: object, **_kwargs: object) -> cli_main.GloggurConfig:
        raise AssertionError("GloggurConfig.load should not be called from _create_runtime")

    monkeypatch.setattr(cli_main.GloggurConfig, "load", _unexpected_reload)

    config, _cache, _vector_store = cli_main._create_runtime(
        config_path=None,
        embedding_provider="openai",
    )

    assert config.embedding_provider == "openai"


def test_create_cache_manager_wraps_cache_recovery_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecoverable cache corruption should map to structured IO failures."""

    class BrokenCacheManager:
        def __init__(self, _config: object) -> None:
            raise CacheRecoveryError("recovery failed")

    monkeypatch.setattr(cli_main, "CacheManager", BrokenCacheManager)
    with pytest.raises(StorageIOError) as exc_info:
        cli_main._create_cache_manager("/tmp/gloggur-cache")

    error = exc_info.value
    assert error.category == "unknown_io_error"
    assert error.operation == "recover corrupted cache database"
    assert error.path.endswith("index.db")
    assert "recovery failed" in error.detail


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo: ["status", "--json"]),
        ("search", lambda repo: ["search", "needle", "--json"]),
        ("inspect", lambda repo: ["inspect", str(repo), "--json"]),
        ("clear-cache", lambda repo: ["clear-cache", "--json"]),
        ("index", lambda repo: ["index", str(repo), "--json"]),
    ],
)
def test_core_commands_surface_cache_recovery_failure_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path], list[str]],
) -> None:
    """Core CLI commands should fail non-zero on unrecoverable cache recovery."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n" "    return a + b\n",
        encoding="utf8",
    )

    class BrokenCacheManager:
        def __init__(self, _config: object) -> None:
            raise CacheRecoveryError("recovery failed")

    monkeypatch.setattr(cli_main, "CacheManager", BrokenCacheManager)
    result = runner.invoke(cli_main.cli, build_args(repo))

    assert result.exit_code != 0
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "recover corrupted cache database"
    assert str(error["path"]).endswith("index.db")
    assert "recovery failed" in result.output
    assert "Traceback (most recent call last)" not in result.output


def test_status_retries_transient_no_such_table_race_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """status should retry once when a concurrent recovery transiently drops tables mid-read."""
    runner = CliRunner()
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )

    class FlakyCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            raise StorageIOError(
                category="unknown_io_error",
                operation="execute cache database transaction",
                path=str(tmp_path / "cache" / "index.db"),
                probable_cause="An unclassified filesystem or database I/O failure occurred.",
                remediation=["Retry with a known-good writable cache directory."],
                detail="OperationalError: no such table: metadata",
            )

    class HealthyCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            return None

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> None:
            return None

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> None:
            return None

        def list_symbols(self) -> list[object]:
            return []

    cache_instances = [FlakyCache(), HealthyCache()]

    def _next_cache(_cache_dir: str) -> object:
        assert cache_instances
        return cache_instances.pop(0)

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", _next_cache)

    result = runner.invoke(cli_main.cli, ["status", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["cache_dir"] == str(tmp_path / "cache")
    assert payload["needs_reindex"] is True
    assert payload["total_symbols"] == 0
    assert not cache_instances


@pytest.mark.parametrize(
    ("operation", "detail"),
    [
        ("execute cache database transaction", "OperationalError: no such table: metadata"),
        ("configure cache database pragmas", "DatabaseError: file is not a database"),
    ],
)
def test_status_remaps_repeated_transient_no_such_table_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    detail: str,
) -> None:
    """status should map repeated transient race failures to cache recovery operation semantics."""
    runner = CliRunner()
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )

    class FlakyCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            raise StorageIOError(
                category="unknown_io_error",
                operation=operation,
                path=str(tmp_path / "cache" / "index.db"),
                probable_cause="An unclassified filesystem or database I/O failure occurred.",
                remediation=["Retry with a known-good writable cache directory."],
                detail=detail,
            )

    call_count = {"value": 0}

    def _flaky_cache(_cache_dir: str) -> FlakyCache:
        call_count["value"] += 1
        return FlakyCache()

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", _flaky_cache)

    result = runner.invoke(cli_main.cli, ["status", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "recover corrupted cache database"
    assert str(error["detail"]) == detail
    assert call_count["value"] == 2
    assert "Traceback (most recent call last)" not in result.output


def test_status_remaps_transient_pragmas_error_during_cache_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """status should remap repeated transient cache-init pragma failures to recovery semantics."""
    runner = CliRunner()
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )
    call_count = {"value": 0}

    def _raise_pragmas_error(_cache_dir: str) -> object:
        call_count["value"] += 1
        raise StorageIOError(
            category="unknown_io_error",
            operation="configure cache database pragmas",
            path=str(tmp_path / "cache" / "index.db"),
            probable_cause="An unclassified filesystem or database I/O failure occurred.",
            remediation=["Retry with a known-good writable cache directory."],
            detail="DatabaseError: file is not a database",
        )

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", _raise_pragmas_error)

    result = runner.invoke(cli_main.cli, ["status", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "recover corrupted cache database"
    assert str(error["detail"]) == "DatabaseError: file is not a database"
    assert call_count["value"] == 2


@pytest.mark.parametrize("as_json", [False, True])
def test_status_surfaces_unrecoverable_corruption_recovery_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    as_json: bool,
) -> None:
    """status should fail with remediation guidance if corruption recovery cannot proceed."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "index.db"
    db_path.write_bytes(b"not-a-sqlite-db")

    def _always_fail_replace(_src: str, _dst: str) -> None:
        raise OSError("replace denied")

    def _always_fail_remove(_path: str) -> None:
        raise OSError("remove denied")

    monkeypatch.setattr(cache_module.os, "replace", _always_fail_replace)
    monkeypatch.setattr(cache_module.os, "remove", _always_fail_remove)

    args = ["status", "--json"] if as_json else ["status"]
    result = runner.invoke(cli_main.cli, args, env={"GLOGGUR_CACHE_DIR": str(cache_dir)})
    assert result.exit_code != 0
    assert "Cache corruption detected but recovery failed" in result.output
    assert "Fix permissions and remove corrupted cache files manually." in result.output
    assert "Traceback (most recent call last)" not in result.output


def _parse_json_output(output: str) -> dict[str, object]:
    """Parse JSON payload from click output that may include stderr text."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    if (
        isinstance(payload, dict)
        and payload.get("ok") is False
        and isinstance(payload.get("compatibility"), dict)
    ):
        compatibility = dict(payload["compatibility"])
        compatibility["_envelope"] = payload
        return compatibility
    if isinstance(payload, dict):
        return attach_legacy_search_contract(payload)
    return payload


def test_cli_main_unknown_command_json_emits_dispatch_envelope(capsys) -> None:
    exit_code = cli_main.main(["definitely-unknown-command", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert exit_code != 0
    assert payload["ok"] is False
    assert payload["error_code"] == "cli_usage_error"
    assert payload["stage"] == "dispatch"
    assert captured.err == ""


def test_status_json_error_emits_single_structured_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(
        cli_main.cli,
        ["status", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.exit_code == 1
    raw = result.output.strip()
    payload = json.loads(raw)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert raw.count("\n") == 0


@pytest.mark.parametrize(
    ("exception_factory", "category", "detail_substring"),
    [
        (
            lambda: PermissionError(errno.EACCES, "permission denied"),
            "permission_denied",
            "PermissionError",
        ),
        (
            lambda: OSError(errno.EROFS, "read-only filesystem"),
            "read_only_filesystem",
            "read-only filesystem",
        ),
        (
            lambda: sqlite3.OperationalError("database or disk is full"),
            "disk_full_or_quota",
            "database or disk is full",
        ),
        (
            lambda: sqlite3.OperationalError("unable to open database file"),
            "path_not_writable",
            "unable to open database file",
        ),
        (
            lambda: OSError(errno.EIO, "i/o error"),
            "unknown_io_error",
            "i/o error",
        ),
    ],
)
def test_status_json_reports_structured_io_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_factory: Callable[[], Exception],
    category: str,
    detail_substring: str,
) -> None:
    """status --json should emit stable machine-readable payloads for IO failure categories."""
    runner = CliRunner()

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise exception_factory()

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(
        cli_main.cli,
        ["status", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == category
    assert "open cache database connection" in str(error["operation"])
    assert str(error["path"]).endswith("index.db")
    assert detail_substring in str(error["detail"])
    remediation = error["remediation"]
    assert isinstance(remediation, list) and remediation


def test_status_plain_output_includes_actionable_io_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """status (non-json) should include stable actionable guidance on stderr."""
    runner = CliRunner()

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(
        cli_main.cli,
        ["status"],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.exit_code == 1
    assert "IO failure [permission_denied]" in result.output
    assert "Probable cause:" in result.output
    assert "Remediation:" in result.output
    assert "PermissionError" in result.output


def test_index_json_reports_vector_store_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """index --json should surface vector id-map write errors with structured payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n" "    return a + b\n",
        encoding="utf8",
    )

    def _raise_disk_full(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(vector_store_module.json, "dump", _raise_disk_full)
    result = runner.invoke(
        cli_main.cli,
        ["index", str(repo), "--json"],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        },
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["category"] == "disk_full_or_quota"
    assert "write vector id map" in str(error["operation"])
    assert str(error["path"]).endswith("vectors.json")
    assert "no space left on device" in str(error["detail"])


@pytest.mark.parametrize("provider", ["local", "openai", "gemini"])
def test_index_plain_progress_reports_done_over_total_for_all_embedding_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
) -> None:
    """index (non-json) should show done/total scan progress regardless of embedding provider."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n" "    return a + b\n",
        encoding="utf8",
    )
    seen_providers: list[str] = []

    class FakeEmbedding:
        provider = "fake"

        def embed_text(self, _text: str) -> list[float]:
            return [0.1, 0.2]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

        def get_dimension(self) -> int:
            return 2

    def _fake_create_embedding(
        config: cli_main.GloggurConfig,
        *,
        require_provider: bool = False,
    ) -> FakeEmbedding:
        _ = require_provider
        seen_providers.append(config.embedding_provider)
        return FakeEmbedding()

    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        _fake_create_embedding,
    )
    result = runner.invoke(
        cli_main.cli,
        ["index", str(repo), "--embedding-provider", provider],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / f"cache-{provider}")},
    )

    assert result.exit_code == 0
    assert seen_providers == [provider]
    assert "Scanning: 1/1 files" in result.output


def test_clear_cache_json_reports_vector_artifact_delete_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """clear-cache --json should surface vector artifact delete failures."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_map = cache_dir / "vectors.json"
    vectors_map.write_text("{}", encoding="utf8")
    original_remove = vector_store_module.os.remove

    def _remove_with_permission_denied(path: str | os.PathLike[str]) -> None:
        normalized_path = os.fspath(path)
        if normalized_path.endswith("vectors.json"):
            raise PermissionError(errno.EACCES, "permission denied")
        original_remove(normalized_path)

    monkeypatch.setattr(vector_store_module.os, "remove", _remove_with_permission_denied)
    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "permission_denied"
    assert error["operation"] == "delete vector artifact"
    assert str(error["path"]).endswith("vectors.json")
    assert "permission denied" in str(error["detail"]).lower()


def test_clear_cache_json_ignores_invalid_faiss_index_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear vector files without loading existing index artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_file = cache_dir / "vectors.index"
    index_file.write_text("invalid-faiss-bytes", encoding="utf8")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not index_file.exists()


def test_clear_cache_json_ignores_invalid_vector_id_map_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear malformed vectors.json artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_map = cache_dir / "vectors.json"
    vectors_map.write_text("{this-is-not-json", encoding="utf8")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not vectors_map.exists()


def test_clear_cache_json_ignores_invalid_fallback_vector_matrix_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear malformed fallback matrix artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback_matrix = cache_dir / "vectors.npy"
    fallback_matrix.write_bytes(b"not-a-valid-npy")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not fallback_matrix.exists()


def test_clear_cache_json_profile_filter_miss_keeps_cache_intact(
    tmp_path: Path,
) -> None:
    """clear-cache with non-matching profile filter should no-op."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache = CacheManager(CacheConfig(str(cache_dir)))
    cache.set_index_profile("local:microsoft/codebert-base")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json", "--profile-filter", "gemini"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is False
    assert payload["reason"] == "profile_filter_miss"
    assert payload["cached_index_profile"] == "local:microsoft/codebert-base"
    reloaded = CacheManager(CacheConfig(str(cache_dir)))
    assert reloaded.get_index_profile() == "local:microsoft/codebert-base"


def test_clear_cache_json_profile_filter_match_clears_cache(
    tmp_path: Path,
) -> None:
    """clear-cache with matching profile filter should clear cache artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache = CacheManager(CacheConfig(str(cache_dir)))
    cache.set_index_profile("local:microsoft/codebert-base")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json", "--profile-filter", "codebert"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert payload["cached_index_profile"] == "local:microsoft/codebert-base"
    reloaded = CacheManager(CacheConfig(str(cache_dir)))
    assert reloaded.get_index_profile() is None


def test_index_json_reports_missing_embedding_provider_configuration(
    tmp_path: Path,
) -> None:
    """index --json should report structured provider error when embedding provider is unset."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / ".gloggur.yaml"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n" "    return a + b\n",
        encoding="utf8",
    )
    config_path.write_text(
        'embedding_provider: ""\n' f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )

    result = runner.invoke(
        cli_main.cli,
        ["index", str(repo), "--json", "--config", str(config_path)],
        env={"GLOGGUR_EMBEDDING_PROVIDER": ""},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "embedding_provider_error"
    assert error["code"] == "embedding_provider_error"
    assert error["provider"] == "unknown"
    assert error["operation"] == "initialize embedding provider"
    assert "embedding provider is not configured" in str(error["detail"])
    assert payload["failure_codes"] == ["embedding_provider_error"]
    assert payload["failure_guidance"] == {
        "embedding_provider_error": error["remediation"],
    }
    assert "Traceback (most recent call last)" not in result.output


def test_search_json_reports_missing_embedding_provider_configuration(
    tmp_path: Path,
) -> None:
    """search --json should remain available via deterministic non-semantic backends when provider is unset."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / ".gloggur.yaml"
    config_path.write_text(
        'embedding_provider: ""\n' f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )
    cache = CacheManager(CacheConfig(str(cache_dir)))
    cache.set_index_metadata(IndexMetadata(version="1", total_symbols=0, indexed_files=0))
    cache.set_index_profile(":unknown")

    result = runner.invoke(
        cli_main.cli,
        ["search", "needle", "--json", "--config", str(config_path)],
        env={"GLOGGUR_EMBEDDING_PROVIDER": "", "GLOGGUR_CACHE_DIR": ""},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["schema_version"] == 2
    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert "needs_reindex" in summary
    assert isinstance(payload.get("hits"), list)
    assert "Traceback (most recent call last)" not in result.output


def test_search_json_requires_reindex_on_tool_version_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should fail closed when cached tool-version marker drifts."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["results"] == []
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["needs_reindex"] is True
    assert "tool version changed" in str(metadata["reindex_reason"])
    assert metadata["resume_decision"] == "reindex_required"
    assert metadata["resume_reason_codes"] == ["tool_version_changed"]


def test_search_json_allows_explicit_tool_version_drift_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should proceed only when explicit tool-version drift override is set."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = filters
            _ = top_k
            _ = context_radius
            return {
                "query": query,
                "results": [
                    {
                        "symbol": "needle",
                        "kind": "function",
                        "file": "sample.py",
                        "line": 1,
                        "signature": "def needle() -> None",
                        "similarity_score": 0.9,
                    }
                ],
                "metadata": {"total_results": 1, "search_time_ms": 1},
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(
        cli_main.cli,
        ["search", "needle", "--json", "--allow-tool-version-drift"],
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["results"]
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["resume_decision"] == "resume_ok"
    assert metadata["needs_reindex"] is False
    assert metadata["resume_reason_codes"] == ["tool_version_changed_override"]
    assert metadata["tool_version_drift_detected"] is True
    assert metadata["allow_tool_version_drift"] is True
    assert metadata["tool_version_drift_override_applied"] is True


def test_search_json_retries_once_for_low_confidence_and_emits_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Low-confidence retrieval should remain single-pass under router v2 with explicit low-confidence metadata."""
    runner = CliRunner()
    search_top_k_calls: list[int] = []

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query
            _ = filters
            search_top_k_calls.append(top_k)
            _ = context_radius
            score = 0.2 if len(search_top_k_calls) == 1 else 0.95
            return {
                "query": "needle",
                "results": [
                    {
                        "symbol": "needle",
                        "kind": "function",
                        "file": "sample.py",
                        "line": 1,
                        "signature": "def needle() -> None",
                        "similarity_score": score,
                    }
                ],
                "metadata": {"total_results": 1, "search_time_ms": 1},
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(
        cli_main.cli,
        [
            "search",
            "needle",
            "--json",
            "--mode",
            "semantic",
            "--confidence-threshold",
            "0.90",
            "--max-requery-attempts",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert search_top_k_calls == [10]
    payload = _parse_json_output(result.output)
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["initial_confidence"] == pytest.approx(0.2)
    assert metadata["final_confidence"] == pytest.approx(0.2)
    assert metadata["retry_performed"] is False
    assert metadata["retry_attempts"] == 0
    assert metadata["retry_strategy"] == "router"
    assert metadata["initial_top_k"] == 10
    assert metadata["final_top_k"] == 10
    assert metadata["low_confidence"] is True


def test_search_json_requery_is_bounded_by_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Router v2 should remain deterministic and single-pass even when retry knobs are set."""
    runner = CliRunner()
    search_top_k_calls: list[int] = []

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query
            _ = filters
            search_top_k_calls.append(top_k)
            _ = context_radius
            return {
                "query": "needle",
                "results": [
                    {
                        "symbol": "needle",
                        "kind": "function",
                        "file": "sample.py",
                        "line": 1,
                        "signature": "def needle() -> None",
                        "similarity_score": 0.1,
                    }
                ],
                "metadata": {"total_results": 1, "search_time_ms": 1},
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(
        cli_main.cli,
        [
            "search",
            "needle",
            "--json",
            "--mode",
            "semantic",
            "--confidence-threshold",
            "0.90",
            "--max-requery-attempts",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert search_top_k_calls == [10]
    payload = _parse_json_output(result.output)
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["retry_performed"] is False
    assert metadata["retry_attempts"] == 0
    assert metadata["max_requery_attempts"] == 1
    assert metadata["low_confidence"] is True
    assert metadata["final_confidence"] == pytest.approx(0.1)


def test_search_json_disable_bounded_requery_skips_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit retry disable should preserve low-confidence marker without second query."""
    runner = CliRunner()
    search_top_k_calls: list[int] = []

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query
            _ = filters
            search_top_k_calls.append(top_k)
            _ = context_radius
            return {
                "query": "needle",
                "results": [
                    {
                        "symbol": "needle",
                        "kind": "function",
                        "file": "sample.py",
                        "line": 1,
                        "signature": "def needle() -> None",
                        "similarity_score": 0.2,
                    }
                ],
                "metadata": {"total_results": 1, "search_time_ms": 1},
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(
        cli_main.cli,
        [
            "search",
            "needle",
            "--json",
            "--mode",
            "semantic",
            "--confidence-threshold",
            "0.90",
            "--max-requery-attempts",
            "1",
            "--disable-bounded-requery",
        ],
    )

    assert result.exit_code == 0, result.output
    assert search_top_k_calls == [10]
    payload = _parse_json_output(result.output)
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["retry_enabled"] is False
    assert metadata["retry_performed"] is False
    assert metadata["retry_attempts"] == 0
    assert metadata["low_confidence"] is True


def test_search_json_rejects_invalid_confidence_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Out-of-range confidence threshold should fail loud with deterministic CLI contract code."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", lambda *_args, **_kwargs: object())

    result = runner.invoke(
        cli_main.cli,
        ["search", "needle", "--json", "--confidence-threshold", "1.5"],
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "search_confidence_threshold_invalid"


def test_search_json_rejects_stream_with_grounding_contract_options() -> None:
    """Stream mode should fail closed when evidence/validation payloads are requested."""
    runner = CliRunner()
    result = runner.invoke(
        cli_main.cli,
        ["search", "needle", "--json", "--stream", "--validate-grounding"],
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "search_contract_v1_removed"


def test_search_json_opt_in_evidence_trace_and_validation_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy evidence-trace options should fail closed under v2 contract."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query
            _ = filters
            _ = top_k
            _ = context_radius
            return {
                "query": "needle",
                "results": [
                    {
                        "symbol_id": "sample.py:1:needle",
                        "symbol": "needle",
                        "kind": "function",
                        "file": "sample.py",
                        "line": 1,
                        "line_end": 1,
                        "signature": "def needle() -> None",
                        "similarity_score": 0.95,
                    }
                ],
                "metadata": {"total_results": 1, "search_time_ms": 1},
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(
        cli_main.cli,
        [
            "search",
            "needle",
            "--json",
            "--with-evidence-trace",
            "--validate-grounding",
            "--evidence-min-confidence",
            "0.8",
            "--evidence-min-items",
            "1",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "search_contract_v1_removed"
    assert payload["failure_codes"] == ["search_contract_v1_removed"]


def test_search_json_fail_on_ungrounded_exits_nonzero_with_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy fail-on-ungrounded option should fail closed under v2 contract."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query
            _ = filters
            _ = top_k
            _ = context_radius
            return {
                "query": "needle",
                "results": [],
                "metadata": {"total_results": 0, "search_time_ms": 1},
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(
        cli_main.cli,
        [
            "search",
            "needle",
            "--json",
            "--validate-grounding",
            "--fail-on-ungrounded",
        ],
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "search_contract_v1_removed"
    assert payload["failure_codes"] == ["search_contract_v1_removed"]


def test_search_json_default_is_backward_compatible_without_trace_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default search path should remain opt-in for trace/validation payload fields."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query
            _ = filters
            _ = top_k
            _ = context_radius
            return {
                "query": "needle",
                "results": [
                    {
                        "symbol_id": "sample.py:1:needle",
                        "symbol": "needle",
                        "kind": "function",
                        "file": "sample.py",
                        "line": 1,
                        "signature": "def needle() -> None",
                        "similarity_score": 0.7,
                    }
                ],
                "metadata": {"total_results": 1, "search_time_ms": 1},
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert "evidence_trace" not in payload
    assert "validation" not in payload
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["grounding_validation_enabled"] is False
    assert metadata["grounding_validation_passed"] is None
    assert metadata["ranking_mode"] == "balanced"
    assert metadata["query_intent"] in {"exact", "semantic", "hybrid"}
    assert metadata["explicit_test_intent"] is False
    assert metadata["test_penalty_applied"] is False


def test_search_json_source_first_ranking_mode_is_forwarded_and_reflected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --ranking-mode should be preserved in metadata while router applies deterministic backend defaults."""
    runner = CliRunner()
    captured_filters: dict[str, str] = {}

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query
            _ = top_k
            _ = context_radius
            captured_filters.update(filters)
            return {
                "query": "needle query",
                "results": [],
                "metadata": {
                    "total_results": 0,
                    "search_time_ms": 1,
                    "ranking_mode": "source-first",
                    "query_intent": "semantic",
                    "explicit_test_intent": False,
                    "test_penalty_applied": True,
                },
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(
        cli_main.cli,
        ["search", "needle query", "--json", "--mode", "semantic", "--ranking-mode", "source-first"],
    )

    assert result.exit_code == 0, result.output
    assert captured_filters["ranking_mode"] == "balanced"
    payload = _parse_json_output(result.output)
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["ranking_mode"] == "source-first"
    assert metadata["query_intent"] == "semantic"
    assert metadata["explicit_test_intent"] is False
    assert metadata["test_penalty_applied"] is False


def test_search_json_context_radius_is_forwarded_and_reflected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --context-radius should be reflected in metadata while semantic backend uses router-default snippet radius."""
    runner = CliRunner()
    captured_context_radius: list[int] = []

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return cli_main.GLOGGUR_VERSION

    class FakeEmbedding:
        provider = "local"

    class FakeHybridSearch:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def search(
            self,
            query: str,
            *,
            filters: dict[str, str],
            top_k: int,
            context_radius: int,
        ) -> dict[str, object]:
            _ = query
            _ = filters
            _ = top_k
            captured_context_radius.append(context_radius)
            return {
                "query": "needle",
                "results": [],
                "metadata": {"total_results": 0, "search_time_ms": 1},
            }

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )
    monkeypatch.setattr(cli_main, "MetadataStore", lambda _cfg: object())
    monkeypatch.setattr(cli_main, "HybridSearch", FakeHybridSearch)

    result = runner.invoke(
        cli_main.cli,
        [
            "search",
            "needle",
            "--json",
            "--mode",
            "semantic",
            "--context-radius",
            "15",
            "--max-requery-attempts",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_context_radius == [8]
    payload = _parse_json_output(result.output)
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["context_radius"] == 15


def test_search_json_wraps_metadata_store_connect_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should map metadata-store sqlite failures to structured io_failure."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> None:
            return None

    class FakeVectorStore:
        def search(self, _query_vector: list[float], k: int) -> list[tuple[str, float]]:
            _ = k
            return [("symbol-1", 0.0)]

    class FakeEmbedding:
        provider = "local"

        def embed_text(self, _text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), FakeVectorStore()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(metadata_store_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "path_not_writable"
    assert error["operation"] == "open metadata database connection"
    assert str(error["path"]).endswith("index.db")
    assert "unable to open database file" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo, cfg: ["status", "--json", "--config", str(cfg)]),
        ("search", lambda repo, cfg: ["search", "needle", "--json", "--config", str(cfg)]),
        ("inspect", lambda repo, cfg: ["inspect", str(repo), "--json", "--config", str(cfg)]),
        ("clear-cache", lambda repo, cfg: ["clear-cache", "--json", "--config", str(cfg)]),
        ("index", lambda repo, cfg: ["index", str(repo), "--json", "--config", str(cfg)]),
    ],
)
def test_core_commands_wrap_malformed_config_as_structured_io_failure(
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path, Path], list[str]],
) -> None:
    """Core commands should map malformed config files to stable io_failure payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n" "    return a + b\n",
        encoding="utf8",
    )
    bad_config = tmp_path / "bad.gloggur.json"
    bad_config.write_text("{not-valid-json", encoding="utf8")

    result = runner.invoke(cli_main.cli, build_args(repo, bad_config))

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "read gloggur config"
    assert str(error["path"]) == str(bad_config)
    assert "malformed" in str(error["probable_cause"]).lower()
    remediation = error.get("remediation")
    assert isinstance(remediation, list)
    assert remediation
    assert "fix config syntax" in str(remediation[0]).lower()
    assert "JSONDecodeError" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo, cfg: ["status", "--json", "--config", str(cfg)]),
        ("search", lambda repo, cfg: ["search", "needle", "--json", "--config", str(cfg)]),
        ("inspect", lambda repo, cfg: ["inspect", str(repo), "--json", "--config", str(cfg)]),
        ("clear-cache", lambda repo, cfg: ["clear-cache", "--json", "--config", str(cfg)]),
        ("index", lambda repo, cfg: ["index", str(repo), "--json", "--config", str(cfg)]),
    ],
)
def test_core_commands_wrap_non_mapping_config_payload_as_structured_io_failure(
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path, Path], list[str]],
) -> None:
    """Core commands should map non-mapping config payloads to stable io_failure payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n" "    return a + b\n",
        encoding="utf8",
    )
    bad_config = tmp_path / "bad.gloggur.yaml"
    bad_config.write_text("- item1\n- item2\n", encoding="utf8")

    result = runner.invoke(cli_main.cli, build_args(repo, bad_config))

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "read gloggur config"
    assert str(error["path"]) == str(bad_config)
    assert "malformed" in str(error["probable_cause"]).lower()
    remediation = error.get("remediation")
    assert isinstance(remediation, list)
    assert remediation
    assert "top-level mapping" in str(remediation[0]).lower()
    assert "ValueError" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    "command",
    ["status", "search", "inspect", "clear-cache", "index"],
)
def test_core_commands_wrap_sqlite_database_error_as_structured_io_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    """Core commands should surface sqlite DatabaseError as structured IO failures."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n" "    return a + b\n",
        encoding="utf8",
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)

    args_map = {
        "status": ["status", "--json"],
        "search": ["search", "add", "--json"],
        "inspect": ["inspect", str(repo), "--json"],
        "clear-cache": ["clear-cache", "--json"],
        "index": ["index", str(repo), "--json"],
    }
    result = runner.invoke(
        cli_main.cli,
        args_map[command],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        },
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert "cache database" in str(error["operation"])
    assert str(error["path"]).endswith("index.db")
    assert "database disk image is malformed" in str(error["detail"])


@pytest.mark.parametrize(
    ("exception_factory", "category", "detail_substring"),
    [
        (
            lambda: PermissionError(errno.EACCES, "permission denied"),
            "permission_denied",
            "PermissionError",
        ),
        (
            lambda: OSError(errno.EROFS, "read-only filesystem"),
            "read_only_filesystem",
            "read-only filesystem",
        ),
        (
            lambda: sqlite3.OperationalError("database or disk is full"),
            "disk_full_or_quota",
            "database or disk is full",
        ),
        (
            lambda: sqlite3.OperationalError("unable to open database file"),
            "path_not_writable",
            "unable to open database file",
        ),
        (
            lambda: OSError(errno.EIO, "i/o error"),
            "unknown_io_error",
            "i/o error",
        ),
    ],
)
@pytest.mark.parametrize(
    "command",
    ["status", "search", "inspect", "clear-cache", "index"],
)
def test_core_commands_wrap_io_failure_categories_consistently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_factory: Callable[[], Exception],
    category: str,
    detail_substring: str,
    command: str,
) -> None:
    """Core commands should classify IO failure categories consistently."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n" "    return a + b\n",
        encoding="utf8",
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise exception_factory()

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    args_map = {
        "status": ["status", "--json"],
        "search": ["search", "add", "--json"],
        "inspect": ["inspect", str(repo), "--json"],
        "clear-cache": ["clear-cache", "--json"],
        "index": ["index", str(repo), "--json"],
    }
    result = runner.invoke(
        cli_main.cli,
        args_map[command],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        },
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == category
    assert str(error["path"]).endswith("index.db")
    assert detail_substring in str(error["detail"])
    remediation = error.get("remediation")
    assert isinstance(remediation, list) and remediation
