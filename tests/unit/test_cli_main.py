from __future__ import annotations

import errno
import json
import os
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

import gloggur.indexer.cache as cache_module
import gloggur.storage.metadata_store as metadata_store_module
import gloggur.storage.vector_store as vector_store_module
import gloggur.support_runtime as support_runtime_module
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
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager, CacheRecoveryError
from gloggur.io_failures import StorageIOError
from gloggur.models import EdgeRecord, IndexMetadata, Symbol, SymbolChunk
from gloggur.search import attach_legacy_search_contract
from gloggur.search.router.types import ContextHit, ContextPack, ContextSpan
from gloggur.support_runtime import (
    SUPPORT_RUNTIME_DEGRADED_WARNING_CODE,
    write_support_runtime_config,
)


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


def test_profile_reindex_reason_allows_legacy_unsuffixed_edge_profiles_for_reads() -> None:
    """Read path should accept legacy cache profiles predating `embed_graph_edges`."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:microsoft/codebert-base",
        expected_profile="local:microsoft/codebert-base|embed_graph_edges=0",
        allow_legacy_unsuffixed_edges=True,
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


def test_build_status_payload_surfaces_missing_search_integrity_markers(tmp_path: Path) -> None:
    """Status should expose semantic suppression when integrity markers have not been persisted."""
    cache_dir = str(tmp_path / "cache")
    config = cli_main.GloggurConfig(
        cache_dir=cache_dir,
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
    )
    cache = CacheManager(CacheConfig(cache_dir))
    cache.set_index_metadata(IndexMetadata(version="1", total_symbols=2, indexed_files=1))
    cache.set_index_profile(config.embedding_profile())

    payload = cli_main._build_status_payload(config, cache)

    assert payload["semantic_search_allowed"] is False
    assert "vector_integrity_missing" in payload["warning_codes"]
    assert "chunk_span_integrity_missing" in payload["warning_codes"]
    assert payload["index_stats"]["embedded_edge_vectors"] == 0
    assert payload["extension_policy"]["valid"] is True
    assert payload["language_support_contract"]["schema_version"] == "1"
    assert ".c" in payload["language_support_contract"]["supported_extensions"]
    assert ".cpp" in payload["language_support_contract"]["supported_extensions"]
    assert ".jsx" in payload["language_support_contract"]["supported_extensions"]


def test_build_status_payload_zeroes_total_symbols_for_non_ready_cache(tmp_path: Path) -> None:
    """Status should separate raw symbol rows from searchable rows on non-ready caches."""
    cache_dir = str(tmp_path / "cache")
    config = cli_main.GloggurConfig(
        cache_dir=cache_dir,
        embedding_provider="test",
        local_embedding_model="model-a",
    )
    cache = CacheManager(CacheConfig(cache_dir))
    cache.upsert_symbols(
        [
            Symbol(
                id="sample.py:1:add",
                name="add",
                kind="function",
                file_path="sample.py",
                start_line=1,
                end_line=2,
                signature="def add(a, b):",
                body_hash="abc123",
            )
        ]
    )
    active_pid = os.getpid()
    cache.write_build_state(
        {
            "state": "building",
            "build_id": "build-1",
            "pid": active_pid,
            "started_at": "2026-03-07T00:00:00+00:00",
            "updated_at": "2026-03-07T00:00:01+00:00",
            "stage": "embed_chunks",
            "cleanup_pending": False,
        }
    )

    payload = cli_main._build_status_payload(config, cache)

    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["build_in_progress", "missing_index_metadata"]
    assert payload["raw_total_symbols"] == 1
    assert payload["total_symbols"] == 0
    assert payload["index_stats"]["symbol_count"] == 1
    assert payload["index_stats"]["embedded_edge_vectors"] == 0
    assert payload["build_state"] == {
        "state": "building",
        "build_id": "build-1",
        "pid": active_pid,
        "started_at": "2026-03-07T00:00:00+00:00",
        "updated_at": "2026-03-07T00:00:01+00:00",
        "stage": "embed_chunks",
        "cleanup_pending": False,
    }


def test_build_status_payload_includes_index_stats(tmp_path: Path) -> None:
    """Status should surface chunk, edge, and embedded-vector counts separately."""
    cache_dir = str(tmp_path / "cache")
    config = cli_main.GloggurConfig(
        cache_dir=cache_dir,
        embedding_provider="test",
        local_embedding_model="model-a",
        embed_graph_edges=True,
    )
    cache = CacheManager(CacheConfig(cache_dir))
    cache.upsert_symbols(
        [
            Symbol(
                id="sym-1",
                name="alpha",
                kind="function",
                file_path="sample.py",
                start_line=1,
                end_line=3,
                signature="def alpha() -> None:",
                body_hash="body-1",
            )
        ]
    )
    cache.upsert_chunks(
        [
            SymbolChunk(
                chunk_id="chunk-1",
                symbol_id="sym-1",
                chunk_part_index=1,
                chunk_part_total=1,
                text="alpha body",
                file_path="sample.py",
                start_line=1,
                end_line=3,
                start_byte=0,
                end_byte=10,
                embedding_vector=[0.1, 0.2],
            )
        ]
    )
    cache.upsert_edges(
        [
            EdgeRecord(
                edge_id="edge-1",
                edge_type="CALLS",
                from_id="sym-1",
                to_id="sym-2",
                from_kind="function",
                to_kind="function",
                file_path="sample.py",
                line=2,
                confidence=0.8,
                text="EDGE_TYPE: CALLS",
                embedding_vector=[0.3, 0.4],
            )
        ]
    )
    cache.set_index_metadata(IndexMetadata(version="1", total_symbols=1, indexed_files=1))
    cache.set_index_profile(config.embedding_profile())
    cache.set_search_integrity(
        {
            "vector_cache": {"status": "passed", "reason_codes": []},
            "chunk_span": {"status": "passed", "reason_codes": []},
        }
    )

    payload = cli_main._build_status_payload(config, cache)

    assert payload["index_stats"] == {
        "symbol_count": 1,
        "chunk_count": 1,
        "graph_edge_count": 1,
        "embedded_symbol_vectors": 1,
        "embedded_edge_vectors": 1,
        "embedded_vector_count": 2,
    }


def test_build_status_payload_marks_dead_build_pid_as_stale(tmp_path: Path) -> None:
    """Dead build PIDs should be treated as stale instead of active in-progress builds."""
    cache_dir = str(tmp_path / "cache")
    config = cli_main.GloggurConfig(
        cache_dir=cache_dir,
        embedding_provider="test",
        local_embedding_model="model-a",
    )
    cache = CacheManager(CacheConfig(cache_dir))
    cache.write_build_state(
        {
            "state": "building",
            "build_id": "build-stale",
            "pid": 424242,
            "started_at": "2026-03-07T00:00:00+00:00",
            "updated_at": "2026-03-07T00:00:01+00:00",
            "stage": "embed_chunks",
            "cleanup_pending": False,
        }
    )

    payload = cli_main._build_status_payload(config, cache)

    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["stale_build_state", "missing_index_metadata"]
    assert "stale_build_state" in payload["warning_codes"]
    assert "build_in_progress" not in payload["warning_codes"]
    assert payload["build_state"] == {
        "state": "interrupted",
        "build_id": "build-stale",
        "pid": 424242,
        "started_at": "2026-03-07T00:00:00+00:00",
        "updated_at": "2026-03-07T00:00:01+00:00",
        "stage": "embed_chunks",
        "cleanup_pending": True,
    }


def test_warm_embedding_provider_bootstraps_local_only() -> None:
    """Pre-lock warmup should bootstrap local models but not remote providers."""

    class _LocalProbe(EmbeddingProvider):
        provider = "local"

        def __init__(self) -> None:
            self.calls = 0

        def embed_text(self, text: str) -> list[float]:
            raise AssertionError("not used in this test")

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("not used in this test")

        def get_dimension(self) -> int:
            self.calls += 1
            return 768

    class _RemoteProbe(EmbeddingProvider):
        provider = "openai"

        def embed_text(self, text: str) -> list[float]:
            raise AssertionError("not used in this test")

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("not used in this test")

        def get_dimension(self) -> int:
            raise AssertionError("remote providers should not be probed pre-lock")

    local = _LocalProbe()

    assert cli_main._warm_embedding_provider(local) == {"embedding_dimension": 768}
    assert local.calls == 1
    assert cli_main._warm_embedding_provider(_RemoteProbe()) == {}


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


def test_index_stage_recorder_preserves_order_and_not_run_entries() -> None:
    """Stage recorder should keep deterministic order and default not-run status."""
    recorder = cli_main.IndexStageRecorder()
    recorder.record(
        "bootstrap_model",
        status="completed",
        duration_ms=12,
        counts={"embedding_dimension": 768},
    )
    recorder.record(
        "update_symbol_index",
        status="failed",
        duration_ms=34,
        counts={"failed": 1},
    )

    payload = recorder.as_payload()

    assert [stage["name"] for stage in payload] == list(cli_main.INDEX_STAGE_ORDER)
    assert payload[0]["status"] == "completed"
    assert payload[0]["counts"] == {"embedding_dimension": 768}
    commit_stage = next(stage for stage in payload if stage["name"] == "commit_metadata")
    assert commit_stage["status"] == "not_run"
    symbol_stage = next(stage for stage in payload if stage["name"] == "update_symbol_index")
    assert symbol_stage["status"] == "failed"


def test_terminate_index_children_reports_child_and_resource_tracker_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupt cleanup should terminate active children and the stdlib resource tracker."""
    terminated_pids: list[tuple[int, int]] = []

    class FakeChild:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.exitcode = None

        def terminate(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout

    monkeypatch.setattr(cli_main.multiprocessing, "active_children", lambda: [FakeChild(111)])
    monkeypatch.setattr(cli_main, "_current_resource_tracker_pid", lambda: 222)
    monkeypatch.setattr(cli_main.os, "kill", lambda pid, sig: terminated_pids.append((pid, sig)))

    payload = cli_main._terminate_index_children()

    assert payload == {
        "terminated_child_pids": [111],
        "resource_tracker_pid": 222,
    }
    assert terminated_pids == [
        (111, cli_main.signal.SIGKILL),
        (222, cli_main.signal.SIGTERM),
    ]


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
        "repo_config_trust_env_invalid",
        "local_fallback_env_unsupported",
    ):
        assert code in CLI_FAILURE_REMEDIATION
        guidance = CLI_FAILURE_REMEDIATION[code]
        assert isinstance(guidance, list)
        assert guidance


def test_cli_failure_catalog_includes_support_codes() -> None:
    """Support bundle failure codes should stay documented in the CLI failure catalog."""
    for code in (
        "support_command_invalid",
        "support_session_missing",
        "support_session_invalid",
        "support_destination_exists",
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
        "artifact_manifest_provenance_missing",
        "artifact_manifest_sha256_mismatch",
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


def test_cli_failure_catalog_includes_find_about_conflict_code() -> None:
    """`find --about` contract conflicts should stay documented in the CLI failure catalog."""
    guidance = CLI_FAILURE_REMEDIATION["find_about_contract_conflict"]
    assert isinstance(guidance, list)
    assert guidance


def test_resolve_router_repo_root_keeps_workspace_root_for_relative_paths(tmp_path: Path) -> None:
    """Repo-relative symbol paths should keep the caller workspace root as router root."""

    class FakeMetadataStore:
        def sample_symbol_file_paths(self, *, limit: int = 64) -> list[str]:
            assert limit == 64
            return ["src/a.py", "src/nested/b.py"]

    resolved = cli_main._resolve_router_repo_root(
        metadata_store=FakeMetadataStore(),
        fallback=tmp_path,
    )

    assert resolved == tmp_path


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
    assert "extension_policy" in payload
    assert payload["extension_policy"]["valid"] is True
    support_contract = payload["language_support_contract"]
    assert support_contract["schema_version"] == "1"
    assert ".c" in support_contract["supported_extensions"]
    assert ".cpp" in support_contract["supported_extensions"]
    assert ".jsx" in support_contract["supported_extensions"]
    assert ".tsx" in support_contract["supported_extensions"]
    assert ".html" not in support_contract["supported_extensions"]


def test_parsers_check_reports_required_and_known_gap_cases(tmp_path: Path) -> None:
    """parsers check should emit machine-readable required/known-gap summary."""
    runner = CliRunner()
    env = {"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")}

    result = runner.invoke(
        cli_main.cli,
        ["parsers", "check", "--json"],
        env=env,
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["schema_version"] == "1"
    assert payload["required_case_counts"]["failed"] == 0
    assert payload["known_gap_case_counts"]["total"] > 0
    assert payload["known_gap_case_counts"]["confirmed"] >= 1
    assert "language_support_contract" in payload
    assert payload["language_support_contract"]["schema_version"] == "1"


def test_parsers_check_exits_non_zero_when_capability_checks_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """parsers check should exit non-zero when required capability checks fail."""
    config = cli_main.GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
    )
    emitted: dict[str, object] = {}

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: config)
    monkeypatch.setattr(cli_main, "ParserRegistry", lambda **_kwargs: object())
    monkeypatch.setattr(
        cli_main,
        "run_parser_capability_check",
        lambda **_kwargs: {"ok": False, "required_case_counts": {"failed": 1}},
    )
    monkeypatch.setattr(
        cli_main,
        "_emit",
        lambda payload, as_json: emitted.update({"payload": payload, "as_json": as_json}),
    )

    with pytest.raises(cli_main.click.exceptions.Exit) as exc_info:
        cli_main._parsers_check_impl(config_path=None, as_json=True)

    assert exc_info.value.exit_code == 1
    assert emitted["as_json"] is True
    payload = emitted["payload"]
    assert isinstance(payload, dict)
    assert payload["ok"] is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("py", ".py"),
        (" PY ", ".py"),
        (".tsx", ".tsx"),
    ],
)
def test_normalize_extension_token_normalizes_case_spacing_and_dot_prefix(
    raw: str,
    expected: str,
) -> None:
    """Extension tokens should normalize to dot-prefixed lowercase values."""
    assert cli_main._normalize_extension_token(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "."])
def test_normalize_extension_token_rejects_empty_and_dot_only_tokens(raw: str) -> None:
    """Invalid extension tokens should fail closed before policy validation."""
    with pytest.raises(ValueError):
        cli_main._normalize_extension_token(raw)


def test_dedupe_extensions_preserves_order_and_drops_duplicates() -> None:
    """Duplicate extension tokens should be removed after normalization."""
    deduped = cli_main._dedupe_extensions([".py", "py", ".JS", ".js"])
    assert deduped == [".py", ".js"]


def test_validate_extension_policy_wraps_invalid_extension_tokens(
    tmp_path: Path,
) -> None:
    """Invalid extension tokens should surface as structured IO failures."""
    config = cli_main.GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
        supported_extensions=[" "],
    )
    config_path = tmp_path / "broken.gloggur.yaml"

    with pytest.raises(StorageIOError) as exc_info:
        cli_main._validate_extension_policy(config=config, config_path=str(config_path))

    error = exc_info.value
    assert error.operation == "validate extension policy"
    assert "invalid extension tokens" in error.probable_cause.lower()
    assert "ValueError" in error.detail


def test_extension_label_returns_placeholder_for_extensionless_paths() -> None:
    """Skipped-extension diagnostics should bucket extensionless files explicitly."""
    assert cli_main._extension_label("README") == "<no_extension>"


def test_build_skipped_extension_diagnostics_returns_empty_payload_when_disabled() -> None:
    """Disabled skipped-extension diagnostics should return an empty contract payload."""
    payload = cli_main._build_skipped_extension_diagnostics(
        enabled=False,
        counts={".txt": 2},
        samples=["notes.txt"],
    )
    assert payload == {
        "enabled": False,
        "warning_code": None,
        "skipped_files": 0,
        "by_extension": {},
        "sample_paths": [],
    }


def test_collect_index_skipped_extension_diagnostics_records_directory_unsupported_files(
    tmp_path: Path,
) -> None:
    """index skipped-extension diagnostics should record unsupported files in directories."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("def ok() -> None:\n    pass\n", encoding="utf8")
    (repo / "notes.txt").write_text("plain text\n", encoding="utf8")
    config = cli_main.GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
    )

    diagnostics = cli_main._collect_index_skipped_extension_diagnostics(
        path=str(repo),
        config=config,
        enabled=True,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["warning_code"] == "unsupported_extensions_skipped"
    assert diagnostics["skipped_files"] == 1
    assert diagnostics["by_extension"] == {".txt": 1}


def test_collect_index_skipped_extension_diagnostics_returns_empty_for_excluded_single_path(
    tmp_path: Path,
) -> None:
    """index skipped-extension diagnostics should ignore unsupported files under excluded dirs."""
    excluded_dir = tmp_path / "excluded"
    excluded_dir.mkdir()
    target = excluded_dir / "notes.txt"
    target.write_text("plain text\n", encoding="utf8")
    config = cli_main.GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
        excluded_dirs=["excluded"],
    )

    diagnostics = cli_main._collect_index_skipped_extension_diagnostics(
        path=str(target),
        config=config,
        enabled=True,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["warning_code"] is None
    assert diagnostics["skipped_files"] == 0


def test_collect_inspect_skipped_extension_diagnostics_records_directory_unsupported_files(
    tmp_path: Path,
) -> None:
    """inspect skipped-extension diagnostics should record unsupported files in-scope."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("def ok() -> None:\n    pass\n", encoding="utf8")
    (repo / "notes.txt").write_text("plain text\n", encoding="utf8")
    config = cli_main.GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
    )

    diagnostics = cli_main._collect_inspect_skipped_extension_diagnostics(
        path=str(repo),
        config=config,
        include_tests=False,
        include_scripts=False,
        enabled=True,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["warning_code"] == "unsupported_extensions_skipped"
    assert diagnostics["skipped_files"] == 1
    assert diagnostics["by_extension"] == {".txt": 1}


def test_collect_inspect_skipped_extension_diagnostics_returns_empty_for_excluded_path(
    tmp_path: Path,
) -> None:
    """inspect skipped-extension diagnostics should ignore excluded single-file paths."""
    excluded_dir = tmp_path / "excluded"
    excluded_dir.mkdir()
    target = excluded_dir / "notes.txt"
    target.write_text("plain text\n", encoding="utf8")
    config = cli_main.GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
        excluded_dirs=["excluded"],
    )

    diagnostics = cli_main._collect_inspect_skipped_extension_diagnostics(
        path=str(target),
        config=config,
        include_tests=False,
        include_scripts=False,
        enabled=True,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["warning_code"] is None
    assert diagnostics["skipped_files"] == 0


def test_collect_inspect_skipped_extension_diagnostics_respects_default_scope_filters(
    tmp_path: Path,
) -> None:
    """inspect skipped-extension diagnostics should not count out-of-scope test paths."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    target = tests_dir / "notes.txt"
    target.write_text("plain text\n", encoding="utf8")
    config = cli_main.GloggurConfig(
        cache_dir=str(tmp_path / "cache"),
        embedding_provider="test",
    )

    diagnostics = cli_main._collect_inspect_skipped_extension_diagnostics(
        path=str(target),
        config=config,
        include_tests=False,
        include_scripts=False,
        enabled=True,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["warning_code"] is None
    assert diagnostics["skipped_files"] == 0


def test_attach_skipped_extension_diagnostics_normalizes_existing_warning_codes() -> None:
    """warning_codes should be normalized, deduplicated, and sorted before emission."""
    payload: dict[str, object] = {"warning_codes": ["zeta", "", 1]}
    diagnostics = {
        "enabled": True,
        "warning_code": "unsupported_extensions_skipped",
        "skipped_files": 1,
        "by_extension": {".txt": 1},
        "sample_paths": ["notes.txt"],
    }

    cli_main._attach_skipped_extension_diagnostics(
        payload=payload,
        warn_on_skipped_extensions=True,
        diagnostics=diagnostics,
    )

    assert payload["warn_on_skipped_extensions"] is True
    assert payload["skipped_extension_diagnostics"] == diagnostics
    assert payload["warning_codes"] == ["1", "unsupported_extensions_skipped", "zeta"]


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
        expected_profile="local:microsoft/codebert-base|embed_graph_edges=0",
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
    assert payload["cached_resume_fingerprint"] == payload["expected_resume_fingerprint"]


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


def test_resume_contract_build_in_progress_has_dual_reason_codes() -> None:
    """An active build without metadata should expose build_in_progress plus a legacy code."""
    payload = _build_resume_contract(
        metadata=None,
        build_state={
            "state": "building",
            "build_id": "build-1",
            "pid": 123,
            "started_at": "2026-03-07T00:00:00+00:00",
            "updated_at": "2026-03-07T00:00:01+00:00",
            "stage": "scan_source",
            "cleanup_pending": False,
        },
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile=None,
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
    )

    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["build_in_progress", "missing_index_metadata"]


def test_resume_contract_stale_build_state_has_explicit_reason_codes() -> None:
    """Stale build state should emit explicit stale code before legacy metadata fallback."""
    payload = _build_resume_contract(
        metadata=None,
        build_state={
            "state": "interrupted",
            "build_id": "build-1",
            "pid": 424242,
            "started_at": "2026-03-07T00:00:00+00:00",
            "updated_at": "2026-03-07T00:00:01+00:00",
            "stage": "embed_chunks",
            "cleanup_pending": True,
        },
        schema_version="2",
        expected_profile="local:model-a",
        cached_profile=None,
        reset_reason=None,
        needs_reindex=True,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        stale_build_state=True,
    )

    assert payload["resume_decision"] == "reindex_required"
    assert payload["resume_reason_codes"] == ["stale_build_state", "missing_index_metadata"]
    remediation = payload["resume_remediation"]
    assert isinstance(remediation, dict)
    assert "stale_build_state" in remediation
    assert isinstance(remediation["stale_build_state"], list)
    assert remediation["stale_build_state"]


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


def test_resume_contract_treats_tool_version_drift_as_warning_only() -> None:
    """Tool-version drift should remain detectable without forcing reindex_required."""
    metadata = IndexMetadata(version="1", total_symbols=2, indexed_files=1)
    old_fingerprint = cli_main._resume_fingerprint(
        workspace_path_hash=cli_main._hash_content(os.path.abspath(os.getcwd())),
        schema_version="2",
        index_profile="local:model-a",
        metadata_digest=cli_main._index_metadata_digest(metadata),
        tool_version="0.1.0",
    )
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
    assert payload["resume_decision"] == "resume_ok"
    assert payload["resume_reason_codes"] == []
    assert payload["resume_fingerprint_match"] is True
    assert payload["last_success_tool_version_match"] is False
    assert payload["tool_version_drift_detected"] is True
    assert payload["tool_version_drift_override_applied"] is False
    assert payload["last_success_resume_fingerprint_match"] is True


def test_resume_contract_tool_version_override_input_is_backward_compatible_noop() -> None:
    """Explicit override input should not change the default advisory drift behavior."""
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
    assert payload["resume_reason_codes"] == []
    assert payload["tool_version_drift_detected"] is True
    assert payload["allow_tool_version_drift"] is True
    assert payload["tool_version_drift_override_applied"] is False
    assert payload["resume_remediation"] == {}


def test_resume_contract_tool_version_override_input_does_not_bypass_missing_metadata() -> None:
    """Override input must not mask true missing-metadata reindex requirements."""
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
    assert payload["tool_version_drift_override_applied"] is False
    codes = set(payload["resume_reason_codes"])
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
            return config.embedding_profile()

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


def test_persist_last_success_resume_state_repairs_only_tool_version_marker() -> None:
    """Pure tool-version drift with a matching fingerprint should update only the version marker."""

    writes: list[tuple[str, str]] = []
    metadata = IndexMetadata(
        version="1",
        total_symbols=5,
        indexed_files=2,
        last_updated=datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc),
    )
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir="/tmp/cache",
    )
    expected_fingerprint = _build_resume_contract(
        metadata=metadata,
        build_state=None,
        schema_version="2",
        expected_profile=config.embedding_profile(),
        cached_profile=config.embedding_profile(),
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        tool_version=cli_main.GLOGGUR_VERSION,
    )["expected_resume_fingerprint"]
    assert isinstance(expected_fingerprint, str)
    original_resume_at = "2026-03-01T00:00:00+00:00"

    class FakeCache:
        last_reset_reason = None
        _fingerprint = expected_fingerprint
        _at = original_resume_at
        _tool_version = "0.0.1"

        def get_index_metadata(self) -> IndexMetadata:
            return metadata

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return config.embedding_profile()

        def get_last_success_resume_fingerprint(self) -> str:
            return self._fingerprint

        def set_last_success_resume_fingerprint(self, fp: str) -> None:
            writes.append(("fingerprint", fp))
            self._fingerprint = fp

        def get_last_success_resume_at(self) -> str:
            return self._at

        def set_last_success_resume_at(self, ts: str) -> None:
            writes.append(("at", ts))
            self._at = ts

        def get_last_success_tool_version(self) -> str:
            return self._tool_version

        def set_last_success_tool_version(self, v: str) -> None:
            writes.append(("tool_version", v))
            self._tool_version = v

    cache = FakeCache()

    _persist_last_success_resume_state(config, cache)

    assert writes == [("tool_version", cli_main.GLOGGUR_VERSION)]
    assert cache._fingerprint == expected_fingerprint
    assert cache._at == original_resume_at


def test_persist_last_success_resume_state_rewrites_legacy_fingerprint_without_timestamp_bump() -> (
    None
):
    """Legacy tool-version-inclusive fingerprints should be rewritten in-place on no-op index."""

    writes: list[tuple[str, str]] = []
    metadata = IndexMetadata(
        version="1",
        total_symbols=5,
        indexed_files=2,
        last_updated=datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc),
    )
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir="/tmp/cache",
    )
    expected_fingerprint = _build_resume_contract(
        metadata=metadata,
        build_state=None,
        schema_version="2",
        expected_profile=config.embedding_profile(),
        cached_profile=config.embedding_profile(),
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        tool_version=cli_main.GLOGGUR_VERSION,
    )["expected_resume_fingerprint"]
    assert isinstance(expected_fingerprint, str)
    legacy_fingerprint = cli_main._resume_fingerprint(
        workspace_path_hash=cli_main._hash_content(os.path.abspath(os.getcwd())),
        schema_version="2",
        index_profile=config.embedding_profile(),
        metadata_digest=cli_main._index_metadata_digest(metadata),
        tool_version="0.0.1",
    )
    original_resume_at = "2026-03-01T00:00:00+00:00"

    class FakeCache:
        last_reset_reason = None
        _fingerprint = legacy_fingerprint
        _at = original_resume_at
        _tool_version = "0.0.1"

        def get_index_metadata(self) -> IndexMetadata:
            return metadata

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return config.embedding_profile()

        def get_last_success_resume_fingerprint(self) -> str:
            return self._fingerprint

        def set_last_success_resume_fingerprint(self, fp: str) -> None:
            writes.append(("fingerprint", fp))
            self._fingerprint = fp

        def get_last_success_resume_at(self) -> str:
            return self._at

        def set_last_success_resume_at(self, ts: str) -> None:
            writes.append(("at", ts))
            self._at = ts

        def get_last_success_tool_version(self) -> str:
            return self._tool_version

        def set_last_success_tool_version(self, v: str) -> None:
            writes.append(("tool_version", v))
            self._tool_version = v

    cache = FakeCache()

    _persist_last_success_resume_state(config, cache)

    assert writes == [
        ("fingerprint", expected_fingerprint),
        ("tool_version", cli_main.GLOGGUR_VERSION),
    ]
    assert cache._fingerprint == expected_fingerprint
    assert cache._at == original_resume_at
    assert cache._tool_version == cli_main.GLOGGUR_VERSION


def test_persist_last_success_resume_state_repairs_tool_version_drift_with_stale_fingerprint() -> (
    None
):
    """Pure tool-version drift with stale/missing last-success state should repair all markers."""

    writes: list[tuple[str, str]] = []
    metadata = IndexMetadata(
        version="1",
        total_symbols=5,
        indexed_files=2,
        last_updated=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
    )
    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir="/tmp/cache",
    )
    expected_fingerprint = _build_resume_contract(
        metadata=metadata,
        build_state=None,
        schema_version="2",
        expected_profile=config.embedding_profile(),
        cached_profile=config.embedding_profile(),
        reset_reason=None,
        needs_reindex=False,
        last_success_resume_fingerprint=None,
        last_success_resume_at=None,
        tool_version=cli_main.GLOGGUR_VERSION,
    )["expected_resume_fingerprint"]
    assert isinstance(expected_fingerprint, str)

    class FakeCache:
        last_reset_reason = None
        _fingerprint = "stale-fingerprint"
        _at = "2026-03-01T00:00:00+00:00"
        _tool_version = "0.0.1"

        def get_index_metadata(self) -> IndexMetadata:
            return metadata

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return config.embedding_profile()

        def get_last_success_resume_fingerprint(self) -> str:
            return self._fingerprint

        def set_last_success_resume_fingerprint(self, fp: str) -> None:
            writes.append(("fingerprint", fp))
            self._fingerprint = fp

        def get_last_success_resume_at(self) -> str:
            return self._at

        def set_last_success_resume_at(self, ts: str) -> None:
            writes.append(("at", ts))
            self._at = ts

        def get_last_success_tool_version(self) -> str:
            return self._tool_version

        def set_last_success_tool_version(self, v: str) -> None:
            writes.append(("tool_version", v))
            self._tool_version = v

    cache = FakeCache()

    _persist_last_success_resume_state(config, cache)

    assert writes == [
        ("fingerprint", expected_fingerprint),
        ("at", metadata.last_updated.isoformat()),
        ("tool_version", cli_main.GLOGGUR_VERSION),
    ]
    assert cache._fingerprint == expected_fingerprint
    assert cache._at == metadata.last_updated.isoformat()
    assert cache._tool_version == cli_main.GLOGGUR_VERSION


def test_build_status_payload_warns_on_tool_version_drift_without_requiring_reindex() -> None:
    """Status payload should treat tool-version drift as advisory by default."""
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
            return config.embedding_profile()

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def list_symbols(self) -> list[object]:
            return [object(), object(), object()]

        def get_search_integrity(self) -> dict[str, object]:
            return {
                "vector_cache": {"status": "passed", "reason_codes": []},
                "chunk_span": {"status": "passed", "reason_codes": []},
            }

    payload = cli_main._build_status_payload(config, FakeCache())

    assert payload["needs_reindex"] is False
    assert payload["reindex_reason"] is None
    assert payload["resume_decision"] == "resume_ok"
    assert payload["resume_reason_codes"] == []
    assert payload["resume_fingerprint_match"] is True
    assert payload["warning_codes"] == ["tool_version_changed"]
    assert payload["tool_version_drift_detected"] is True


def test_build_status_payload_accepts_tool_version_override_input_as_noop() -> None:
    """Status payload should keep backward-compatible override input without changing behavior."""
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
            return config.embedding_profile()

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def list_symbols(self) -> list[object]:
            return [object(), object(), object()]

        def get_search_integrity(self) -> dict[str, object]:
            return {
                "vector_cache": {"status": "passed", "reason_codes": []},
                "chunk_span": {"status": "passed", "reason_codes": []},
            }

    payload = cli_main._build_status_payload(
        config,
        FakeCache(),
        allow_tool_version_drift=True,
    )

    assert payload["needs_reindex"] is False
    assert payload["reindex_reason"] is None
    assert payload["resume_decision"] == "resume_ok"
    assert payload["resume_reason_codes"] == []
    assert payload["warning_codes"] == ["tool_version_changed"]
    assert payload["tool_version_drift_detected"] is True
    assert payload["allow_tool_version_drift"] is True
    assert payload["tool_version_drift_override_applied"] is False


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


def test_status_json_rejects_invalid_repo_config_trust_env_var() -> None:
    """status --json should fail closed on malformed repo trust env values."""
    runner = CliRunner()

    result = runner.invoke(
        cli_main.cli,
        ["status", "--json"],
        env={"GLOGGUR_REPO_CONFIG_TRUST": "sometimes"},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "repo_config_trust_env_invalid"
    assert payload["failure_codes"] == ["repo_config_trust_env_invalid"]


def test_status_json_includes_untrusted_repo_config_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """status JSON should surface config source, trust mode, warnings, and remote host details."""
    runner = CliRunner()
    config_path = tmp_path / ".gloggur.yaml"
    config_path.write_text(
        "cache_dir: cache\n"
        "embedding_provider: openai\n"
        "openai_base_url: https://proxy.example.test/v1\n",
        encoding="utf8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli_main.cli, ["status", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["config_source"] == "auto_discovered"
    assert payload["config_trust_mode"] == "auto"
    assert payload["remote_embedding"] == {
        "provider": "openai",
        "host": "proxy.example.test",
    }
    warning_codes = payload["security_warning_codes"]
    assert isinstance(warning_codes, list)
    assert "untrusted_repo_config" in warning_codes
    assert "untrusted_remote_provider_requested" in warning_codes
    assert "custom_embedding_endpoint_requested" in warning_codes


def test_status_json_marks_explicit_config_as_trusted(tmp_path: Path) -> None:
    """Explicit config paths should report explicit source and omit untrusted warning codes."""
    runner = CliRunner()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "cache_dir: cache\n" "embedding_provider: openai\n",
        encoding="utf8",
    )

    result = runner.invoke(
        cli_main.cli,
        ["status", "--json", "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["config_source"] == "explicit"
    assert payload["config_trust_mode"] == "auto"
    warning_codes = payload["security_warning_codes"]
    assert isinstance(warning_codes, list)
    assert "untrusted_repo_config" not in warning_codes


def test_decorate_payload_includes_trace_warning_codes_without_active_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace warnings should be additive even when config metadata is unavailable."""

    class _FakeTraceSession:
        def warning_codes(self) -> list[str]:
            return [SUPPORT_RUNTIME_DEGRADED_WARNING_CODE]

    monkeypatch.setattr(cli_main, "current_trace_session", lambda: _FakeTraceSession())
    payload = cli_main._decorate_payload_with_security_metadata({})
    assert payload["warning_codes"] == [SUPPORT_RUNTIME_DEGRADED_WARNING_CODE]


def test_status_json_remains_successful_when_support_runtime_metadata_write_degrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Support-runtime metadata write failures should degrade tracing, not fail status."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / ".git").mkdir()
    write_support_runtime_config(repo, enabled=True)
    monkeypatch.chdir(repo)

    def _raise_runtime_metadata_write(path: Path, payload: dict[str, object]) -> None:
        _ = payload
        if path.name == "meta.json":
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(path))
        raise AssertionError(f"unexpected path passed to _atomic_write_json: {path}")

    monkeypatch.setattr(
        support_runtime_module,
        "_atomic_write_json",
        _raise_runtime_metadata_write,
    )
    result = runner.invoke(
        cli_main.cli,
        ["status", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")},
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    warning_codes = payload.get("warning_codes")
    assert isinstance(warning_codes, list)
    assert SUPPORT_RUNTIME_DEGRADED_WARNING_CODE in warning_codes
    assert "FileNotFoundError" in result.output
    assert "support runtime tracing degraded" in result.output
    if "error" in payload:
        error = payload["error"]
        assert not (isinstance(error, dict) and error.get("type") == "io_failure")


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
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli_main.cli,
        build_args(repo),
        env={
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
            "GLOGGUR_CACHE_DIR": str(cache_dir),
        },
    )

    assert result.exit_code != 0
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "recover corrupted cache database"
    assert str(error["path"]) == str(cache_dir / "index.db")
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

        def get_search_integrity(self) -> dict[str, object]:
            return {
                "vector_cache": {"status": "passed", "reason_codes": []},
                "chunk_span": {"status": "passed", "reason_codes": []},
            }

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
        (
            "execute cache database transaction",
            "DatabaseError: database disk image is malformed",
        ),
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
    cache.set_index_profile("local:microsoft/codebert-base|embed_graph_edges=0")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json", "--profile-filter", "gemini"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is False
    assert payload["reason"] == "profile_filter_miss"
    assert payload["cached_index_profile"] == "local:microsoft/codebert-base|embed_graph_edges=0"
    reloaded = CacheManager(CacheConfig(str(cache_dir)))
    assert reloaded.get_index_profile() == "local:microsoft/codebert-base|embed_graph_edges=0"


def test_clear_cache_json_profile_filter_match_clears_cache(
    tmp_path: Path,
) -> None:
    """clear-cache with matching profile filter should clear cache artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache = CacheManager(CacheConfig(str(cache_dir)))
    cache.set_index_profile("local:microsoft/codebert-base|embed_graph_edges=0")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json", "--profile-filter", "codebert"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert payload["cached_index_profile"] == "local:microsoft/codebert-base|embed_graph_edges=0"
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
    """search --json should stay available via deterministic non-semantic backends."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / ".gloggur.yaml"
    config_path.write_text(
        'embedding_provider: ""\n' f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )
    cache = CacheManager(CacheConfig(str(cache_dir)))
    cache.set_index_metadata(IndexMetadata(version="1", total_symbols=0, indexed_files=0))
    cache.set_index_profile(":unknown|embed_graph_edges=0")

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


def test_search_json_allows_tool_version_drift_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should allow pure tool-version drift and emit a warning."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return config.embedding_profile()

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

        def get_search_integrity(self) -> dict[str, object]:
            return {
                "vector_cache": {"status": "passed", "reason_codes": []},
                "chunk_span": {"status": "passed", "reason_codes": []},
            }

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

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["results"]
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["needs_reindex"] is False
    assert metadata["reindex_reason"] is None
    assert metadata["resume_decision"] == "resume_ok"
    assert metadata["resume_reason_codes"] == []
    assert "tool_version_changed" in set(metadata["warning_codes"])
    assert metadata["tool_version_drift_detected"] is True


def test_search_json_requires_reindex_while_build_is_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should return the stable non-ready error contract for active builds."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            return None

        def get_build_state(self) -> dict[str, object]:
            return {
                "state": "building",
                "build_id": "build-1",
                "pid": 123,
                "started_at": "2026-03-07T00:00:00+00:00",
                "updated_at": "2026-03-07T00:00:01+00:00",
                "stage": "embed_chunks",
                "cleanup_pending": False,
            }

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

        def get_search_integrity(self) -> None:
            return None

    config = cli_main.GloggurConfig(
        embedding_provider="test",
        local_embedding_model="model-a",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(cli_main, "is_process_running", lambda _pid: True)

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 1, result.output
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "search_cache_not_ready"
    metadata = payload["metadata"]
    assert metadata["build_state"] == {
        "state": "building",
        "build_id": "build-1",
        "pid": 123,
        "started_at": "2026-03-07T00:00:00+00:00",
        "updated_at": "2026-03-07T00:00:01+00:00",
        "stage": "embed_chunks",
        "cleanup_pending": False,
    }
    assert metadata["resume_reason_codes"] == ["build_in_progress", "missing_index_metadata"]


def test_search_json_reports_stale_build_state_for_dead_pid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should classify dead-PID build markers as stale, not in-progress."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            return None

        def get_build_state(self) -> dict[str, object]:
            return {
                "state": "building",
                "build_id": "build-1",
                "pid": 424242,
                "started_at": "2026-03-07T00:00:00+00:00",
                "updated_at": "2026-03-07T00:00:01+00:00",
                "stage": "embed_chunks",
                "cleanup_pending": False,
            }

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

        def get_search_integrity(self) -> None:
            return None

    config = cli_main.GloggurConfig(
        embedding_provider="test",
        local_embedding_model="model-a",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), object()),
    )
    monkeypatch.setattr(cli_main, "is_process_running", lambda _pid: False)

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 1, result.output
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "search_cache_not_ready"
    metadata = payload["metadata"]
    assert metadata["build_state"] == {
        "state": "interrupted",
        "build_id": "build-1",
        "pid": 424242,
        "started_at": "2026-03-07T00:00:00+00:00",
        "updated_at": "2026-03-07T00:00:01+00:00",
        "stage": "embed_chunks",
        "cleanup_pending": True,
    }
    assert metadata["resume_reason_codes"] == ["stale_build_state", "missing_index_metadata"]
    warning_codes = set(metadata["warning_codes"])
    assert "stale_build_state" in warning_codes
    assert "build_in_progress" not in warning_codes


def test_search_json_accepts_tool_version_drift_override_input_as_noop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should keep backward-compatible override input without changing behavior."""
    runner = CliRunner()

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return config.embedding_profile()

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> str:
            return "0.0.1"

        def get_search_integrity(self) -> dict[str, object]:
            return {
                "vector_cache": {"status": "passed", "reason_codes": []},
                "chunk_span": {"status": "passed", "reason_codes": []},
            }

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
    assert metadata["resume_reason_codes"] == []
    assert "tool_version_changed" in set(metadata["warning_codes"])
    assert metadata["tool_version_drift_detected"] is True
    assert metadata["allow_tool_version_drift"] is True
    assert metadata["tool_version_drift_override_applied"] is False


def test_search_json_retries_once_for_low_confidence_and_emits_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Low-confidence retrieval should stay single-pass with explicit metadata."""
    runner = CliRunner()
    search_top_k_calls: list[int] = []

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return config.embedding_profile()

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
            return config.embedding_profile()

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
            return config.embedding_profile()

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
            return config.embedding_profile()

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


def test_search_help_hides_legacy_grounding_flags() -> None:
    """Removed grounding flags should stay accepted but absent from public help."""
    runner = CliRunner()

    result = runner.invoke(cli_main.cli, ["search", "--help"])

    assert result.exit_code == 0
    for flag in (
        "--with-evidence-trace",
        "--validate-grounding",
        "--evidence-min-confidence",
        "--evidence-min-items",
        "--fail-on-ungrounded",
    ):
        assert flag not in result.output


def test_find_help_hides_legacy_grounding_flags() -> None:
    """find help should not advertise removed grounding options either."""
    runner = CliRunner()

    result = runner.invoke(cli_main.cli, ["find", "--help"])

    assert result.exit_code == 0
    for flag in (
        "--with-evidence-trace",
        "--validate-grounding",
        "--evidence-min-confidence",
        "--evidence-min-items",
        "--fail-on-ungrounded",
    ):
        assert flag not in result.output


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
            return config.embedding_profile()

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
            return config.embedding_profile()

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
            return config.embedding_profile()

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
    """search --ranking-mode should be preserved while router applies backend defaults."""
    runner = CliRunner()
    captured_filters: dict[str, str] = {}

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return config.embedding_profile()

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
        [
            "search",
            "needle query",
            "--json",
            "--mode",
            "semantic",
            "--ranking-mode",
            "source-first",
        ],
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
    """search --context-radius should be reflected while the backend keeps its default radius."""
    runner = CliRunner()
    captured_context_radius: list[int] = []

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_schema_version(self) -> str:
            return "2"

        def get_index_profile(self) -> str:
            return config.embedding_profile()

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
            return config.embedding_profile()

        def get_last_success_resume_fingerprint(self) -> None:
            return None

        def get_last_success_resume_at(self) -> None:
            return None

        def get_last_success_tool_version(self) -> None:
            return None

        def get_search_integrity(self) -> dict[str, object]:
            return {
                "vector_cache": {"status": "passed", "reason_codes": []},
                "chunk_span": {"status": "passed", "reason_codes": []},
            }

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


def _make_context_hit(
    rank: int,
    *,
    path: str | None = None,
    snippet: str | None = None,
    tags: tuple[str, ...] = ("literal_match", "symbol_match"),
) -> ContextHit:
    """Build a deterministic ContextPack hit for CLI tests."""
    return ContextHit(
        path=path or f"src/sample_{rank}.py",
        span=ContextSpan(start_line=rank, end_line=rank + 1),
        snippet=snippet or f"def sample_{rank}():\n    return {rank}",
        score=max(0.1, 1.0 - (rank * 0.01)),
        start_byte=rank * 10,
        end_byte=(rank * 10) + 8,
        tags=tags,
    )


def _make_context_pack(
    *,
    query: str = "needle",
    hits: tuple[ContextHit, ...] = (_make_context_hit(1),),
    summary_overrides: dict[str, object] | None = None,
    debug: dict[str, object] | None = None,
) -> ContextPack:
    """Build a deterministic ContextPack payload for CLI tests."""
    summary: dict[str, object] = {
        "strategy": "exact",
        "reason": "quality_threshold_met",
        "winner": "exact",
        "hits": len(hits),
        "warning_codes": [],
        "backend_thresholds": {},
        "query_kind": "identifier",
        "decisive": True,
        "next_action": "open_hit_1",
    }
    if summary_overrides:
        summary.update(summary_overrides)
    return ContextPack(
        query=query,
        summary=summary,
        hits=hits,
        debug=(
            debug
            if debug is not None
            else {"backend_scores": {"exact": 0.99}, "backend_errors": {}}
        ),
    )


def _install_routed_search_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pack: ContextPack,
    captures: dict[str, object],
    health_overrides: dict[str, object] | None = None,
    security_warning_codes: list[str] | None = None,
) -> cli_main.GloggurConfig:
    """Install a fake routed-search runtime that returns a fixed ContextPack."""
    config = cli_main.GloggurConfig(
        embedding_provider="test",
        local_embedding_model="model-a",
        cache_dir=str(tmp_path / "cache"),
    )
    if security_warning_codes is not None:
        config.security_warning_codes = list(security_warning_codes)

    base_health: dict[str, object] = {
        "needs_reindex": False,
        "reindex_reason": None,
        "resume_contract": {},
        "warning_codes": [],
        "semantic_search_allowed": True,
        "search_integrity": {
            "vector_cache": {"status": "passed", "reason_codes": []},
            "chunk_span": {"status": "passed", "reason_codes": []},
        },
        "build_state": None,
        "expected_index_profile": "local:model-a",
        "cached_index_profile": "local:model-a",
        "entrypoint": "search_cli_v2",
        "contract_version": "contextpack_v2",
    }
    if health_overrides:
        base_health.update(health_overrides)

    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, object(), object()),
    )

    def _build_health(_config: object, _cache: object, **kwargs: object) -> dict[str, object]:
        payload = dict(base_health)
        payload["entrypoint"] = kwargs["entrypoint"]
        payload["contract_version"] = kwargs["contract_version"]
        return payload

    monkeypatch.setattr(cli_main, "_build_search_health_snapshot", _build_health)
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda _cfg: object())
    monkeypatch.setattr(
        cli_main,
        "_resolve_router_repo_root",
        lambda **_kwargs: tmp_path,
    )
    monkeypatch.setattr(cli_main, "SymbolIndexStore", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli_main, "load_search_router_config", lambda _repo_root: object())

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            captures["router_init"] = kwargs

        def search(
            self,
            *,
            query: str,
            intent: object,
            mode: str,
            include_debug: bool,
        ) -> ContextPack:
            captures["query"] = query
            captures["mode"] = mode
            captures["include_debug"] = include_debug
            captures["intent"] = {
                "search_mode": getattr(intent, "search_mode", None),
                "semantic_query": getattr(intent, "semantic_query", None),
                "ranking_mode": getattr(intent, "ranking_mode", None),
                "result_profile": getattr(intent, "result_profile", None),
                "semantic_assist_mode": getattr(intent, "semantic_assist_mode", None),
                "language": getattr(intent, "language", None),
                "path_prefix": getattr(intent, "path_prefix", None),
                "max_files": getattr(intent, "max_files", None),
                "max_snippets": getattr(intent, "max_snippets", None),
                "time_budget_ms": getattr(intent, "time_budget_ms", None),
            }
            return pack

    monkeypatch.setattr(cli_main, "SearchRouter", FakeRouter)
    return config


def test_find_text_output_reports_decision_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find should emit a terse first-line status across result-decision cases."""
    runner = CliRunner()
    cases = [
        (
            _make_context_pack(),
            "status=decisive strategy=exact query_kind=identifier next=open_hit_1",
        ),
        (
            _make_context_pack(
                hits=(_make_context_hit(1), _make_context_hit(2)),
                summary_overrides={
                    "decisive": False,
                    "query_kind": "mixed",
                    "next_action": "narrow_by_path",
                    "suggested_path_prefix": "src/sample_1.py",
                },
            ),
            (
                "status=ambiguous strategy=exact query_kind=mixed next=narrow_by_path "
                "suggested_path_prefix=src/sample_1.py"
            ),
        ),
        (
            _make_context_pack(
                hits=(),
                summary_overrides={
                    "hits": 0,
                    "decisive": False,
                    "query_kind": "natural_language",
                    "next_action": "broaden_query",
                },
            ),
            "status=no_match strategy=exact query_kind=natural_language next=broaden_query",
        ),
        (
            _make_context_pack(
                hits=(),
                summary_overrides={
                    "strategy": "suppressed",
                    "hits": 0,
                    "decisive": False,
                    "query_kind": "natural_language",
                    "next_action": "broaden_query",
                    "warning_codes": ["ungrounded_results_suppressed"],
                },
            ),
            "status=suppressed strategy=suppressed query_kind=natural_language next=broaden_query",
        ),
    ]

    for pack, expected_first_line in cases:
        captures: dict[str, object] = {}
        _install_routed_search_runtime(
            monkeypatch,
            tmp_path,
            pack=pack,
            captures=captures,
        )
        result = runner.invoke(cli_main.cli, ["find", pack.query])
        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[0] == expected_first_line


def test_find_text_output_appends_suggested_next_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ambiguous find output should emit one reusable narrowing command."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack(
        query="needle",
        hits=(_make_context_hit(1), _make_context_hit(2)),
        summary_overrides={
            "decisive": False,
            "query_kind": "mixed",
            "next_action": "narrow_by_path",
            "suggested_path_prefix": "src/sample_1.py",
        },
    )
    _install_routed_search_runtime(monkeypatch, tmp_path, pack=pack, captures=captures)

    result = runner.invoke(cli_main.cli, ["find", "needle"])

    assert result.exit_code == 0, result.output
    assert "suggested_next_command: gloggur find needle --file src/sample_1.py" in result.output


def test_find_json_contract_is_slim_and_security_gated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find --json should emit the slim v1 contract and omit bulky search metadata."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack(
        summary_overrides={"warning_codes": ["identifier_query_high_top_k"]},
    )
    _install_routed_search_runtime(
        monkeypatch,
        tmp_path,
        pack=pack,
        captures=captures,
        security_warning_codes=["untrusted_repo_config"],
    )

    result = runner.invoke(cli_main.cli, ["find", "needle", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["schema_version"] == 1
    assert payload["contract_version"] == "find_v1"
    assert payload["query"] == "needle"
    assert payload["decision"] == {
        "status": "decisive",
        "strategy": "exact",
        "reason": "quality_threshold_met",
        "query_kind": "identifier",
        "next_action": "open_hit_1",
        "assist": "none",
    }
    assert payload["warning_codes"] == ["identifier_query_high_top_k"]
    assert payload["security_warning_codes"] == ["untrusted_repo_config"]
    assert payload["config_trust_mode"] == "auto"
    assert "config_source" not in payload
    assert "remote_embedding" not in payload
    assert "search_integrity" not in payload
    hit = payload["hits"][0]
    assert hit["rank"] == 1
    assert hit["start_byte"] == 10
    assert hit["end_byte"] == 18


def test_find_json_adds_assist_and_suggested_next_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find should surface assist state and an additive next-command hint."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack(
        query="needle",
        hits=(_make_context_hit(1), _make_context_hit(2)),
        summary_overrides={
            "decisive": False,
            "query_kind": "mixed",
            "next_action": "narrow_by_path",
            "suggested_path_prefix": "src/sample_1.py",
            "assist": "rerank",
        },
    )
    _install_routed_search_runtime(monkeypatch, tmp_path, pack=pack, captures=captures)

    result = runner.invoke(cli_main.cli, ["find", "needle", "--json", "--about", "cache warmup"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    decision = payload["decision"]
    assert decision["assist"] == "rerank"
    assert (
        decision["suggested_next_command"]
        == "gloggur find needle --file src/sample_1.py --about 'cache warmup'"
    )


def test_find_json_includes_about_and_forwards_semantic_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find --about should expose the semantic description and forward it to routing."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack(
        summary_overrides={"strategy": "hybrid", "reason": "semantic_query_requested"},
    )
    _install_routed_search_runtime(monkeypatch, tmp_path, pack=pack, captures=captures)

    result = runner.invoke(
        cli_main.cli,
        ["find", 'rg "token" src/', "--json", "--about", "cache warmup"],
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["about"] == "cache warmup"
    assert captures["query"] == 'rg "token" src/'
    assert captures["intent"]["semantic_query"] == "cache warmup"
    assert captures["intent"]["ranking_mode"] == "source-first"
    assert captures["intent"]["result_profile"] == "locator"
    assert captures["intent"]["semantic_assist_mode"] == "bounded_about"


def test_find_blank_about_is_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Blank `--about` input should behave the same as omitting the option."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    _install_routed_search_runtime(
        monkeypatch,
        tmp_path,
        pack=_make_context_pack(),
        captures=captures,
    )

    result = runner.invoke(
        cli_main.cli,
        ["find", "needle", "--json", "--about", "   "],
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert "about" not in payload
    assert captures["intent"]["semantic_query"] is None
    assert captures["intent"]["ranking_mode"] == "source-first"
    assert captures["intent"]["result_profile"] == "locator"
    assert captures["intent"]["semantic_assist_mode"] == "none"


def test_find_unprocessed_query_tokens_infer_file_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find should accept shell-level query tokens and peel a trailing file path."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    target = tmp_path / "src" / "flask" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("def make_response():\n    return None\n", encoding="utf8")
    monkeypatch.chdir(tmp_path)
    _install_routed_search_runtime(
        monkeypatch, tmp_path, pack=_make_context_pack(), captures=captures
    )

    result = runner.invoke(
        cli_main.cli,
        ["find", "make_response", "src/flask/app.py", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captures["query"] == "make_response"
    assert captures["intent"]["path_prefix"] == str(target.resolve())


def test_find_unprocessed_rg_tokens_infer_directory_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find should preserve grep-like tokens while peeling a trailing directory path."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    _install_routed_search_runtime(
        monkeypatch, tmp_path, pack=_make_context_pack(), captures=captures
    )

    result = runner.invoke(
        cli_main.cli,
        ["find", "rg", "-S", "-g", "*.py", "AuthToken", "src", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captures["query"] == "rg -S -g *.py AuthToken"
    assert captures["intent"]["path_prefix"] == str(source_dir.resolve())


def test_find_explicit_scope_disables_implicit_trailing_path_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit scope flags should keep trailing tokens in the lexical query."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    explicit_target = tmp_path / "src"
    explicit_target.mkdir()
    trailing = tmp_path / "docs"
    trailing.mkdir()
    monkeypatch.chdir(tmp_path)
    _install_routed_search_runtime(
        monkeypatch, tmp_path, pack=_make_context_pack(), captures=captures
    )

    result = runner.invoke(
        cli_main.cli,
        ["find", "AuthToken", "docs", "--path-prefix", "src", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert captures["query"] == "AuthToken docs"
    assert captures["intent"]["path_prefix"] == str(explicit_target.resolve())
    assert "warning_codes" not in payload


def test_find_after_double_dash_preserves_query_tokens_named_like_cli_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Tokens after `--` should stay part of the find query."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    _install_routed_search_runtime(
        monkeypatch, tmp_path, pack=_make_context_pack(), captures=captures
    )

    result = runner.invoke(
        cli_main.cli,
        ["find", "--json", "--", "rg", "--about", "token"],
    )

    assert result.exit_code == 0, result.output
    assert captures["query"] == "rg --about token"


def test_find_pathlike_trailing_token_missing_recovers_when_query_remains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Recoverable missing trailing path-like tokens should not block find."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack(
        query="make_response",
        hits=(
            _make_context_hit(1, path="src/flask/app.py"),
            _make_context_hit(2, path="src/flask/app.py"),
        ),
        summary_overrides={
            "decisive": False,
            "query_kind": "identifier",
            "next_action": "narrow_by_path",
            "suggested_path_prefix": "src/flask/app.py",
        },
    )
    monkeypatch.chdir(tmp_path)
    _install_routed_search_runtime(monkeypatch, tmp_path, pack=pack, captures=captures)

    result = runner.invoke(
        cli_main.cli,
        ["find", "make_response", "src/flask/app.py", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    warning_codes = payload.get("warning_codes")
    assert isinstance(warning_codes, list)
    assert cli_main.FIND_TRAILING_PATH_MISSING_RECOVERED_WARNING in warning_codes
    assert captures["query"] == "make_response"
    decision = payload["decision"]
    assert (
        decision["suggested_next_command"]
        == "gloggur find make_response --file src/flask/app.py"
    )


def test_find_pathlike_trailing_token_missing_text_output_adds_recovery_note(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Recovered find text output should call out the ignored missing path token."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack(query="make_response")
    monkeypatch.chdir(tmp_path)
    _install_routed_search_runtime(monkeypatch, tmp_path, pack=pack, captures=captures)

    result = runner.invoke(
        cli_main.cli,
        ["find", "make_response", "src/flask/app.py"],
    )

    assert result.exit_code == 0, result.output
    assert "ignored missing trailing path-like token 'src/flask/app.py'" in result.output
    assert "use --file or --path-prefix" in result.output
    assert captures["query"] == "make_response"


def test_find_pathlike_single_token_missing_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A path-like token with no remaining query should keep failing closed."""
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        cli_main.cli,
        ["find", "src/flask/app.py", "--json"],
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "find_trailing_path_missing"


def test_find_generic_dotted_trailing_token_remains_query_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generic dotted tokens should stay in the query unless they are clearly path-like."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    _install_routed_search_runtime(
        monkeypatch, tmp_path, pack=_make_context_pack(), captures=captures
    )

    result = runner.invoke(
        cli_main.cli,
        ["find", "AuthToken", "flask.app", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert captures["query"] == "AuthToken flask.app"
    assert "warning_codes" not in payload


@pytest.mark.parametrize("query", ["http:///example.com", "//example.com/path", r"\\example.com\\path"])
def test_find_url_like_single_token_remains_query_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    query: str,
) -> None:
    """Malformed URL-like literals should not be reinterpreted as missing scope paths."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    _install_routed_search_runtime(
        monkeypatch, tmp_path, pack=_make_context_pack(query=query), captures=captures
    )

    result = runner.invoke(cli_main.cli, ["find", query, "--json"])

    assert result.exit_code == 0, result.output
    assert captures["query"] == query


@pytest.mark.parametrize(
    "args",
    [
        ["find", "needle", "--json", "--about", "cache warmup", "--mode", "exact"],
        ["find", "needle", "--json", "--about", "cache warmup", "--mode", "semantic"],
        ["find", "needle", "--json", "--about", "cache warmup", "--search-mode", "by_path"],
    ],
)
def test_find_about_rejects_conflicting_modes(args: list[str]) -> None:
    """find --about should fail closed when flags would drop one side of the query."""
    runner = CliRunner()

    result = runner.invoke(cli_main.cli, args)

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "find_about_contract_conflict"
    assert payload["failure_codes"] == ["find_about_contract_conflict"]


@pytest.mark.parametrize(
    ("args", "expected_hits"),
    [
        (["find", "needle", "--json"], 5),
        (["find", "needle", "--json", "--top-k", "7"], 7),
        (["find", "needle", "--json", "--max-snippets", "3"], 3),
    ],
)
def test_find_hit_limit_defaults_to_five_but_honors_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    args: list[str],
    expected_hits: int,
) -> None:
    """find should default to five hits and respect explicit top-k/max-snippets overrides."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack(hits=tuple(_make_context_hit(index) for index in range(1, 11)))
    _install_routed_search_runtime(monkeypatch, tmp_path, pack=pack, captures=captures)

    result = runner.invoke(cli_main.cli, args)

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert len(payload["hits"]) == expected_hits
    assert captures["intent"]["max_snippets"] == expected_hits


def test_find_stream_emits_ndjson_hits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find --stream should emit one slim NDJSON object per hit."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack(hits=(_make_context_hit(1), _make_context_hit(2)))
    _install_routed_search_runtime(monkeypatch, tmp_path, pack=pack, captures=captures)

    result = runner.invoke(cli_main.cli, ["find", "needle", "--stream"])

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["rank"] == 1
    assert "decision" not in lines[0]
    assert lines[0]["start_byte"] == 10
    assert lines[0]["end_byte"] == 18


def test_find_stream_rejects_debug_router_with_structured_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find --stream should fail closed when router debug is requested."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    _install_routed_search_runtime(
        monkeypatch,
        tmp_path,
        pack=_make_context_pack(),
        captures=captures,
    )

    result = runner.invoke(cli_main.cli, ["find", "needle", "--stream", "--debug-router"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "find_stream_contract_conflict"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["SearchRouter", "--json", "--top-k", "4"],
        ["how to decode auth token", "--json", "--mode", "hybrid", "--top-k", "4"],
        [
            "src/services",
            "--json",
            "--search-mode",
            "by_path",
            "--path-prefix",
            "src/services",
            "--top-k",
            "4",
        ],
        ['rg -S "retry" src/', "--json", "--top-k", "4"],
    ],
)
def test_find_matches_search_router_invocation_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: list[str],
) -> None:
    """find should preserve shared routing fields while applying locator defaults."""
    runner = CliRunner()
    pack = _make_context_pack()

    search_capture: dict[str, object] = {}
    _install_routed_search_runtime(
        monkeypatch,
        tmp_path,
        pack=pack,
        captures=search_capture,
    )
    search_result = runner.invoke(cli_main.cli, ["search", *extra_args])
    assert search_result.exit_code == 0, search_result.output

    find_capture: dict[str, object] = {}
    _install_routed_search_runtime(
        monkeypatch,
        tmp_path,
        pack=pack,
        captures=find_capture,
    )
    find_result = runner.invoke(cli_main.cli, ["find", *extra_args])
    assert find_result.exit_code == 0, find_result.output

    assert find_capture["query"] == search_capture["query"]
    assert find_capture["mode"] == search_capture["mode"]
    for field in (
        "search_mode",
        "semantic_query",
        "language",
        "path_prefix",
        "max_files",
        "max_snippets",
        "time_budget_ms",
    ):
        assert find_capture["intent"][field] == search_capture["intent"][field]
    assert search_capture["intent"]["ranking_mode"] == "balanced"
    assert search_capture["intent"]["result_profile"] == "default"
    assert search_capture["intent"]["semantic_assist_mode"] == "none"
    assert find_capture["intent"]["ranking_mode"] == "source-first"
    assert find_capture["intent"]["result_profile"] == "locator"
    assert find_capture["intent"]["semantic_assist_mode"] == "none"


def test_search_json_contract_remains_contextpack_v2_after_find_addition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search should retain its existing ContextPack v2 success shape."""
    runner = CliRunner()
    captures: dict[str, object] = {}
    pack = _make_context_pack()
    _install_routed_search_runtime(monkeypatch, tmp_path, pack=pack, captures=captures)

    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["schema_version"] == 2
    assert "summary" in payload
    assert "decision" not in payload
    assert payload["hits"][0]["start_byte"] == 10
    assert payload["hits"][0]["end_byte"] == 18


def test_find_json_is_smaller_than_search_json_for_same_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """find --json should serialize fewer bytes than search --json for the same result set."""
    runner = CliRunner()
    pack = _make_context_pack(hits=tuple(_make_context_hit(index) for index in range(1, 6)))

    search_capture: dict[str, object] = {}
    _install_routed_search_runtime(
        monkeypatch,
        tmp_path,
        pack=pack,
        captures=search_capture,
    )
    search_result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])
    assert search_result.exit_code == 0, search_result.output

    find_capture: dict[str, object] = {}
    _install_routed_search_runtime(
        monkeypatch,
        tmp_path,
        pack=pack,
        captures=find_capture,
    )
    find_result = runner.invoke(cli_main.cli, ["find", "needle", "--json"])
    assert find_result.exit_code == 0, find_result.output

    assert len(find_result.output.encode("utf8")) < len(search_result.output.encode("utf8"))


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


@pytest.mark.parametrize(
    ("config_text", "detail_fragment"),
    [
        (
            "supported_extensions:\n" "  - .py\n" "  - .mjs\n",
            "missing parser mappings",
        ),
        (
            "supported_extensions:\n" "  - .py\n" "parser_extension_map:\n" "  .mjs: javascript\n",
            "missing from supported_extensions",
        ),
    ],
)
def test_status_wraps_invalid_extension_policy_as_structured_io_failure(
    tmp_path: Path,
    config_text: str,
    detail_fragment: str,
) -> None:
    """status should fail closed when extension policy config is inconsistent."""
    runner = CliRunner()
    bad_config = tmp_path / "bad.gloggur.yaml"
    bad_config.write_text(config_text, encoding="utf8")

    result = runner.invoke(cli_main.cli, ["status", "--json", "--config", str(bad_config)])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["operation"] == "validate extension policy"
    assert detail_fragment in str(error["detail"])
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
