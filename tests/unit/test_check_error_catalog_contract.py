from __future__ import annotations

from pathlib import Path

from scripts.check_error_catalog_contract import check_error_catalog_contract


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")


def test_error_catalog_contract_passes_when_required_headings_and_codes_exist(
    tmp_path: Path,
) -> None:
    docs_path = tmp_path / "docs" / "ERROR_CODES.md"

    _write(
        docs_path,
        "\n".join(
            [
                "## CLI Contract Errors",
                "`cli_usage_error`",
                "`watch_mode_conflict`",
                "`watch_path_missing`",
                "`watch_mode_invalid`",
                "`allow_tool_version_drift_env_invalid`",
                "`artifact_source_missing`",
                "`artifact_source_not_directory`",
                "`artifact_source_uninitialized`",
                "`artifact_destination_unsupported`",
                "`artifact_destination_exists`",
                "`artifact_destination_inside_source`",
                "`artifact_path_missing`",
                "`artifact_path_not_file`",
                "`artifact_archive_invalid`",
                "`artifact_manifest_missing`",
                "`artifact_manifest_invalid`",
                "`artifact_manifest_schema_unsupported`",
                "`artifact_manifest_file_mismatch`",
                "`artifact_manifest_totals_mismatch`",
                "`artifact_restore_destination_exists`",
                "`artifact_restore_destination_not_directory`",
                "`artifact_uploader_command_invalid`",
                "`artifact_uploader_failed`",
                "`artifact_uploader_timeout`",
                "`artifact_http_upload_failed`",
                "`artifact_http_upload_timeout`",
                "`search_top_k_invalid`",
                "`search_confidence_threshold_invalid`",
                "`search_max_requery_attempts_invalid`",
                "`search_result_payload_invalid`",
                "`search_evidence_min_confidence_invalid`",
                "`search_evidence_min_items_invalid`",
                "`search_evidence_trace_invalid`",
                "`search_grounding_validation_failed`",
                "`search_stream_contract_conflict`",
                "## Index Failure Codes",
                "`decode_error`",
                "`read_error`",
                "`parser_unavailable`",
                "`parse_error`",
                "`storage_error`",
                "`embedding_provider_error`",
                "`stale_cleanup_error`",
                "`vector_metadata_mismatch`",
                "`vector_consistency_unverifiable`",
                "## Inspect Failure Codes",
                "`decode_error`",
                "`read_error`",
                "`parser_unavailable`",
                "`parse_error`",
                "## Watch Status Failure Codes",
                "`watch_state_inconsistent`",
                "`watch_last_batch_inconsistent`",
                "## Resume Reason Codes",
                "`missing_index_metadata`",
                "`index_interrupted`",
                "`missing_cached_profile`",
                "`embedding_profile_changed`",
                "`tool_version_changed`",
                "`tool_version_changed_override`",
                "`cache_corruption_recovered`",
                "`cache_schema_rebuilt`",
            ]
        ),
    )

    payload = check_error_catalog_contract(docs_path)

    assert payload["ok"] is True
    assert payload["missing_headings"] == []
    assert payload["missing_codes_by_section"] == {}


def test_error_catalog_contract_fails_when_required_content_is_missing(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs" / "ERROR_CODES.md"
    _write(docs_path, "# incomplete\n")

    payload = check_error_catalog_contract(docs_path)

    assert payload["ok"] is False
    assert payload["failure"]["code"] == "error_catalog_contract_violation"
    assert "## CLI Contract Errors" in payload["missing_headings"]
    missing_codes = payload["missing_codes_by_section"]
    assert "cli" in missing_codes
    assert "cli_usage_error" in missing_codes["cli"]


def test_error_catalog_contract_fails_when_docs_file_is_missing(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs" / "ERROR_CODES.md"

    payload = check_error_catalog_contract(docs_path)

    assert payload["ok"] is False
    assert payload["failure"]["code"] == "error_catalog_docs_missing"
