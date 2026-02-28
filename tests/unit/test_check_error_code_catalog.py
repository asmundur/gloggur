from __future__ import annotations

from pathlib import Path

from scripts.check_error_code_catalog import check_error_code_catalog


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")


def test_error_code_catalog_passes_when_required_headings_and_codes_exist(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs" / "ERROR_CODES.md"
    _write(
        docs_path,
        "\n".join(
            [
                "# Error Code Catalog",
                "## CLI Contract Errors",
                "`cli_usage_error`",
                "## Embedding Provider Failures",
                "`embedding_provider_error`",
                "## Index Failure Codes",
                "`vector_metadata_mismatch`",
                "## Inspect Failure Codes",
                "`parse_error`",
                "## Watch Status Failure Codes",
                "`watch_state_inconsistent`",
                "## Resume Reason Codes",
                "`tool_version_changed`",
            ]
        ),
    )

    payload = check_error_code_catalog(
        docs_path,
        surface_code_map={
            "cli_preflight": ["cli_usage_error"],
            "embedding_provider": ["embedding_provider_error"],
            "index_watch_incremental": ["vector_metadata_mismatch"],
            "inspect_file": ["parse_error"],
            "watch_status": ["watch_state_inconsistent"],
            "resume": ["tool_version_changed"],
        },
    )

    assert payload["ok"] is True
    assert payload["missing_headings"] == []
    assert payload["missing_codes_by_surface"] == {}


def test_error_code_catalog_fails_when_heading_and_code_are_missing(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs" / "ERROR_CODES.md"
    _write(docs_path, "# Error Code Catalog\n")

    payload = check_error_code_catalog(
        docs_path,
        surface_code_map={
            "cli_preflight": ["cli_usage_error"],
            "embedding_provider": ["embedding_provider_error"],
        },
    )

    assert payload["ok"] is False
    assert payload["failure"]["code"] == "error_code_catalog_violation"
    assert "## CLI Contract Errors" in payload["missing_headings"]
    assert payload["missing_codes_by_surface"]["cli_preflight"] == ["cli_usage_error"]
