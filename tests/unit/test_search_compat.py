from __future__ import annotations

from gloggur.search.compat import attach_legacy_search_contract


def test_attach_legacy_search_contract_preserves_provenance_fields() -> None:
    """Legacy compatibility projection should retain source schema and warning provenance."""
    payload = {
        "schema_version": 2,
        "query": "Database setup",
        "summary": {
            "strategy": "suppressed",
            "warning_codes": ["ungrounded_results_suppressed", "legacy_search_surface"],
            "entrypoint": "search_cli_v2",
            "contract_version": "contextpack_v2",
            "semantic_search_allowed": False,
            "search_integrity": {
                "vector_cache": {"status": "failed", "reason_codes": ["vector_metadata_mismatch"]},
                "chunk_span": {"status": "passed", "reason_codes": []},
            },
        },
        "hits": [],
    }

    projected = attach_legacy_search_contract(payload)

    metadata = projected["metadata"]
    assert metadata["source_schema_version"] == 2
    assert metadata["source_entrypoint"] == "search_cli_v2"
    assert metadata["warning_codes"] == [
        "ungrounded_results_suppressed",
        "legacy_search_surface",
    ]
    assert metadata["semantic_search_allowed"] is False
