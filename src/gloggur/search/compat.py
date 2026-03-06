from __future__ import annotations

from typing import Any


def attach_legacy_search_contract(payload: dict[str, object]) -> dict[str, object]:
    """Attach legacy `results`/`metadata` views for internal migrations.

    CLI output remains ContextPack v2; this adapter is for tests/harnesses that still
    assert legacy invariants while contracts are remapped.
    """
    if payload.get("schema_version") != 2:
        return payload
    if "results" in payload and "metadata" in payload:
        return payload

    hits = payload.get("hits")
    if not isinstance(hits, list):
        return payload

    debug_payload = payload.get("debug")
    summary_payload = payload.get("summary")
    debug = debug_payload if isinstance(debug_payload, dict) else {}
    summary = summary_payload if isinstance(summary_payload, dict) else {}

    timings = debug.get("timings")
    timing_map = timings if isinstance(timings, dict) else {}
    total_ms = int(timing_map.get("total_ms", 0) or 0)

    results: list[dict[str, object]] = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        span = item.get("span")
        span_dict = span if isinstance(span, dict) else {}
        results.append(
            {
                "file": item.get("path"),
                "line": span_dict.get("start_line"),
                "line_end": span_dict.get("end_line"),
                "start_byte": item.get("start_byte"),
                "end_byte": item.get("end_byte"),
                "context": item.get("snippet"),
                "similarity_score": item.get("score"),
                "ranking_score": item.get("score"),
                "tags": item.get("tags") if isinstance(item.get("tags"), list) else [],
            }
        )

    top_score = 0.0
    if results:
        raw_top = results[0].get("ranking_score")
        if isinstance(raw_top, (int, float)):
            top_score = max(0.0, min(1.0, float(raw_top)))

    confidence_threshold = summary.get("legacy_confidence_threshold")
    if isinstance(confidence_threshold, (int, float)):
        threshold = float(confidence_threshold)
    else:
        threshold = 0.0

    initial_top_k = summary.get("legacy_top_k")
    initial_top_k_value = int(initial_top_k) if isinstance(initial_top_k, int) else len(results)

    metadata: dict[str, Any] = {
        "total_results": len(results),
        "search_time_ms": total_ms,
        "needs_reindex": bool(summary.get("needs_reindex", False)),
        "reindex_reason": summary.get("reindex_reason"),
        "entrypoint": summary.get("entrypoint"),
        "contract_version": summary.get("contract_version"),
        "source_schema_version": payload.get("schema_version"),
        "source_entrypoint": summary.get("entrypoint"),
        "warning_codes": (
            list(summary.get("warning_codes", []))
            if isinstance(summary.get("warning_codes"), list)
            else []
        ),
        "file_filter": summary.get("file_filter"),
        "file_filter_match_mode": "exact_or_prefix" if summary.get("file_filter") else None,
        "file_filter_warning_codes": (
            ["file_filter_no_match"] if summary.get("file_filter") and not results else []
        ),
        "context_radius": summary.get("legacy_context_radius"),
        "ranking_mode": summary.get("legacy_ranking_mode"),
        "initial_confidence": top_score,
        "final_confidence": top_score,
        "retry_enabled": bool(summary.get("legacy_retry_enabled", True)),
        "retry_performed": False,
        "retry_attempts": 0,
        "retry_strategy": "router",
        "initial_top_k": initial_top_k_value,
        "final_top_k": initial_top_k_value,
        "max_requery_attempts": summary.get("legacy_max_requery_attempts"),
        "low_confidence": bool(results == [] or (threshold > 0.0 and top_score < threshold)),
        "grounding_validation_enabled": False,
        "grounding_validation_passed": None,
        "query_intent": summary.get("strategy", "auto"),
        "explicit_test_intent": False,
        "test_penalty_applied": False,
        "tool_version": summary.get("tool_version"),
        "semantic_search_allowed": summary.get("semantic_search_allowed"),
        "search_integrity": summary.get("search_integrity"),
    }

    # Resume-contract fields are flattened into search summary in v2 for compatibility.
    resume_fields = {
        "resume_decision",
        "resume_reason_codes",
        "resume_remediation",
        "resume_fingerprint_match",
        "expected_resume_fingerprint",
        "last_success_resume_fingerprint",
        "last_success_resume_fingerprint_match",
        "last_success_resume_at",
        "last_success_tool_version",
        "last_success_tool_version_match",
        "allow_tool_version_drift",
        "tool_version_drift_detected",
        "tool_version_drift_override_applied",
    }
    for field in resume_fields:
        if field in summary:
            metadata[field] = summary[field]

    resume_contract = debug.get("resume_contract")
    if isinstance(resume_contract, dict):
        for field in resume_fields:
            if field in resume_contract and field not in metadata:
                metadata[field] = resume_contract[field]

    payload["results"] = results
    payload["metadata"] = metadata
    return payload
