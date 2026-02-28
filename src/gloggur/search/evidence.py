from __future__ import annotations

from typing import Dict, List, Optional


DEFAULT_EVIDENCE_VALIDATOR = "min_confident_evidence_v1"


def build_evidence_trace(results: object) -> List[Dict[str, object]]:
    """Build normalized evidence trace payload from search results."""
    if not isinstance(results, list):
        raise ValueError("search payload 'results' must be a list")
    trace: List[Dict[str, object]] = []
    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"search result at index {rank - 1} is not an object")
        symbol_id = item.get("symbol_id")
        if not isinstance(symbol_id, str) or not symbol_id.strip():
            raise ValueError(f"search result at index {rank - 1} has missing/invalid symbol_id")
        symbol = item.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"search result at index {rank - 1} has missing/invalid symbol")
        file_path = item.get("file")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError(f"search result at index {rank - 1} has missing/invalid file")
        line_raw = item.get("line")
        try:
            line_start = int(line_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"search result at index {rank - 1} has non-numeric line") from exc
        if line_start < 1:
            raise ValueError(f"search result at index {rank - 1} has invalid line (<1)")
        line_end_raw = item.get("line_end", line_start)
        try:
            line_end = int(line_end_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"search result at index {rank - 1} has non-numeric line_end") from exc
        if line_end < line_start:
            raise ValueError(f"search result at index {rank - 1} has line_end before line")
        score_raw = item.get("similarity_score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"search result at index {rank - 1} has non-numeric similarity_score") from exc
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        trace.append(
            {
                "rank": rank,
                "symbol_id": symbol_id,
                "symbol": symbol,
                "file": file_path,
                "line_span": {"start": line_start, "end": line_end},
                "confidence_contribution": score,
            }
        )
    return trace


def validate_evidence_trace(
    evidence_trace: object,
    *,
    min_confidence: float,
    min_items: int,
) -> Dict[str, object]:
    """Validate grounding quality over normalized evidence trace."""
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    if min_items < 1:
        raise ValueError("min_items must be >= 1")
    if not isinstance(evidence_trace, list):
        raise ValueError("evidence_trace must be a list")

    matched_symbol_ids: List[str] = []
    for idx, item in enumerate(evidence_trace):
        if not isinstance(item, dict):
            raise ValueError(f"evidence trace item at index {idx} is not an object")
        symbol_id = item.get("symbol_id")
        if not isinstance(symbol_id, str) or not symbol_id:
            raise ValueError(f"evidence trace item at index {idx} has missing/invalid symbol_id")
        score_raw = item.get("confidence_contribution")
        try:
            score = float(score_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"evidence trace item at index {idx} has non-numeric confidence_contribution"
            ) from exc
        if score >= min_confidence:
            matched_symbol_ids.append(symbol_id)

    matched_items = len(matched_symbol_ids)
    passed = matched_items >= min_items
    reason_code = "grounding_sufficient"
    reason = "Evidence meets grounding threshold."
    suggested_repair_action: Optional[str] = None
    if not passed:
        if len(evidence_trace) == 0:
            reason_code = "grounding_evidence_missing"
            reason = "No evidence items were available for grounding validation."
            suggested_repair_action = (
                "Retry retrieval with a broader query or larger --top-k, then re-run validation."
            )
        else:
            reason_code = "grounding_confidence_below_threshold"
            reason = "Evidence exists, but not enough items met the confidence threshold."
            suggested_repair_action = (
                "Retry retrieval with broader terms or lower threshold explicitly if policy allows."
            )
    return {
        "validator": DEFAULT_EVIDENCE_VALIDATOR,
        "passed": passed,
        "reason_code": reason_code,
        "reason": reason,
        "min_confidence": min_confidence,
        "min_items": min_items,
        "matched_items": matched_items,
        "matched_symbol_ids": matched_symbol_ids,
        "suggested_repair_action": suggested_repair_action,
    }
