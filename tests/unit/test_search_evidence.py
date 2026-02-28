from __future__ import annotations

import pytest

from gloggur.search.evidence import build_evidence_trace, validate_evidence_trace


def test_build_evidence_trace_normalizes_result_schema() -> None:
    """Evidence trace should normalize symbol/result metadata into stable schema."""
    trace = build_evidence_trace(
        [
            {
                "symbol_id": "sample.py:10:alpha",
                "symbol": "alpha",
                "file": "sample.py",
                "line": 10,
                "line_end": 12,
                "similarity_score": 0.8,
            },
            {
                "symbol_id": "sample.py:20:beta",
                "symbol": "beta",
                "file": "sample.py",
                "line": 20,
                "similarity_score": 1.2,
            },
        ]
    )
    assert len(trace) == 2
    first = trace[0]
    assert first["rank"] == 1
    assert first["symbol_id"] == "sample.py:10:alpha"
    assert first["line_span"] == {"start": 10, "end": 12}
    assert first["confidence_contribution"] == pytest.approx(0.8)
    second = trace[1]
    assert second["rank"] == 2
    assert second["line_span"] == {"start": 20, "end": 20}
    # clamped to [0, 1]
    assert second["confidence_contribution"] == pytest.approx(1.0)


def test_build_evidence_trace_rejects_missing_symbol_id() -> None:
    """Evidence trace generation should fail loudly on malformed result payloads."""
    with pytest.raises(ValueError, match="missing/invalid symbol_id"):
        build_evidence_trace(
            [
                {
                    "symbol": "alpha",
                    "file": "sample.py",
                    "line": 10,
                    "similarity_score": 0.8,
                }
            ]
        )


def test_validate_evidence_trace_passes_when_threshold_met() -> None:
    """Validator should pass when enough evidence items meet threshold."""
    trace = [
        {
            "rank": 1,
            "symbol_id": "sample.py:10:alpha",
            "symbol": "alpha",
            "file": "sample.py",
            "line_span": {"start": 10, "end": 10},
            "confidence_contribution": 0.9,
        }
    ]
    payload = validate_evidence_trace(trace, min_confidence=0.7, min_items=1)
    assert payload["passed"] is True
    assert payload["reason_code"] == "grounding_sufficient"
    assert payload["matched_items"] == 1
    assert payload["suggested_repair_action"] is None


def test_validate_evidence_trace_fails_when_confidence_too_low() -> None:
    """Validator should fail with deterministic reason code when evidence is low quality."""
    trace = [
        {
            "rank": 1,
            "symbol_id": "sample.py:10:alpha",
            "symbol": "alpha",
            "file": "sample.py",
            "line_span": {"start": 10, "end": 10},
            "confidence_contribution": 0.2,
        }
    ]
    payload = validate_evidence_trace(trace, min_confidence=0.7, min_items=1)
    assert payload["passed"] is False
    assert payload["reason_code"] == "grounding_confidence_below_threshold"
    assert payload["matched_items"] == 0
    assert isinstance(payload["suggested_repair_action"], str)


def test_validate_evidence_trace_fails_when_no_evidence() -> None:
    """Empty trace should fail with explicit missing-evidence code."""
    payload = validate_evidence_trace([], min_confidence=0.7, min_items=1)
    assert payload["passed"] is False
    assert payload["reason_code"] == "grounding_evidence_missing"
    assert payload["matched_items"] == 0
    assert isinstance(payload["suggested_repair_action"], str)

