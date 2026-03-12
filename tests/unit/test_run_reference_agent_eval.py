from __future__ import annotations

import subprocess

import pytest

from scripts.run_reference_agent_eval import (
    _build_eval_summary,
    _expand_top_k,
    execute_reference_loop,
)


def test_execute_reference_loop_retries_once_then_succeeds() -> None:
    calls: list[tuple[str, int]] = []

    def search_json(query: str, top_k: int) -> dict[str, object]:
        calls.append((query, top_k))
        if len(calls) == 1:
            return {
                "schema_version": 2,
                "summary": {"strategy": "hybrid", "reason": "hits_missing"},
                "hits": [],
            }
        return {
            "schema_version": 2,
            "summary": {"strategy": "exact", "reason": "hits_present"},
            "hits": [
                {
                    "path": "math_ops.py",
                    "span": {"start_line": 1, "end_line": 1},
                    "snippet": "def add_numbers(a: int, b: int) -> int:",
                    "score": 0.9,
                    "tags": ["literal_match"],
                }
            ],
        }

    result = execute_reference_loop(
        query="add numbers token",
        top_k=2,
        max_retries=1,
        timeout_seconds=10.0,
        search_json=search_json,
    )

    assert result.ok is True
    assert result.status == "grounded"
    assert result.retry_performed is True
    assert result.attempts_used == 2
    assert calls[0] == ("add numbers token", 2)
    assert calls[1][1] == 4
    assert result.top_symbol == "add_numbers"
    assert result.top_symbol_id == "math_ops.py:1"
    assert [entry["step"] for entry in result.logs].count("decide") == 2


def test_execute_reference_loop_fails_closed_after_retry_budget_exhausted() -> None:
    def search_json(_query: str, _top_k: int) -> dict[str, object]:
        return {
            "schema_version": 2,
            "summary": {"strategy": "hybrid", "reason": "hits_missing"},
            "hits": [],
        }

    result = execute_reference_loop(
        query="missing token",
        top_k=3,
        max_retries=0,
        timeout_seconds=10.0,
        search_json=search_json,
    )

    assert result.ok is False
    assert result.status == "ungrounded"
    assert result.retry_performed is False
    assert result.failure is not None
    assert result.failure["code"] == "agent_grounding_failed"


def test_execute_reference_loop_fails_on_timeout_exception() -> None:
    def search_json(_query: str, _top_k: int) -> dict[str, object]:
        raise subprocess.TimeoutExpired(cmd=["gloggur", "search"], timeout=0.01)

    result = execute_reference_loop(
        query="token",
        top_k=3,
        max_retries=1,
        timeout_seconds=10.0,
        search_json=search_json,
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure["code"] == "agent_search_timeout"


def test_execute_reference_loop_fails_on_invalid_payload_schema() -> None:
    def search_json(_query: str, _top_k: int) -> dict[str, object]:
        return {"schema_version": 2, "summary": {"strategy": "exact"}}

    result = execute_reference_loop(
        query="token",
        top_k=3,
        max_retries=1,
        timeout_seconds=10.0,
        search_json=search_json,
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure["code"] == "agent_search_payload_invalid"


def test_build_eval_summary_is_deterministic_and_fail_closed_below_threshold() -> None:
    summary = _build_eval_summary(
        [
            {"case_id": "a", "passed": True},
            {"case_id": "b", "passed": False},
            {"case_id": "c", "passed": True},
        ],
        min_pass_rate=0.8,
    )

    assert summary == {
        "ok": False,
        "total_cases": 3,
        "passed": 2,
        "failed": 1,
        "pass_rate": 0.6667,
        "required_pass_rate": 0.8,
    }


def test_expand_top_k_is_deterministic_and_rejects_invalid_input() -> None:
    assert _expand_top_k(4) == 8
    assert _expand_top_k(63) == 64
    assert _expand_top_k(64) == 64
    with pytest.raises(ValueError, match="current_top_k must be >= 1"):
        _expand_top_k(0)
