from __future__ import annotations

import io

import pytest

import gloggur.indexer.concurrency as concurrency
from gloggur.io_failures import StorageIOError


def test_build_backoff_schedule_is_bounded() -> None:
    """Backoff schedule should stay within timeout budget."""
    policy = concurrency.LockRetryPolicy(
        timeout_ms=100,
        initial_backoff_ms=10,
        max_backoff_ms=40,
        multiplier=2.0,
    )

    delays = concurrency.build_backoff_schedule(policy)

    assert delays
    assert abs(sum(delays) - 0.1) <= 1e-6
    assert max(delays) <= 0.04


def test_acquire_lock_with_retry_succeeds_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock acquisition should retry and eventually succeed within budget."""
    policy = concurrency.LockRetryPolicy(
        timeout_ms=120,
        initial_backoff_ms=20,
        max_backoff_ms=20,
        multiplier=2.0,
    )
    attempts = [False, False, True]
    sleeps: list[float] = []

    def _fake_try(_handle: io.BytesIO) -> bool:
        return attempts.pop(0)

    monkeypatch.setattr(concurrency, "_try_acquire_lock", _fake_try)
    monkeypatch.setattr(concurrency.time, "sleep", lambda delay: sleeps.append(delay))

    attempt_count = concurrency._acquire_lock_with_retry(
        io.BytesIO(),
        "/tmp/cache.lock",
        policy,
    )

    assert attempt_count == 3
    assert sleeps == [0.02, 0.02]


def test_acquire_lock_with_retry_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock timeout should raise structured error with explicit lock operation."""
    policy = concurrency.LockRetryPolicy(
        timeout_ms=50,
        initial_backoff_ms=25,
        max_backoff_ms=25,
        multiplier=2.0,
    )
    monkeypatch.setattr(concurrency, "_try_acquire_lock", lambda _handle: False)
    monkeypatch.setattr(concurrency.time, "sleep", lambda _delay: None)
    monotonic_values = iter([0.0, 0.051])
    monkeypatch.setattr(concurrency.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(StorageIOError) as exc_info:
        concurrency._acquire_lock_with_retry(io.BytesIO(), "/tmp/cache.lock", policy)

    error = exc_info.value
    assert error.category == "unknown_io_error"
    assert error.operation == "acquire cache write lock"
    assert "timed out" in error.detail
