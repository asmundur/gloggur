from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Iterator, Optional

from gloggur.io_failures import StorageIOError, wrap_io_error

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is unavailable on Windows.
    fcntl = None  # type: ignore[assignment]


DEFAULT_LOCK_TIMEOUT_MS = 5_000
DEFAULT_INITIAL_BACKOFF_MS = 25
DEFAULT_MAX_BACKOFF_MS = 250
DEFAULT_BACKOFF_MULTIPLIER = 2.0
LOCK_FILE_NAME = ".cache-write.lock"


@dataclass(frozen=True)
class LockRetryPolicy:
    """Retry settings for cross-process cache write lock acquisition."""

    timeout_ms: int = DEFAULT_LOCK_TIMEOUT_MS
    initial_backoff_ms: int = DEFAULT_INITIAL_BACKOFF_MS
    max_backoff_ms: int = DEFAULT_MAX_BACKOFF_MS
    multiplier: float = DEFAULT_BACKOFF_MULTIPLIER

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        if self.initial_backoff_ms <= 0:
            raise ValueError("initial_backoff_ms must be > 0")
        if self.max_backoff_ms <= 0:
            raise ValueError("max_backoff_ms must be > 0")
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be >= 1.0")

    @classmethod
    def from_env(cls) -> "LockRetryPolicy":
        """Create retry policy from environment variables with safe fallbacks."""

        return cls(
            timeout_ms=_read_int_env("GLOGGUR_CACHE_LOCK_TIMEOUT_MS", DEFAULT_LOCK_TIMEOUT_MS),
            initial_backoff_ms=_read_int_env(
                "GLOGGUR_CACHE_LOCK_INITIAL_BACKOFF_MS",
                DEFAULT_INITIAL_BACKOFF_MS,
            ),
            max_backoff_ms=_read_int_env(
                "GLOGGUR_CACHE_LOCK_MAX_BACKOFF_MS",
                DEFAULT_MAX_BACKOFF_MS,
            ),
            multiplier=_read_float_env(
                "GLOGGUR_CACHE_LOCK_BACKOFF_MULTIPLIER",
                DEFAULT_BACKOFF_MULTIPLIER,
            ),
        )


def build_backoff_schedule(policy: LockRetryPolicy) -> list[float]:
    """Return bounded retry delays (seconds) whose total does not exceed timeout."""

    delays: list[float] = []
    remaining_ms = policy.timeout_ms
    next_delay_ms = policy.initial_backoff_ms
    while remaining_ms > 0:
        delay_ms = min(next_delay_ms, remaining_ms)
        delays.append(delay_ms / 1000.0)
        remaining_ms -= delay_ms
        next_delay_ms = min(
            int(round(next_delay_ms * policy.multiplier)),
            policy.max_backoff_ms,
        )
    return delays


@contextmanager
def cache_write_lock(
    cache_dir: str,
    *,
    policy: Optional[LockRetryPolicy] = None,
) -> Iterator[None]:
    """Acquire a cross-process cache writer lock with bounded retry/timeout."""

    active_policy = policy or LockRetryPolicy.from_env()
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        raise wrap_io_error(exc, operation="create cache directory", path=cache_dir) from exc

    lock_path = os.path.join(cache_dir, LOCK_FILE_NAME)
    try:
        handle = open(lock_path, "a+b")
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="open cache write lock file",
            path=lock_path,
        ) from exc

    try:
        attempts = _acquire_lock_with_retry(handle, lock_path, active_policy)
        _ = attempts
        try:
            yield
        finally:
            _release_lock(handle)
    finally:
        handle.close()


def _acquire_lock_with_retry(
    handle: IO[bytes],
    lock_path: str,
    policy: LockRetryPolicy,
) -> int:
    """Acquire lock, retrying with bounded backoff; return number of attempts."""

    start = time.monotonic()
    attempts = 1
    if _try_acquire_lock(handle):
        return attempts

    for delay in build_backoff_schedule(policy):
        time.sleep(delay)
        attempts += 1
        if _try_acquire_lock(handle):
            return attempts

    waited_ms = int((time.monotonic() - start) * 1000)
    raise _lock_timeout_error(lock_path, waited_ms, attempts)


def _try_acquire_lock(handle: IO[bytes]) -> bool:
    """Try to acquire the lock in non-blocking mode."""

    if fcntl is None:
        return True
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _release_lock(handle: IO[bytes]) -> None:
    """Release an acquired lock."""

    if fcntl is None:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_timeout_error(lock_path: str, waited_ms: int, attempts: int) -> StorageIOError:
    """Build a stable lock-timeout error payload for CLI/CI consumers."""

    return StorageIOError(
        category="unknown_io_error",
        operation="acquire cache write lock",
        path=lock_path,
        probable_cause="Another gloggur process is updating the cache.",
        remediation=[
            "Wait for active `index`, `watch`, or `clear-cache` operations to finish.",
            "Retry the command with default settings.",
            "Increase GLOGGUR_CACHE_LOCK_TIMEOUT_MS for longer waits if contention is expected.",
        ],
        detail=(
            "cache write lock timed out "
            f"after {waited_ms}ms ({attempts} attempts)"
        ),
    )


def _read_int_env(name: str, default: int) -> int:
    """Parse positive integer env var or return default."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_float_env(name: str, default: float) -> float:
    """Parse float env var or return default."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 1.0 else default
