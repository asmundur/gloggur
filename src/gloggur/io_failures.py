from __future__ import annotations

import errno
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

IO_ERROR_CATEGORIES = {
    "permission_denied",
    "read_only_filesystem",
    "disk_full_or_quota",
    "path_not_writable",
    "unknown_io_error",
}

_CATEGORY_DETAILS: Dict[str, Tuple[str, List[str]]] = {
    "permission_denied": (
        "The process does not have permission for this filesystem operation.",
        [
            "Check path ownership and filesystem permissions.",
            "Run with a user that can read/write the cache directory.",
        ],
    ),
    "read_only_filesystem": (
        "The target path is on a read-only filesystem.",
        [
            "Use a writable cache directory (set GLOGGUR_CACHE_DIR).",
            "If appropriate, remount the filesystem as read-write.",
        ],
    ),
    "disk_full_or_quota": (
        "The filesystem is out of space or quota for writes.",
        [
            "Free disk space or increase quota.",
            "Retry the command after capacity is restored.",
        ],
    ),
    "path_not_writable": (
        "The target path or one of its parent directories is not writable/usable.",
        [
            "Ensure the parent directory exists and is writable.",
            "Set GLOGGUR_CACHE_DIR to a valid writable path.",
        ],
    ),
    "unknown_io_error": (
        "An unclassified filesystem or database I/O failure occurred.",
        [
            "Inspect the original error detail for OS-specific context.",
            "Retry with a known-good writable cache directory.",
        ],
    ),
}

_PERMISSION_ERRNOS = {errno.EACCES, errno.EPERM}
_READ_ONLY_ERRNOS = {errno.EROFS}
_DISK_ERRNOS = {errno.ENOSPC}
_PATH_ERRNOS = {
    errno.ENOENT,
    errno.ENOTDIR,
    errno.EISDIR,
    errno.ENAMETOOLONG,
    errno.ELOOP,
}
if hasattr(errno, "EDQUOT"):
    _DISK_ERRNOS.add(errno.EDQUOT)


@dataclass(frozen=True)
class StorageIOError(RuntimeError):
    """Structured I/O failure for deterministic user/CI handling."""

    category: str
    operation: str
    path: str
    probable_cause: str
    remediation: List[str]
    detail: str

    def __post_init__(self) -> None:
        if self.category not in IO_ERROR_CATEGORIES:
            raise ValueError(f"unsupported io error category: {self.category}")

    def __str__(self) -> str:
        return (
            f"IO failure [{self.category}] during '{self.operation}' at {self.path}: "
            f"{self.detail}"
        )

    def to_payload(self) -> dict[str, object]:
        """Return machine-readable error payload for JSON outputs."""
        return {
            "error": {
                "type": "io_failure",
                "category": self.category,
                "operation": self.operation,
                "path": self.path,
                "probable_cause": self.probable_cause,
                "remediation": self.remediation,
                "detail": self.detail,
            }
        }


def wrap_io_error(exc: Exception, *, operation: str, path: str) -> StorageIOError:
    """Convert an OSError/sqlite exception into a structured StorageIOError."""
    if isinstance(exc, StorageIOError):
        return exc
    category = classify_io_error(exc)
    probable_cause, remediation = _CATEGORY_DETAILS[category]
    detail = f"{type(exc).__name__}: {exc}"
    return StorageIOError(
        category=category,
        operation=operation,
        path=path,
        probable_cause=probable_cause,
        remediation=list(remediation),
        detail=detail,
    )


def classify_io_error(exc: Exception) -> str:
    """Classify an exception into one of the stable IO error categories."""
    if isinstance(exc, OSError):
        return _classify_os_error(exc)
    if isinstance(exc, sqlite3.OperationalError):
        return _classify_sqlite_operational_error(exc)
    return "unknown_io_error"


def _classify_os_error(exc: OSError) -> str:
    if isinstance(exc, PermissionError) or exc.errno in _PERMISSION_ERRNOS:
        return "permission_denied"
    if exc.errno in _READ_ONLY_ERRNOS:
        return "read_only_filesystem"
    if exc.errno in _DISK_ERRNOS:
        return "disk_full_or_quota"
    if exc.errno in _PATH_ERRNOS:
        return "path_not_writable"
    return "unknown_io_error"


def _classify_sqlite_operational_error(exc: sqlite3.OperationalError) -> str:
    detail = str(exc).lower()
    if any(token in detail for token in ("readonly", "read-only")):
        return "read_only_filesystem"
    if "permission denied" in detail:
        return "permission_denied"
    if any(token in detail for token in ("disk is full", "database or disk is full", "quota")):
        return "disk_full_or_quota"
    if "unable to open database file" in detail:
        return "path_not_writable"
    if "i/o error" in detail:
        return "unknown_io_error"
    return "unknown_io_error"


def format_io_error_message(error: StorageIOError) -> str:
    """Create stable human-readable stderr output for I/O failures."""
    lines = [
        str(error),
        f"Probable cause: {error.probable_cause}",
        "Remediation:",
    ]
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(error.remediation, start=1))
    return "\n".join(lines)
