from __future__ import annotations

import errno
import sqlite3

import pytest

from gloggur.io_failures import classify_io_error, format_io_error_message, wrap_io_error


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (PermissionError(errno.EACCES, "permission denied"), "permission_denied"),
        (OSError(errno.EROFS, "read-only filesystem"), "read_only_filesystem"),
        (OSError(errno.ENOSPC, "no space left on device"), "disk_full_or_quota"),
        (OSError(errno.ENOENT, "no such file or directory"), "path_not_writable"),
        (OSError(errno.EIO, "i/o error"), "unknown_io_error"),
    ],
)
def test_classify_io_error_os_cases(exc: OSError, expected: str) -> None:
    """Classify common OS errno values into stable category labels."""
    assert classify_io_error(exc) == expected


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            sqlite3.OperationalError("attempt to write a readonly database"),
            "read_only_filesystem",
        ),
        (sqlite3.OperationalError("database or disk is full"), "disk_full_or_quota"),
        (sqlite3.OperationalError("unable to open database file"), "path_not_writable"),
        (sqlite3.OperationalError("permission denied"), "permission_denied"),
        (sqlite3.OperationalError("disk I/O error"), "unknown_io_error"),
        (
            sqlite3.DatabaseError("database disk image is malformed"),
            "unknown_io_error",
        ),
        (
            sqlite3.DatabaseError("attempt to write a readonly database"),
            "read_only_filesystem",
        ),
    ],
)
def test_classify_io_error_sqlite_cases(
    exc: sqlite3.DatabaseError,
    expected: str,
) -> None:
    """Classify sqlite operational/database failures into stable category labels."""
    assert classify_io_error(exc) == expected


def test_wrap_io_error_includes_actionable_metadata() -> None:
    """Wrapped errors include operation/path/detail/probable-cause/remediation fields."""
    wrapped = wrap_io_error(
        sqlite3.OperationalError("database or disk is full"),
        operation="execute cache database transaction",
        path="/tmp/gloggur-cache/index.db",
    )
    payload = wrapped.to_payload()
    assert payload["error"]["category"] == "disk_full_or_quota"
    assert payload["error"]["operation"] == "execute cache database transaction"
    assert payload["error"]["path"] == "/tmp/gloggur-cache/index.db"
    assert "OperationalError: database or disk is full" in str(payload["error"]["detail"])
    assert len(payload["error"]["remediation"]) >= 2

    message = format_io_error_message(wrapped)
    assert "Probable cause:" in message
    assert "Remediation:" in message
