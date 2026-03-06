from __future__ import annotations

import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


class RepoPathResolutionError(ValueError):
    """Raised when a repo-relative path is invalid or escapes the repo root."""


def discover_repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Return the nearest repo/workspace root for the provided start path."""
    candidate = Path(start or os.getcwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for current in (candidate, *candidate.parents):
        if (current / ".git").exists() or (current / ".gloggur").exists():
            return current
    return candidate


def is_path_absolute(raw_path: str) -> bool:
    """Return True for POSIX, UNC, or Windows-drive absolute paths."""
    if os.path.isabs(raw_path):
        return True
    return raw_path.startswith("\\\\") or bool(_WINDOWS_ABS_RE.match(raw_path))


def normalize_repo_relative_path(raw_path: str) -> str:
    """Return a deterministic POSIX-style relative path string."""
    normalized = raw_path.replace("\\", "/").strip()
    normalized = posixpath.normpath(normalized)
    if normalized == ".":
        return ""
    return normalized


def resolve_repo_relative_path(repo_root: Path, raw_path: str) -> Path:
    """Resolve a raw repo-relative path under repo_root or raise."""
    if not isinstance(raw_path, str):
        raise RepoPathResolutionError("path must be a string")
    if not raw_path.strip():
        raise RepoPathResolutionError("path must not be empty")
    if is_path_absolute(raw_path):
        raise RepoPathResolutionError("absolute paths are not allowed")

    normalized = normalize_repo_relative_path(raw_path)
    if not normalized or normalized == ".." or normalized.startswith("../"):
        raise RepoPathResolutionError("path escapes repo root")

    root = repo_root.resolve()
    candidate = (root / normalized.replace("/", os.sep)).resolve()
    try:
        within_root = os.path.commonpath([str(root), str(candidate)]) == str(root)
    except ValueError as exc:
        raise RepoPathResolutionError("path escapes repo root") from exc
    if not within_root:
        raise RepoPathResolutionError("path escapes repo root")
    return candidate


def to_repo_relative_path(repo_root: Path, path: str) -> str:
    """Normalize an absolute or relative path into a repo-relative POSIX path."""
    if not path.strip():
        return path
    root = repo_root.resolve()
    if is_path_absolute(path):
        candidate = Path(path).resolve()
    else:
        candidate = (root / path).resolve()
    try:
        within_root = os.path.commonpath([str(root), str(candidate)]) == str(root)
    except ValueError:
        within_root = False
    if within_root:
        relative = os.path.relpath(str(candidate), str(root))
        return normalize_repo_relative_path(relative)
    return normalize_repo_relative_path(path)


@dataclass(frozen=True)
class LineByteSpanIndex:
    """Map 1-based logical line numbers to byte spans in the original file bytes."""

    raw_bytes: bytes
    line_starts: tuple[int, ...]
    line_ends: tuple[int, ...]

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> LineByteSpanIndex:
        """Build a deterministic line-to-byte index from raw file bytes."""
        starts: list[int] = []
        ends: list[int] = []
        current_start = 0
        for index, value in enumerate(raw_bytes):
            if value == 0x0A:
                starts.append(current_start)
                ends.append(index + 1)
                current_start = index + 1
        if current_start < len(raw_bytes):
            starts.append(current_start)
            ends.append(len(raw_bytes))
        return cls(
            raw_bytes=raw_bytes,
            line_starts=tuple(starts),
            line_ends=tuple(ends),
        )

    @property
    def line_count(self) -> int:
        return len(self.line_starts)

    @property
    def total_bytes(self) -> int:
        return len(self.raw_bytes)

    def span_for_lines(self, start_line: int, end_line: int) -> tuple[int, int]:
        """Return the byte span for an inclusive logical line range."""
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid line range")
        if self.line_count == 0:
            if start_line == 1 and end_line == 1:
                return (0, 0)
            raise ValueError("line range out of bounds")
        if end_line > self.line_count:
            raise ValueError("line range out of bounds")
        return (
            self.line_starts[start_line - 1],
            self.line_ends[end_line - 1],
        )

    def extract_bytes(self, start_byte: int, end_byte: int) -> bytes:
        """Return the exact raw-byte slice for a validated span."""
        if start_byte < 0 or end_byte < start_byte:
            raise ValueError("invalid byte range")
        if end_byte > self.total_bytes:
            raise IndexError("byte range out of bounds")
        return self.raw_bytes[start_byte:end_byte]

    def extract_text(self, start_byte: int, end_byte: int) -> str:
        """Decode an extracted byte slice with UTF-8 replacement."""
        return self.extract_bytes(start_byte, end_byte).decode("utf8", errors="replace")
