from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SymbolOccurrence:
    """One symbol index row for a definition/reference occurrence."""

    symbol: str
    kind: str
    path: str
    line: int
    language: str | None = None
    container: str | None = None
    signature: str | None = None


@dataclass(frozen=True)
class IndexedFile:
    """Incremental index metadata for one file."""

    path: str
    content_hash: str
    mtime_ns: int
    language: str | None = None
    last_indexed: datetime = field(default_factory=_utc_now)


@dataclass
class SymbolIndexResult:
    """Result counters emitted by symbol-index indexing runs."""

    db_path: str
    files_considered: int = 0
    files_changed: int = 0
    files_unchanged: int = 0
    defs_indexed: int = 0
    refs_indexed: int = 0
    files_removed: int = 0
    failed: int = 0
    failed_reasons: dict[str, int] = field(default_factory=dict)
    failed_samples: list[str] = field(default_factory=list)

    def add_failure(self, reason: str, sample: str | None = None) -> None:
        normalized_reason = reason.strip() if reason.strip() else "symbol_index_error"
        self.failed += 1
        self.failed_reasons[normalized_reason] = self.failed_reasons.get(normalized_reason, 0) + 1
        if sample and len(self.failed_samples) < 5:
            self.failed_samples.append(sample)

    def as_payload(self) -> dict[str, object]:
        return {
            "db_path": self.db_path,
            "files_considered": self.files_considered,
            "files_changed": self.files_changed,
            "files_unchanged": self.files_unchanged,
            "defs_indexed": self.defs_indexed,
            "refs_indexed": self.refs_indexed,
            "files_removed": self.files_removed,
            "failed": self.failed,
            "failed_reasons": dict(sorted(self.failed_reasons.items())),
            "failed_samples": list(self.failed_samples),
        }
