from __future__ import annotations

from dataclasses import dataclass

from gloggur.byte_spans import LineByteSpanIndex
from gloggur.models import Symbol


@dataclass
class ParsedFileSnapshot:
    """Shared parsed-file payload reused across index phases in one CLI command."""

    path: str
    source: str
    content_hash: str
    mtime_ns: int | None
    language: str | None
    symbols: list[Symbol]
    span_index: LineByteSpanIndex


@dataclass
class FileTimingTrace:
    """Per-file timing breakdown surfaced by index --debug-timings."""

    path: str
    status: str
    parse_ms: int = 0
    edge_ms: int = 0
    embed_ms: int = 0
    persist_ms: int = 0
    symbol_count: int = 0
    chunk_count: int = 0
    reason: str | None = None
    detail: str | None = None

    @property
    def total_ms(self) -> int:
        return self.parse_ms + self.edge_ms + self.embed_ms + self.persist_ms

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path,
            "status": self.status,
            "total_ms": self.total_ms,
            "parse_ms": self.parse_ms,
            "edge_ms": self.edge_ms,
            "embed_ms": self.embed_ms,
            "persist_ms": self.persist_ms,
            "symbol_count": self.symbol_count,
            "chunk_count": self.chunk_count,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.detail:
            payload["detail"] = self.detail
        return payload
