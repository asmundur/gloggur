from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC time for metadata defaults."""
    return datetime.now(timezone.utc)


class Signal(BaseModel):
    """Normalized parser/analysis signal attached to a symbol."""

    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    confidence: float | None = None


class Symbol(BaseModel):
    """Symbol metadata: name, kind, file path, lines, signature, docstring."""

    id: str
    name: str
    kind: str
    fqname: str | None = None
    file_path: str
    start_line: int
    end_line: int
    container_id: str | None = None
    container_fqname: str | None = None
    signature: str | None = None
    docstring: str | None = None
    body_hash: str
    embedding_vector: list[float] | None = None
    language: str | None = None
    repo_id: str | None = None
    commit: str | None = None
    visibility: str | None = None
    exported: bool | None = None
    tokens_estimate: int | None = None
    invariants: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    covered_by: list[str] = Field(default_factory=list)
    is_serialization_boundary: bool = False
    implicit_contract: str | None = None
    signals: list[Signal] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class FileMetadata(BaseModel):
    """Index metadata for a file, including content hash and symbol ids."""

    path: str
    language: str | None = None
    content_hash: str
    last_indexed: datetime = Field(default_factory=_utc_now)
    symbols: list[str] = Field(default_factory=list)


class SymbolChunk(BaseModel):
    """One embedding chunk belonging to a symbol."""

    chunk_id: str
    symbol_id: str
    chunk_part_index: int
    chunk_part_total: int
    text: str
    file_path: str
    start_line: int
    end_line: int
    tokens_estimate: int | None = None
    language: str | None = None
    repo_id: str | None = None
    commit: str | None = None
    embedding_vector: list[float] | None = None


class EdgeRecord(BaseModel):
    """One reference-graph edge record used for graph queries and retrieval."""

    edge_id: str
    edge_type: str
    from_id: str
    to_id: str
    from_kind: str
    to_kind: str
    file_path: str
    line: int
    confidence: float
    repo_id: str | None = None
    commit: str | None = None
    text: str | None = None
    embedding_vector: list[float] | None = None


class IndexMetadata(BaseModel):
    """Aggregate metadata for an index run (counts, version, timestamp)."""

    version: str
    last_updated: datetime = Field(default_factory=_utc_now)
    total_symbols: int = 0
    indexed_files: int = 0


class AuditFileMetadata(BaseModel):
    """Audit metadata for a file (hash and last audited time)."""

    path: str
    content_hash: str
    last_audited: datetime = Field(default_factory=_utc_now)
