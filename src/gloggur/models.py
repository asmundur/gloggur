from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC time for metadata defaults."""
    return datetime.now(timezone.utc)


class Symbol(BaseModel):
    """Symbol metadata: name, kind, file path, lines, signature, docstring."""

    id: str
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    body_hash: str
    embedding_vector: list[float] | None = None
    language: str | None = None
    invariants: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    covered_by: list[str] = Field(default_factory=list)
    is_serialization_boundary: bool = False
    implicit_contract: str | None = None


class FileMetadata(BaseModel):
    """Index metadata for a file, including content hash and symbol ids."""

    path: str
    language: str | None = None
    content_hash: str
    last_indexed: datetime = Field(default_factory=_utc_now)
    symbols: list[str] = Field(default_factory=list)


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
