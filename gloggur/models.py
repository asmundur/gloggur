from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Symbol(BaseModel):
    id: str
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    body_hash: str
    embedding_vector: Optional[List[float]] = None
    language: Optional[str] = None


class FileMetadata(BaseModel):
    path: str
    language: Optional[str] = None
    content_hash: str
    last_indexed: datetime = Field(default_factory=_utc_now)
    symbols: List[str] = Field(default_factory=list)


class IndexMetadata(BaseModel):
    version: str
    last_updated: datetime = Field(default_factory=_utc_now)
    total_symbols: int = 0
    indexed_files: int = 0
