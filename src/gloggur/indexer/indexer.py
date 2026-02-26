from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import wrap_embedding_error
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.models import FileMetadata, IndexMetadata, Symbol
from gloggur.parsers.registry import ParserRegistry
from gloggur.storage.vector_store import VectorStore


@dataclass
class IndexResult:
    """Dataclass for indexing results with explicit terminal file outcomes."""

    files_considered: int
    indexed: int
    unchanged: int
    failed: int
    indexed_symbols: int
    duration_ms: int
    failed_reasons: Dict[str, int] = field(default_factory=dict)
    failed_samples: List[str] = field(default_factory=list)

    @property
    def indexed_files(self) -> int:
        """Backward-compatible alias for indexed files."""

        return self.indexed

    @property
    def skipped_files(self) -> int:
        """Backward-compatible alias for unchanged files."""

        return self.unchanged

    def as_payload(self) -> Dict[str, object]:
        """Build a CLI payload with explicit outcomes and legacy aliases."""

        return {
            "files_considered": self.files_considered,
            "indexed": self.indexed,
            "unchanged": self.unchanged,
            "failed": self.failed,
            "failed_reasons": dict(self.failed_reasons),
            "failed_samples": list(self.failed_samples),
            "indexed_symbols": self.indexed_symbols,
            "duration_ms": self.duration_ms,
            "indexed_files": self.indexed_files,
            "skipped_files": self.skipped_files,
        }


@dataclass(frozen=True)
class FileIndexOutcome:
    """Per-file terminal outcome used to classify index runs."""

    status: str
    symbols_indexed: int = 0
    reason: Optional[str] = None
    detail: Optional[str] = None


class Indexer:
    """Incremental repository indexer that hashes files and stores symbols."""

    def __init__(
        self,
        config: GloggurConfig,
        cache: Optional[CacheManager] = None,
        parser_registry: Optional[ParserRegistry] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        """Initialize the indexer with cache, parsers, and embeddings."""

        self.config = config
        self.cache = cache or CacheManager(CacheConfig(config.cache_dir))
        self.parser_registry = parser_registry or ParserRegistry()
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index_repository(self, path: str) -> IndexResult:
        """Index all supported files under a repository root."""

        start = time.time()
        self.cache.delete_index_metadata()
        # Test-only hook to make interruption windows deterministic.
        self._maybe_pause_after_metadata_delete()

        files_considered = 0
        indexed_files = 0
        unchanged_files = 0
        failed_files = 0
        indexed_symbols = 0
        failed_reasons: Dict[str, int] = {}
        failed_samples: List[str] = []

        for file_path in self._iter_source_files(path):
            files_considered += 1
            outcome = self.index_file_with_outcome(file_path)
            if outcome.status == "indexed":
                indexed_files += 1
                indexed_symbols += outcome.symbols_indexed
                continue
            if outcome.status == "unchanged":
                unchanged_files += 1
                continue

            failed_files += 1
            reason = outcome.reason or "indexing_error"
            failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
            if len(failed_samples) < 5:
                if outcome.detail:
                    failed_samples.append(f"{file_path}: {outcome.detail}")
                else:
                    failed_samples.append(file_path)

        if self.vector_store:
            self.vector_store.save()

        if failed_files == 0:
            metadata = IndexMetadata(
                version=self.config.index_version,
                total_symbols=len(self.cache.list_symbols()),
                indexed_files=self.cache.count_files(),
            )
            self.cache.set_index_metadata(metadata)
            self.cache.set_index_profile(self.config.embedding_profile())

        duration_ms = int((time.time() - start) * 1000)
        return IndexResult(
            files_considered=files_considered,
            indexed=indexed_files,
            unchanged=unchanged_files,
            failed=failed_files,
            indexed_symbols=indexed_symbols,
            duration_ms=duration_ms,
            failed_reasons=failed_reasons,
            failed_samples=failed_samples,
        )

    @staticmethod
    def _maybe_pause_after_metadata_delete() -> None:
        """Pause after metadata invalidation when enabled for integration tests."""

        raw_value = os.getenv("GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_MS")
        if not raw_value:
            return
        try:
            pause_ms = int(raw_value)
        except ValueError:
            return
        if pause_ms <= 0:
            return
        time.sleep(pause_ms / 1000.0)

    def index_file_with_outcome(self, path: str) -> FileIndexOutcome:
        """Index a file and return an explicit terminal outcome."""

        try:
            with open(path, "r", encoding="utf8") as handle:
                source = handle.read()
        except UnicodeDecodeError as exc:
            return FileIndexOutcome(
                status="failed",
                reason="decode_error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        except OSError as exc:
            return FileIndexOutcome(
                status="failed",
                reason="read_error",
                detail=f"{type(exc).__name__}: {exc}",
            )

        content_hash = self._hash_content(source)
        existing = self.cache.get_file_metadata(path)
        if existing and existing.content_hash == content_hash:
            return FileIndexOutcome(status="unchanged")

        parser_entry = self.parser_registry.get_parser_for_path(path)
        if not parser_entry:
            return FileIndexOutcome(
                status="failed",
                reason="parser_unavailable",
                detail="No parser registered for file extension.",
            )

        try:
            symbols = parser_entry.parser.extract_symbols(path, source)
        except Exception as exc:
            return FileIndexOutcome(
                status="failed",
                reason="parse_error",
                detail=f"{type(exc).__name__}: {exc}",
            )

        try:
            symbols = self._apply_embeddings(symbols, source)
            previous_symbol_ids = existing.symbols if existing else []
            if self.vector_store and previous_symbol_ids:
                self.vector_store.remove_ids(previous_symbol_ids)
            self.cache.delete_symbols_for_file(path)
            self.cache.upsert_symbols(symbols)
            self.cache.upsert_file_metadata(
                FileMetadata(
                    path=path,
                    language=parser_entry.language,
                    content_hash=content_hash,
                    symbols=[symbol.id for symbol in symbols],
                )
            )
            if self.vector_store and symbols:
                self.vector_store.upsert_vectors(symbols)
        except Exception as exc:
            return FileIndexOutcome(
                status="failed",
                reason="storage_error",
                detail=f"{type(exc).__name__}: {exc}",
            )

        return FileIndexOutcome(status="indexed", symbols_indexed=len(symbols))

    def index_file(self, path: str) -> Optional[int]:
        """Index a file and return symbol count when indexed, else None."""

        outcome = self.index_file_with_outcome(path)
        if outcome.status == "indexed":
            return outcome.symbols_indexed
        return None

    def _iter_source_files(self, root: str) -> Iterable[str]:
        """Yield supported source files under a root directory."""

        excludes = set(self.config.excluded_dirs)
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in excludes]
            for filename in files:
                full_path = os.path.join(current_root, filename)
                if self._is_supported_file(full_path):
                    yield full_path

    def _is_supported_file(self, path: str) -> bool:
        """Return whether a file path has a supported extension."""

        for ext in self.config.supported_extensions:
            if path.endswith(ext):
                return True
        return False

    def _apply_embeddings(self, symbols: List[Symbol], source: str) -> List[Symbol]:
        """Attach embedding vectors to symbols when a provider is available."""

        if not self.embedding_provider:
            return symbols
        lines = source.splitlines()
        texts = [self._symbol_text(symbol, lines) for symbol in symbols]
        if not texts:
            return symbols
        try:
            vectors = self.embedding_provider.embed_batch(texts)
        except Exception as exc:
            raise wrap_embedding_error(
                exc,
                provider=self.config.embedding_provider,
                operation="embed symbol batch for indexing",
            ) from exc
        for symbol, vector in zip(symbols, vectors):
            symbol.embedding_vector = vector
        return symbols

    @staticmethod
    def _hash_content(source: str) -> str:
        """Return sha256 hash of UTF-8 source text for change detection."""

        return hashlib.sha256(source.encode("utf8")).hexdigest()

    @staticmethod
    def _symbol_text(symbol: Symbol, lines: List[str]) -> str:
        """Build embedding text by slicing lines and joining signature/docstring/snippet."""

        snippet_start = max(0, symbol.start_line - 1)
        snippet_end = min(len(lines), snippet_start + 3)
        snippet = "\n".join(lines[snippet_start:snippet_end]).strip()
        parts = [symbol.signature or "", symbol.docstring or "", snippet]
        return "\n".join(part for part in parts if part).strip()
