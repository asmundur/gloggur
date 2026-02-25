from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import wrap_embedding_error
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.models import FileMetadata, IndexMetadata, Symbol
from gloggur.parsers.registry import ParserRegistry
from gloggur.storage.vector_store import VectorStore


@dataclass
class IndexResult:
    """Dataclass for indexing results: indexed_files, indexed_symbols, skipped_files, duration_ms."""
    indexed_files: int
    indexed_symbols: int
    skipped_files: int
    duration_ms: int


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
        indexed_files = 0
        indexed_symbols = 0
        skipped_files = 0
        for file_path in self._iter_source_files(path):
            result = self.index_file(file_path)
            if result:
                indexed_files += 1
                indexed_symbols += result
            else:
                skipped_files += 1
        if self.vector_store:
            self.vector_store.save()
        metadata = IndexMetadata(
            version=self.config.index_version,
            total_symbols=len(self.cache.list_symbols()),
            indexed_files=self.cache.count_files(),
        )
        self.cache.set_index_metadata(metadata)
        self.cache.set_index_profile(self.config.embedding_profile())
        duration_ms = int((time.time() - start) * 1000)
        return IndexResult(
            indexed_files=indexed_files,
            indexed_symbols=indexed_symbols,
            skipped_files=skipped_files,
            duration_ms=duration_ms,
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

    def index_file(self, path: str) -> Optional[int]:
        """Index a file: hash content, parse symbols, update cache/vector store, return count."""
        try:
            with open(path, "r", encoding="utf8") as handle:
                source = handle.read()
        except (OSError, UnicodeDecodeError):
            return None
        content_hash = self._hash_content(source)
        existing = self.cache.get_file_metadata(path)
        if existing and existing.content_hash == content_hash:
            return None
        parser_entry = self.parser_registry.get_parser_for_path(path)
        if not parser_entry:
            return None
        previous_symbol_ids = existing.symbols if existing else []
        if self.vector_store and previous_symbol_ids:
            self.vector_store.remove_ids(previous_symbol_ids)
        symbols = parser_entry.parser.extract_symbols(path, source)
        symbols = self._apply_embeddings(symbols, source)
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
        return len(symbols)

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
        """Build embedding text by slicing lines (start_line-1 .. start_line+3) and joining with signature/docstring."""
        snippet_start = max(0, symbol.start_line - 1)
        snippet_end = min(len(lines), snippet_start + 3)
        snippet = "\n".join(lines[snippet_start:snippet_end]).strip()
        parts = [symbol.signature or "", symbol.docstring or "", snippet]
        return "\n".join(part for part in parts if part).strip()
