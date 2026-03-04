from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import EmbeddingProviderError, wrap_embedding_error
from gloggur.graph.extractor import GraphEdgeExtractor
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.models import EdgeRecord, FileMetadata, IndexMetadata, Symbol, SymbolChunk
from gloggur.parsers.registry import ParserRegistry
from gloggur.storage.vector_store import VectorStore

FAILURE_REMEDIATION: dict[str, list[str]] = {
    "decode_error": [
        "Re-save the file in UTF-8 encoding.",
        "Exclude non-source/binary files from the indexing path.",
    ],
    "read_error": [
        "Verify file permissions and that the file still exists.",
        "Retry indexing after fixing filesystem read errors.",
    ],
    "parser_unavailable": [
        "Use a supported extension or register a parser for the file type.",
        "Exclude unsupported files from the repository path.",
    ],
    "parse_error": [
        "Fix syntax errors in the source file and rerun index.",
        "Use --allow-partial to continue while collecting parse failures.",
    ],
    "storage_error": [
        "Check cache directory writability and free disk space.",
        "Retry `gloggur index . --json` after resolving storage issues.",
    ],
    "embedding_provider_error": [
        "Inspect provider credentials, model configuration, "
        "and provider failure details in failed_samples.",
        "Retry `gloggur index . --json` after fixing the embedding provider error.",
    ],
    "stale_cleanup_error": [
        "Ensure stale file paths can be removed from cache metadata.",
        "Run `gloggur clear-cache --json` and rebuild if cleanup keeps failing.",
    ],
    "vector_metadata_mismatch": [
        "Rebuild vectors for the affected repository (`gloggur index . --json`).",
        "Run `gloggur clear-cache --json` if vector/cache state remains inconsistent.",
    ],
    "vector_consistency_unverifiable": [
        "Use a vector store implementation that supports deterministic symbol-id listing.",
        "Rerun `gloggur index . --json` after enabling vector/cache consistency checks.",
    ],
}


@dataclass
class IndexResult:
    """Dataclass for indexing results with explicit terminal file outcomes."""

    files_considered: int
    indexed: int
    unchanged: int
    failed: int
    indexed_symbols: int
    duration_ms: int
    files_changed: int = 0
    files_removed: int = 0
    symbols_added: int = 0
    symbols_updated: int = 0
    symbols_removed: int = 0
    failed_reasons: dict[str, int] = field(default_factory=dict)
    failed_samples: list[str] = field(default_factory=list)

    @property
    def indexed_files(self) -> int:
        """Backward-compatible alias for indexed files."""

        return self.indexed

    @property
    def skipped_files(self) -> int:
        """Backward-compatible alias for unchanged files."""

        return self.unchanged

    def as_payload(self) -> dict[str, object]:
        """Build a CLI payload with explicit outcomes and legacy aliases."""
        payload: dict[str, object] = {
            "files_considered": self.files_considered,
            "files_scanned": self.files_considered,
            "indexed": self.indexed,
            "unchanged": self.unchanged,
            "failed": self.failed,
            "failed_reasons": dict(self.failed_reasons),
            "failed_samples": list(self.failed_samples),
            "indexed_symbols": self.indexed_symbols,
            "files_changed": self.files_changed,
            "files_removed": self.files_removed,
            "symbols_added": self.symbols_added,
            "symbols_updated": self.symbols_updated,
            "symbols_removed": self.symbols_removed,
            "duration_ms": self.duration_ms,
            "indexed_files": self.indexed_files,
            "skipped_files": self.skipped_files,
        }
        if self.failed_reasons:
            payload["failure_codes"] = sorted(self.failed_reasons)
            payload["failure_guidance"] = {
                reason: FAILURE_REMEDIATION.get(
                    reason,
                    [
                        "Inspect failed_samples and rerun "
                        "indexing after resolving the "
                        "underlying error."
                    ],
                )
                for reason in sorted(self.failed_reasons)
            }
        return payload


@dataclass(frozen=True)
class FileIndexOutcome:
    """Per-file terminal outcome used to classify index runs."""

    status: str
    symbols_indexed: int = 0
    symbols_added: int = 0
    symbols_updated: int = 0
    symbols_removed: int = 0
    reason: str | None = None
    detail: str | None = None


@dataclass
class PreparedFileIndex:
    """Prepared file payload carried from parse phase into embedding/persist phase."""

    path: str
    language: str
    content_hash: str
    source: str
    symbols: list[Symbol]
    chunks: list[SymbolChunk]
    edges: list[EdgeRecord]
    previous_symbol_ids: list[str]
    previous_chunk_ids: list[str]
    previous_metadata: FileMetadata | None
    previous_symbols: list[Symbol]
    previous_chunks: list[SymbolChunk]
    previous_edges: list[EdgeRecord]
    symbols_added: int
    symbols_updated: int
    symbols_removed: int


class Indexer:
    """Incremental repository indexer that hashes files and stores symbols."""

    def __init__(
        self,
        config: GloggurConfig,
        cache: CacheManager | None = None,
        parser_registry: ParserRegistry | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize the indexer with cache, parsers, and embeddings."""

        self.config = config
        self.cache = cache or CacheManager(CacheConfig(config.cache_dir))
        self.parser_registry = parser_registry or ParserRegistry(
            extension_map=config.parser_extension_map,
            adapter_overrides=config.adapters if isinstance(config.adapters, dict) else None,
        )
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self._progress_callback: Callable[[int, int], None] | None = None
        self._scan_callback: Callable[[int, int, str], None] | None = None
        self._commit_cache: dict[str, str] = {}

    def index_repository(self, path: str) -> IndexResult:
        """Index all supported files under a repository root."""

        start = time.time()
        self.cache.delete_index_metadata()
        # Test-only hook to make interruption windows deterministic.
        self._maybe_pause_after_metadata_delete()

        source_files = list(self._iter_source_files(path))
        total_files = len(source_files)
        files_considered = 0
        indexed_files = 0
        unchanged_files = 0
        failed_files = 0
        indexed_symbols = 0
        symbols_added = 0
        symbols_updated = 0
        symbols_removed = 0
        files_removed = 0
        failed_reasons: dict[str, int] = {}
        failed_samples: list[str] = []
        seen_paths: set[str] = set()
        prepared_files: list[PreparedFileIndex] = []

        for file_path in source_files:
            seen_paths.add(file_path)
            files_considered += 1
            prepared, outcome = self._prepare_file_for_index(file_path)
            if prepared is not None:
                prepared_files.append(prepared)
                if self._scan_callback is not None:
                    self._scan_callback(files_considered, total_files, "prepared")
                continue
            if self._scan_callback is not None:
                self._scan_callback(files_considered, total_files, outcome.status)
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

        total_embedding_symbols = sum(len(prepared.chunks) for prepared in prepared_files)
        embedded_symbols_done = 0
        for prepared in prepared_files:
            file_symbols_total = len(prepared.chunks)
            file_progress_done = 0
            progress_callback = None
            if self._progress_callback is not None and total_embedding_symbols > 0:

                def _progress(
                    done: int,
                    total: int,
                    base: int = embedded_symbols_done,
                ) -> None:
                    """Translate file-local embedding progress into repository-global counts."""
                    nonlocal file_progress_done
                    _ = total
                    file_progress_done = done
                    self._progress_callback(base + done, total_embedding_symbols)

                progress_callback = _progress

            try:
                prepared.chunks = self._apply_embeddings(
                    prepared.chunks,
                    progress_callback=progress_callback,
                )
                prepared.edges = self._apply_edge_embeddings(prepared.edges)
                self._persist_prepared_file(prepared)
            except Exception as exc:
                failed_files += 1
                reason = (
                    "embedding_provider_error"
                    if isinstance(exc, EmbeddingProviderError)
                    else "storage_error"
                )
                failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                if len(failed_samples) < 5:
                    failed_samples.append(f"{prepared.path}: {type(exc).__name__}: {exc}")
                embedded_symbols_done += file_progress_done
                continue

            embedded_symbols_done += file_symbols_total
            indexed_files += 1
            indexed_symbols += len(prepared.symbols)
            symbols_added += prepared.symbols_added
            symbols_updated += prepared.symbols_updated
            symbols_removed += prepared.symbols_removed

        stale_cleanup = self._prune_stale_files(seen_paths)
        files_removed += stale_cleanup["files_removed"]
        symbols_removed += stale_cleanup["symbols_removed"]
        stale_cleanup_failures = stale_cleanup["failed"]
        failed_files += stale_cleanup_failures
        for reason, count in stale_cleanup["failed_reasons"].items():
            failed_reasons[reason] = failed_reasons.get(reason, 0) + count
        for sample in stale_cleanup["failed_samples"]:
            if len(failed_samples) >= 5:
                break
            failed_samples.append(sample)

        if self.vector_store:
            self.vector_store.save()
        consistency = self._validate_vector_metadata_consistency()
        failed_files += consistency["failed"]
        for reason, count in consistency["failed_reasons"].items():
            failed_reasons[reason] = failed_reasons.get(reason, 0) + count
        for sample in consistency["failed_samples"]:
            if len(failed_samples) >= 5:
                break
            failed_samples.append(sample)

        if failed_files == 0:
            metadata = IndexMetadata(
                version=self.config.index_version,
                total_symbols=self.cache.count_symbols(),
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
            files_changed=indexed_files,
            files_removed=files_removed,
            symbols_added=symbols_added,
            symbols_updated=symbols_updated,
            symbols_removed=symbols_removed,
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
        ready_path = os.getenv("GLOGGUR_TEST_PAUSE_AFTER_METADATA_DELETE_READY_FILE")
        if ready_path:
            ready_dir = os.path.dirname(ready_path)
            if ready_dir:
                os.makedirs(ready_dir, exist_ok=True)
            with open(ready_path, "w", encoding="utf8") as handle:
                handle.write("1")
        time.sleep(pause_ms / 1000.0)

    def index_file_with_outcome(self, path: str) -> FileIndexOutcome:
        """Index a file and return an explicit terminal outcome."""
        prepared, outcome = self._prepare_file_for_index(path)
        if prepared is None:
            return outcome

        try:
            prepared.chunks = self._apply_embeddings(
                prepared.chunks,
                progress_callback=self._progress_callback,
            )
            prepared.edges = self._apply_edge_embeddings(prepared.edges)
            self._persist_prepared_file(prepared)
        except Exception as exc:
            return FileIndexOutcome(
                status="failed",
                reason=(
                    "embedding_provider_error"
                    if isinstance(exc, EmbeddingProviderError)
                    else "storage_error"
                ),
                detail=f"{type(exc).__name__}: {exc}",
            )

        return FileIndexOutcome(
            status="indexed",
            symbols_indexed=len(prepared.symbols),
            symbols_added=prepared.symbols_added,
            symbols_updated=prepared.symbols_updated,
            symbols_removed=prepared.symbols_removed,
        )

    def _prepare_file_for_index(
        self, path: str
    ) -> tuple[PreparedFileIndex | None, FileIndexOutcome]:
        """Read/parse a file and compute symbol diff metadata for later persistence."""

        try:
            with open(path, encoding="utf8") as handle:
                source = handle.read()
        except UnicodeDecodeError as exc:
            return None, FileIndexOutcome(
                status="failed",
                reason="decode_error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        except OSError as exc:
            return None, FileIndexOutcome(
                status="failed",
                reason="read_error",
                detail=f"{type(exc).__name__}: {exc}",
            )

        content_hash = self._hash_content(source)
        existing = self.cache.get_file_metadata(path)
        previous_symbols: list[Symbol] = []
        previous_chunks: list[SymbolChunk] = []
        previous_edges: list[EdgeRecord] = []
        existing_symbols_by_id: dict[str, Symbol] = {}
        if existing:
            previous_symbols = self.cache.list_symbols_for_file(path)
            previous_chunks = self.cache.list_chunks_for_file(path)
            previous_edges = self.cache.list_edges_for_file(path)
            existing_symbols_by_id = {symbol.id: symbol for symbol in previous_symbols}
        if existing and existing.content_hash == content_hash:
            return None, FileIndexOutcome(status="unchanged")

        parser_entry = self.parser_registry.get_parser_for_path(path)
        if not parser_entry:
            return None, FileIndexOutcome(
                status="failed",
                reason="parser_unavailable",
                detail="No parser registered for file extension.",
            )

        try:
            symbols = parser_entry.parser.extract_symbols(path, source)
        except Exception as exc:
            return None, FileIndexOutcome(
                status="failed",
                reason="parse_error",
                detail=f"{type(exc).__name__}: {exc}",
            )

        repo_id = self._repo_id(path)
        commit = self._resolve_commit(path)
        for symbol in symbols:
            if not symbol.repo_id:
                symbol.repo_id = repo_id
            symbol.commit = commit

        chunks = self._build_symbol_chunks(path=path, source=source, symbols=symbols, commit=commit)
        candidate_symbols = self.cache.list_symbols()
        edges = self._build_graph_edges(
            path=path,
            source=source,
            symbols=symbols,
            candidate_symbols=candidate_symbols,
            repo_id=repo_id,
            commit=commit,
            language=parser_entry.language,
        )

        previous_symbol_ids = existing.symbols if existing else []
        previous_chunk_ids = [chunk.chunk_id for chunk in previous_chunks]
        new_symbols_by_id = {symbol.id: symbol for symbol in symbols}
        new_symbol_ids = set(new_symbols_by_id)
        previous_symbol_ids_set = set(previous_symbol_ids)
        symbols_added = len(new_symbol_ids - previous_symbol_ids_set)
        symbols_removed = len(previous_symbol_ids_set - new_symbol_ids)
        symbols_updated = 0
        for symbol_id in new_symbol_ids & previous_symbol_ids_set:
            previous_symbol = existing_symbols_by_id.get(symbol_id)
            current_symbol = new_symbols_by_id.get(symbol_id)
            if previous_symbol is None or current_symbol is None:
                continue
            if previous_symbol.body_hash != current_symbol.body_hash:
                symbols_updated += 1

        prepared = PreparedFileIndex(
            path=path,
            language=parser_entry.language,
            content_hash=content_hash,
            source=source,
            symbols=symbols,
            chunks=chunks,
            edges=edges,
            previous_symbol_ids=previous_symbol_ids,
            previous_chunk_ids=previous_chunk_ids,
            previous_metadata=existing.model_copy(deep=True) if existing else None,
            previous_symbols=[symbol.model_copy(deep=True) for symbol in previous_symbols],
            previous_chunks=[chunk.model_copy(deep=True) for chunk in previous_chunks],
            previous_edges=[edge.model_copy(deep=True) for edge in previous_edges],
            symbols_added=symbols_added,
            symbols_updated=symbols_updated,
            symbols_removed=symbols_removed,
        )
        return prepared, FileIndexOutcome(status="prepared", symbols_indexed=len(symbols))

    def _persist_prepared_file(self, prepared: PreparedFileIndex) -> None:
        """Persist one prepared file into cache and vector store."""
        next_symbol_ids = [symbol.id for symbol in prepared.symbols]
        next_chunk_ids = [chunk.chunk_id for chunk in prepared.chunks]
        try:
            if self.vector_store and prepared.previous_chunk_ids:
                self.vector_store.remove_ids(prepared.previous_chunk_ids)
            self.cache.replace_file_index(
                prepared.path,
                FileMetadata(
                    path=prepared.path,
                    language=prepared.language,
                    content_hash=prepared.content_hash,
                    symbols=next_symbol_ids,
                ),
                prepared.symbols,
                prepared.chunks,
                prepared.edges,
            )
            if self.vector_store and prepared.chunks:
                self.vector_store.upsert_vectors(prepared.chunks)
        except Exception as persist_exc:
            try:
                self._rollback_persisted_file(prepared, next_symbol_ids, next_chunk_ids)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "file persist failed and rollback did not complete "
                    f"(persist={type(persist_exc).__name__}: {persist_exc}; "
                    f"rollback={type(rollback_exc).__name__}: {rollback_exc})"
                ) from persist_exc
            raise

    def _rollback_persisted_file(
        self,
        prepared: PreparedFileIndex,
        next_symbol_ids: list[str],
        next_chunk_ids: list[str],
    ) -> None:
        """Best-effort restore of file-local cache/vector state after persist failure."""
        if prepared.previous_metadata is None:
            self.cache.delete_symbols_for_file(prepared.path)
            self.cache.delete_chunks_for_file(prepared.path)
            self.cache.delete_edges_for_file(prepared.path)
            self.cache.delete_file_metadata(prepared.path)
        else:
            self.cache.replace_file_index(
                prepared.path,
                prepared.previous_metadata,
                prepared.previous_symbols,
                prepared.previous_chunks,
                prepared.previous_edges,
            )

        if not self.vector_store:
            return
        rollback_ids = {symbol_id for symbol_id in next_symbol_ids if symbol_id}
        rollback_ids.update(chunk_id for chunk_id in next_chunk_ids if chunk_id)
        rollback_ids.update(symbol.id for symbol in prepared.previous_symbols if symbol.id)
        rollback_ids.update(chunk.chunk_id for chunk in prepared.previous_chunks if chunk.chunk_id)
        if rollback_ids:
            self.vector_store.remove_ids(sorted(rollback_ids))
        if prepared.previous_chunks:
            self.vector_store.upsert_vectors(prepared.previous_chunks)

    def index_file(self, path: str) -> int | None:
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

    def _prune_stale_files(self, seen_paths: set[str]) -> dict[str, object]:
        """Remove cached file/symbol rows for files no longer present in the index walk."""
        files_removed = 0
        symbols_removed = 0
        failed = 0
        failed_reasons: dict[str, int] = {}
        failed_samples: list[str] = []
        cached_paths = set(self.cache.list_file_paths())
        stale_paths = sorted(cached_paths - seen_paths)
        for stale_path in stale_paths:
            try:
                metadata = self.cache.get_file_metadata(stale_path)
                previous_symbol_ids = metadata.symbols if metadata else []
                previous_chunk_ids = [
                    chunk.chunk_id for chunk in self.cache.list_chunks_for_file(stale_path)
                ]
                if self.vector_store and previous_chunk_ids:
                    self.vector_store.remove_ids(previous_chunk_ids)
                self.cache.delete_symbols_for_file(stale_path)
                self.cache.delete_chunks_for_file(stale_path)
                self.cache.delete_edges_for_file(stale_path)
                self.cache.delete_file_metadata(stale_path)
                files_removed += 1
                symbols_removed += len(previous_symbol_ids)
            except Exception as exc:
                failed += 1
                reason = "stale_cleanup_error"
                failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                if len(failed_samples) < 5:
                    failed_samples.append(f"{stale_path}: {type(exc).__name__}: {exc}")
        return {
            "files_removed": files_removed,
            "symbols_removed": symbols_removed,
            "failed": failed,
            "failed_reasons": failed_reasons,
            "failed_samples": failed_samples,
        }

    def prune_missing_file_entries(self) -> dict[str, object]:
        """Remove cached entries whose files no longer exist on disk."""
        cached_paths = set(self.cache.list_file_paths())
        existing_paths = {path for path in cached_paths if os.path.exists(path)}
        return self._prune_stale_files(existing_paths)

    def _validate_vector_metadata_consistency(self) -> dict[str, object]:
        """Detect vector/cache symbol-id divergence and return stable failure metadata."""
        if self.embedding_provider is None or self.vector_store is None:
            return {"failed": 0, "failed_reasons": {}, "failed_samples": []}
        list_symbol_ids = getattr(self.vector_store, "list_symbol_ids", None)
        if not callable(list_symbol_ids):
            return {
                "failed": 1,
                "failed_reasons": {"vector_consistency_unverifiable": 1},
                "failed_samples": [
                    "vector consistency check unavailable: "
                    "vector store does not expose "
                    "list_symbol_ids()"
                ],
            }
        try:
            vector_symbol_ids = {str(symbol_id) for symbol_id in list_symbol_ids()}
        except Exception as exc:
            return {
                "failed": 1,
                "failed_reasons": {"vector_metadata_mismatch": 1},
                "failed_samples": [f"vector metadata check failed: {type(exc).__name__}: {exc}"],
            }

        cache_chunk_ids = {chunk.chunk_id for chunk in self.cache.list_chunks()}
        missing_vectors = sorted(cache_chunk_ids - vector_symbol_ids)
        stale_vectors = sorted(vector_symbol_ids - cache_chunk_ids)
        if not missing_vectors and not stale_vectors:
            return {"failed": 0, "failed_reasons": {}, "failed_samples": []}
        sample = (
            "vector/cache mismatch "
            f"(missing_vectors={len(missing_vectors)}, stale_vectors={len(stale_vectors)})"
        )
        if missing_vectors:
            sample += f"; missing_example={missing_vectors[0]}"
        if stale_vectors:
            sample += f"; stale_example={stale_vectors[0]}"
        return {
            "failed": 1,
            "failed_reasons": {"vector_metadata_mismatch": 1},
            "failed_samples": [sample],
        }

    def validate_vector_metadata_consistency(self) -> dict[str, object]:
        """Public post-index consistency check used by both repo and single-file index flows."""
        return self._validate_vector_metadata_consistency()

    def _apply_embeddings(
        self,
        chunks: list[SymbolChunk],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SymbolChunk]:
        """Attach embedding vectors to chunk rows when available."""

        if not self.embedding_provider:
            return chunks
        if not chunks:
            return chunks
        texts = [chunk.text for chunk in chunks]
        if not texts:
            return chunks
        total = len(chunks)
        chunk_size = getattr(self.embedding_provider, "_chunk_size", 50)
        if progress_callback is not None:
            progress_callback(0, total)
        for i in range(0, total, chunk_size):
            chunk_texts = texts[i : i + chunk_size]
            chunk_rows = chunks[i : i + chunk_size]
            try:
                vectors = self.embedding_provider.embed_batch(chunk_texts)
                if len(vectors) != len(chunk_rows):
                    raise RuntimeError(
                        "embedding provider returned "
                        f"{len(vectors)} vectors for {len(chunk_rows)} "
                        "chunks during indexing"
                    )
            except Exception as exc:
                raise wrap_embedding_error(
                    exc,
                    provider=self.config.embedding_provider,
                    operation="embed chunk batch for indexing",
                ) from exc
            for chunk_row, vector in zip(chunk_rows, vectors, strict=True):
                chunk_row.embedding_vector = vector
            done = min(i + chunk_size, total)
            if progress_callback is not None:
                progress_callback(done, total)
        return chunks

    def _apply_edge_embeddings(self, edges: list[EdgeRecord]) -> list[EdgeRecord]:
        """Attach embedding vectors to edge fact text when a provider is available."""
        if not self.embedding_provider:
            return edges
        texts = [edge.text or "" for edge in edges]
        if not texts:
            return edges
        batch_size = getattr(self.embedding_provider, "_chunk_size", 50)
        for offset in range(0, len(edges), batch_size):
            edge_batch = edges[offset : offset + batch_size]
            text_batch = texts[offset : offset + batch_size]
            try:
                vectors = self.embedding_provider.embed_batch(text_batch)
                if len(vectors) != len(edge_batch):
                    raise RuntimeError(
                        "embedding provider returned "
                        f"{len(vectors)} vectors for {len(edge_batch)} edges during indexing"
                    )
            except Exception as exc:
                raise wrap_embedding_error(
                    exc,
                    provider=self.config.embedding_provider,
                    operation="embed edge batch for indexing",
                ) from exc
            for edge_row, vector in zip(edge_batch, vectors, strict=True):
                edge_row.embedding_vector = vector
        return edges

    @staticmethod
    def _build_graph_edges(
        *,
        path: str,
        source: str,
        symbols: list[Symbol],
        candidate_symbols: list[Symbol],
        repo_id: str,
        commit: str,
        language: str,
    ) -> list[EdgeRecord]:
        """Extract deterministic graph edges for a file from symbols and source text."""
        extractor = GraphEdgeExtractor(language)
        return extractor.extract_edges(
            path=path,
            source=source,
            symbols=symbols,
            candidate_symbols=candidate_symbols,
            repo_id=repo_id,
            commit=commit,
        )

    @staticmethod
    def _hash_content(source: str) -> str:
        """Return sha256 hash of UTF-8 source text for change detection."""

        return hashlib.sha256(source.encode("utf8")).hexdigest()

    def _build_symbol_chunks(
        self,
        *,
        path: str,
        source: str,
        symbols: list[Symbol],
        commit: str,
    ) -> list[SymbolChunk]:
        """Build deterministic symbol-boundary embedding chunks with split support."""
        lines = source.splitlines()
        imports = self._extract_import_lines(lines)
        chunks: list[SymbolChunk] = []
        for symbol in symbols:
            body_lines = lines[max(0, symbol.start_line - 1) : max(0, symbol.end_line)]
            body_text = "\n".join(body_lines).strip()
            signature_block = self._build_signature_block(symbol)
            imports_block = self._build_imports_in_scope_block(imports=imports, body_text=body_text)
            header_lines = [
                f"FQNAME: {symbol.fqname or symbol.name}",
                f"KIND: {symbol.kind}",
                f"FILE: {symbol.file_path}",
                f"LINES: {symbol.start_line}-{symbol.end_line}",
            ]
            prefix_sections = ["\n".join(header_lines), signature_block]
            if imports_block:
                prefix_sections.append(imports_block)
            prefix = "\n\n".join(section for section in prefix_sections if section).strip()
            max_bytes = max(256, int(getattr(self.config, "max_symbol_chunk_bytes", 12000)))
            parts = self._split_symbol_body_parts(
                prefix=prefix,
                body_lines=body_lines,
                start_line=symbol.start_line,
                max_bytes=max_bytes,
            )
            total_parts = len(parts)
            for index, (part_start, part_end, part_body_text) in enumerate(parts, start=1):
                text_sections = [prefix, "BODY:", part_body_text]
                text = "\n\n".join(section for section in text_sections if section).strip()
                chunk_id = self._hash_chunk_id(
                    symbol_id=symbol.id,
                    part_index=index,
                    part_total=total_parts,
                    start_line=part_start,
                    end_line=part_end,
                )
                chunks.append(
                    SymbolChunk(
                        chunk_id=chunk_id,
                        symbol_id=symbol.id,
                        chunk_part_index=index,
                        chunk_part_total=total_parts,
                        text=text,
                        file_path=path,
                        start_line=part_start,
                        end_line=part_end,
                        tokens_estimate=self._estimate_tokens(text),
                        language=symbol.language,
                        repo_id=symbol.repo_id,
                        commit=commit,
                    )
                )
        return chunks

    @staticmethod
    def _extract_import_lines(lines: list[str]) -> list[str]:
        """Extract candidate import lines for import-in-scope blocks."""
        candidates: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(
                ("import ", "from ", "use ", "package ", "#include ", "require(")
            ):
                candidates.append(stripped)
        return candidates

    @staticmethod
    def _build_signature_block(symbol: Symbol) -> str:
        """Build a deterministic best-effort signature block."""
        signature = (symbol.signature or "").strip()
        if not signature:
            return "SIGNATURE:\nRAW: <unknown>"
        params = "<unknown>"
        returns = "<unknown>"
        modifiers: list[str] = []

        if "(" in signature and ")" in signature:
            params = signature.split("(", 1)[1].split(")", 1)[0].strip() or "<none>"
        if "->" in signature:
            returns = signature.split("->", 1)[1].split(":", 1)[0].strip() or "<unknown>"
        lower = signature.lower()
        for marker in (
            "public",
            "private",
            "protected",
            "static",
            "async",
            "final",
            "pub",
            "export",
        ):
            if marker in lower:
                modifiers.append(marker)
        modifier_text = ", ".join(dict.fromkeys(modifiers)) if modifiers else "<none>"
        return "\n".join(
            [
                "SIGNATURE:",
                f"RAW: {signature}",
                f"PARAMS: {params}",
                f"RETURNS: {returns}",
                f"MODIFIERS: {modifier_text}",
            ]
        )

    @staticmethod
    def _build_imports_in_scope_block(
        *,
        imports: list[str],
        body_text: str,
        max_items: int = 8,
    ) -> str:
        """Build capped import-in-scope section based on simple lexical matching."""
        if not imports:
            return ""
        body_lower = body_text.lower()
        selected: list[str] = []
        for statement in imports:
            tokens = [token for token in re.split(r"[^A-Za-z0-9_]+", statement) if token]
            token_match = any(token.lower() in body_lower for token in tokens if len(token) > 2)
            if token_match:
                selected.append(statement)
            if len(selected) >= max_items:
                break
        if not selected:
            return ""
        return "IMPORTS_IN_SCOPE:\n" + "\n".join(selected)

    @classmethod
    def _split_symbol_body_parts(
        cls,
        *,
        prefix: str,
        body_lines: list[str],
        start_line: int,
        max_bytes: int,
    ) -> list[tuple[int, int, str]]:
        """Split one symbol body into deterministic parts constrained by byte budget."""
        if not body_lines:
            return [(start_line, start_line, "")]

        prefix_bytes = len(prefix.encode("utf8"))
        body_text = "\n".join(body_lines).strip()
        whole_text = "\n\n".join([prefix, "BODY:", body_text]).strip()
        if len(whole_text.encode("utf8")) <= max_bytes:
            return [(start_line, start_line + len(body_lines) - 1, body_text)]

        parts: list[tuple[int, int, str]] = []
        current_lines: list[str] = []
        current_start = start_line
        for offset, line in enumerate(body_lines):
            absolute_line = start_line + offset
            candidate_lines = current_lines + [line]
            candidate_text = "\n".join(candidate_lines).strip()
            candidate_bytes = len(candidate_text.encode("utf8")) + prefix_bytes + len("BODY:") + 4
            if current_lines and candidate_bytes > max_bytes:
                finalized_text = "\n".join(current_lines).strip()
                parts.append((current_start, absolute_line - 1, finalized_text))
                current_lines = [line]
                current_start = absolute_line
            else:
                current_lines.append(line)

        if current_lines:
            finalized_text = "\n".join(current_lines).strip()
            parts.append((current_start, start_line + len(body_lines) - 1, finalized_text))
        return parts

    @staticmethod
    def _hash_chunk_id(
        *,
        symbol_id: str,
        part_index: int,
        part_total: int,
        start_line: int,
        end_line: int,
    ) -> str:
        """Build deterministic chunk id from symbol identity and chunk coordinates."""
        payload = "|".join(
            [symbol_id, str(part_index), str(part_total), str(start_line), str(end_line)]
        )
        return hashlib.sha256(payload.encode("utf8")).hexdigest()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Best-effort token estimate used for chunk metadata sizing."""
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, len(stripped.split()))

    def _repo_id(self, path: str) -> str:
        """Return deterministic repo id for the path using nearest .git root."""
        absolute = os.path.abspath(path)
        current = absolute if os.path.isdir(absolute) else os.path.dirname(absolute)
        while True:
            if os.path.exists(os.path.join(current, ".git")):
                return hashlib.sha256(current.encode("utf8")).hexdigest()
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        fallback = os.path.dirname(absolute) if os.path.isfile(absolute) else absolute
        return hashlib.sha256(fallback.encode("utf8")).hexdigest()

    def _resolve_commit(self, path: str) -> str:
        """Resolve HEAD commit for nearest git root with deterministic fallback."""
        absolute = os.path.abspath(path)
        current = absolute if os.path.isdir(absolute) else os.path.dirname(absolute)
        repo_root: str | None = None
        while True:
            if os.path.exists(os.path.join(current, ".git")):
                repo_root = current
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        if repo_root is None:
            return "unknown"
        cached = self._commit_cache.get(repo_root)
        if cached is not None:
            return cached
        try:
            completed = subprocess.run(
                ["git", "-C", repo_root, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            commit = completed.stdout.strip() or "unknown"
        except Exception:
            commit = "unknown"
        self._commit_cache[repo_root] = commit
        return commit
