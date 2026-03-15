from __future__ import annotations

import hashlib
import multiprocessing
import os
import queue
import re
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from gloggur.byte_spans import LineByteSpanIndex
from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import EmbeddingProviderError, wrap_embedding_error
from gloggur.graph.extractor import GraphEdgeExtractor
from gloggur.indexer.cache import (
    BUILD_FILE_CHECKPOINT_STATE_EMBEDDED_COMPLETE,
    BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE,
    BuildFileCheckpoint,
    CacheConfig,
    CacheManager,
)
from gloggur.indexer.embedding_ledger import EmbeddingLedger, embedding_text_hash
from gloggur.indexer.shared import FileTimingTrace, ParsedFileSnapshot
from gloggur.models import EdgeRecord, FileMetadata, IndexMetadata, Symbol, SymbolChunk
from gloggur.parsers.registry import ParserRegistry
from gloggur.parsers.treesitter_parser import TreeSitterParser
from gloggur.path_filters import filter_index_walk_dirs, is_indexable_source_path
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
    "chunk_span_integrity_error": [
        "Inspect symbol boundaries and persisted chunk spans for the failed file.",
        "Rerun `gloggur index . --json` after fixing parser output or chunk construction drift.",
    ],
    "extract_symbols_timeout": [
        "Inspect `status --json` or `support collect --json` for extract progress metadata "
        "showing the stuck file and subphase.",
        "Retry indexing after investigating the parser or graph-edge extraction path for that "
        "file, or use --allow-partial to continue past the timed-out file.",
    ],
}

EXTRACT_SYMBOLS_TIMEOUT_REASON = "extract_symbols_timeout"
_EXTRACT_WORKER_POLL_SECONDS = 0.1
_EXTRACT_PROGRESS_HEARTBEAT_SECONDS = 1.0
_EXTRACT_WORKER_STARTUP_TIMEOUT_SECONDS = 10.0
_TREE_SITTER_EXTRACT_SYMBOLS_IMPL = TreeSitterParser.extract_symbols


class _ExtractWorkerError(RuntimeError):
    """Raised when the extract-symbols worker reports a non-timeout failure."""


class _ExtractWorkerTimeout(RuntimeError):
    """Raised when one extract-symbols worker job exceeds the configured timeout."""

    def __init__(
        self,
        *,
        kind: str,
        path: str,
        timeout_seconds: float,
        worker_pid: int | None,
    ) -> None:
        """Initialize the timeout with stable reporting fields for CLI consumers."""
        self.kind = kind
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.worker_pid = worker_pid
        super().__init__(
            f"{kind} timed out for {path} after {timeout_seconds:.3f}s"
            + (f" (worker_pid={worker_pid})" if worker_pid is not None else "")
        )


def _extract_symbols_worker_main(
    config: GloggurConfig,
    request_queue: object,
    response_queue: object,
) -> None:
    """Serve extract-symbols jobs in a reusable child process."""
    try:
        cache = CacheManager(CacheConfig(config.cache_dir))
        parser_registry = ParserRegistry(
            extension_map=config.parser_extension_map,
            adapter_overrides=config.adapters if isinstance(config.adapters, dict) else None,
        )
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=parser_registry,
        )
    except BaseException as exc:
        try:
            response_queue.put(
                {
                    "kind": "startup_error",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        return

    try:
        response_queue.put({"kind": "ready", "ok": True})
    except Exception:
        return

    while True:
        request = request_queue.get()
        if request is None:
            return

        job_id = int(request.get("job_id", 0) or 0)
        kind = str(request.get("kind") or "")
        payload = request.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        try:
            if kind == "prepare_file":
                prepared, outcome, timing = indexer._prepare_file_for_index(
                    str(payload["path"]),
                    existing_content_hash=(
                        str(payload["existing_content_hash"])
                        if payload.get("existing_content_hash") is not None
                        else None
                    ),
                    capture_verbose_metrics=bool(payload.get("capture_verbose_metrics")),
                )
                response_queue.put(
                    {
                        "job_id": job_id,
                        "kind": kind,
                        "ok": True,
                        "prepared": prepared,
                        "outcome": outcome,
                        "timing": timing,
                    }
                )
                continue

            if kind == "build_edges":
                edges = Indexer._build_graph_edges(
                    path=str(payload["path"]),
                    source=str(payload["source"]),
                    symbols=list(payload.get("symbols") or []),
                    candidate_symbols=list(payload.get("candidate_symbols") or []),
                    repo_id=str(payload["repo_id"]),
                    commit=str(payload["commit"]),
                    language=str(payload["language"]),
                    include_text=bool(payload.get("include_text")),
                )
                response_queue.put(
                    {
                        "job_id": job_id,
                        "kind": kind,
                        "ok": True,
                        "edges": edges,
                    }
                )
                continue

            raise ValueError(f"unsupported extract worker job kind: {kind}")
        except BaseException as exc:
            try:
                response_queue.put(
                    {
                        "job_id": job_id,
                        "kind": kind,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            except Exception:
                return


class _ExtractSymbolsWorker:
    """Reusable spawned worker for extract-symbols subphases."""

    def __init__(self, config: GloggurConfig) -> None:
        """Capture worker config and initialize lazy process state."""
        self._config = config
        self._ctx = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._request_queue: object | None = None
        self._response_queue: object | None = None
        self._next_job_id = 0
        self._ready = False

    @property
    def pid(self) -> int | None:
        """Return the active worker pid when the child process is running."""
        process = self._process
        return process.pid if process is not None else None

    def ensure_started(self) -> None:
        """Start the worker when it is not already alive."""
        process = self._process
        if process is not None and process.is_alive() and self._ready:
            return
        self._shutdown(force=True)
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_extract_symbols_worker_main,
            args=(self._config, self._request_queue, self._response_queue),
            daemon=True,
        )
        self._process.start()
        self._await_ready()

    def _await_ready(self) -> None:
        """Wait for the spawned worker to finish imports before timing jobs."""
        assert self._response_queue is not None
        deadline = time.monotonic() + _EXTRACT_WORKER_STARTUP_TIMEOUT_SECONDS
        while True:
            process = self._process
            if process is None:
                raise _ExtractWorkerError("extract worker disappeared during startup")
            if not process.is_alive():
                exit_code = process.exitcode
                self._shutdown(force=True)
                raise _ExtractWorkerError(
                    f"extract worker exited during startup (exitcode={exit_code})"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                worker_pid = self.pid
                self._shutdown(force=True)
                detail = "extract worker failed to signal ready before startup timeout"
                if worker_pid is not None:
                    detail += f" (worker_pid={worker_pid})"
                raise _ExtractWorkerError(detail)
            try:
                response = self._response_queue.get(
                    timeout=min(_EXTRACT_WORKER_POLL_SECONDS, remaining)
                )
            except queue.Empty:
                continue
            if not isinstance(response, dict):
                continue
            kind = str(response.get("kind") or "")
            if kind == "ready":
                self._ready = True
                return
            if kind == "startup_error":
                error_type = str(response.get("error_type") or "WorkerStartupError")
                error_message = str(response.get("error_message") or "worker startup failed")
                trace_text = str(response.get("traceback") or "").strip()
                detail = f"{error_type}: {error_message}"
                if trace_text:
                    detail = f"{detail}\n{trace_text}"
                self._shutdown(force=True)
                raise _ExtractWorkerError(detail)

    def run_job(
        self,
        *,
        kind: str,
        payload: dict[str, object],
        timeout_seconds: float,
        on_poll: Callable[[], None] | None = None,
    ) -> dict[str, object]:
        """Run one job or raise a structured timeout/worker error."""
        self.ensure_started()
        assert self._request_queue is not None
        assert self._response_queue is not None
        self._next_job_id += 1
        job_id = self._next_job_id
        self._request_queue.put(
            {
                "job_id": job_id,
                "kind": kind,
                "payload": dict(payload),
            }
        )
        deadline = time.monotonic() + max(0.001, float(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                worker_pid = self.pid
                self._shutdown(force=True)
                raise _ExtractWorkerTimeout(
                    kind=kind,
                    path=str(payload.get("path") or ""),
                    timeout_seconds=float(timeout_seconds),
                    worker_pid=worker_pid,
                )
            try:
                response = self._response_queue.get(
                    timeout=min(_EXTRACT_WORKER_POLL_SECONDS, remaining)
                )
            except queue.Empty:
                if on_poll is not None:
                    on_poll()
                continue
            if not isinstance(response, dict) or int(response.get("job_id", -1)) != job_id:
                continue
            if bool(response.get("ok")):
                return response
            error_type = str(response.get("error_type") or "WorkerError")
            error_message = str(response.get("error_message") or "worker job failed")
            trace_text = str(response.get("traceback") or "").strip()
            detail = f"{error_type}: {error_message}"
            if trace_text:
                detail = f"{detail}\n{trace_text}"
            raise _ExtractWorkerError(detail)

    def restart(self) -> None:
        """Restart the worker after a timeout so later jobs can continue safely."""
        self._shutdown(force=True)
        self.ensure_started()

    def close(self) -> None:
        """Stop the worker process and release its queues."""
        self._shutdown(force=False)

    def _shutdown(self, *, force: bool) -> None:
        """Stop the worker process and tear down queue handles."""
        process = self._process
        request_queue = self._request_queue
        response_queue = self._response_queue
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._ready = False
        if process is not None:
            try:
                if process.is_alive() and request_queue is not None and not force:
                    request_queue.put(None)
                    process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=5)
            except Exception:
                pass
        for handle in (request_queue, response_queue):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception:
                pass
            try:
                handle.join_thread()
            except Exception:
                pass


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
    phase_timings_ms: dict[str, int] = field(default_factory=dict)
    index_stats: dict[str, int] = field(default_factory=dict)
    file_timings: list[FileTimingTrace] = field(default_factory=list)
    parsed_files: list[ParsedFileSnapshot] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    verbose_lines: VerboseLineMetrics | None = None

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
            "index_stats": dict(self.index_stats),
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
class VerboseLineKindMetrics:
    """Line metrics for one embedded artifact kind."""

    vector_count: int = 0
    line_total: int = 0
    line_unique: int = 0

    def as_payload(self) -> dict[str, int]:
        """Serialize one embedded-artifact metric bucket."""
        return {
            "vector_count": self.vector_count,
            "line_total": self.line_total,
            "line_unique": self.line_unique,
        }


@dataclass(frozen=True)
class VerboseLineMetrics:
    """Verbose index line metrics derived from source and embedded spans."""

    source_total: int = 0
    embedded_total: int = 0
    embedded_unique: int = 0
    symbol_chunks: VerboseLineKindMetrics = field(default_factory=VerboseLineKindMetrics)
    graph_edges: VerboseLineKindMetrics = field(default_factory=VerboseLineKindMetrics)

    @property
    def embedded_duplicate(self) -> int:
        """Return duplicate embedded line count across all embedded artifacts."""
        return max(0, self.embedded_total - self.embedded_unique)

    @property
    def vector_count(self) -> int:
        """Return total embedded vector count across artifact kinds."""
        return self.symbol_chunks.vector_count + self.graph_edges.vector_count

    @staticmethod
    def _ratio_or_none(numerator: int, denominator: int) -> float | None:
        """Return a stable floating-point ratio or None when undefined."""
        if denominator <= 0:
            return None
        return round(numerator / denominator, 6)

    def as_payload(self) -> dict[str, object]:
        """Serialize verbose line metrics for CLI output."""
        return {
            "source_total": self.source_total,
            "embedded_total": self.embedded_total,
            "embedded_unique": self.embedded_unique,
            "embedded_duplicate": self.embedded_duplicate,
            "source_coverage_ratio": self._ratio_or_none(
                self.embedded_unique,
                self.source_total,
            ),
            "duplication_ratio": self._ratio_or_none(
                self.embedded_duplicate,
                self.embedded_total,
            ),
            "vector_count": self.vector_count,
            "by_kind": {
                "symbol_chunks": self.symbol_chunks.as_payload(),
                "graph_edges": self.graph_edges.as_payload(),
            },
        }


def _merged_line_count(ranges_by_path: dict[str, list[tuple[int, int]]]) -> int:
    """Return total unique line coverage after merging ranges per file path."""
    total = 0
    for ranges in ranges_by_path.values():
        if not ranges:
            continue
        sorted_ranges = sorted(ranges)
        current_start, current_end = sorted_ranges[0]
        for start_line, end_line in sorted_ranges[1:]:
            if start_line <= current_end + 1:
                current_end = max(current_end, end_line)
                continue
            total += current_end - current_start + 1
            current_start, current_end = start_line, end_line
        total += current_end - current_start + 1
    return total


@dataclass
class _VerboseLineKindAccumulator:
    """Mutable accumulator for one embedded artifact kind."""

    vector_count: int = 0
    line_total: int = 0
    ranges_by_path: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def add_span(self, path: str, start_line: int, end_line: int) -> tuple[int, int]:
        """Record one embedded line span and return the normalized range."""
        normalized_start = max(1, int(start_line))
        normalized_end = max(normalized_start, int(end_line))
        self.vector_count += 1
        self.line_total += normalized_end - normalized_start + 1
        self.ranges_by_path.setdefault(path, []).append((normalized_start, normalized_end))
        return normalized_start, normalized_end

    def build(self) -> VerboseLineKindMetrics:
        """Freeze one embedded-artifact bucket into a payload-ready metrics object."""
        return VerboseLineKindMetrics(
            vector_count=self.vector_count,
            line_total=self.line_total,
            line_unique=_merged_line_count(self.ranges_by_path),
        )


@dataclass
class VerboseLineMetricsAccumulator:
    """Bounded accumulator for verbose source and embedded line metrics."""

    source_total: int = 0
    symbol_chunks: _VerboseLineKindAccumulator = field(default_factory=_VerboseLineKindAccumulator)
    graph_edges: _VerboseLineKindAccumulator = field(default_factory=_VerboseLineKindAccumulator)
    overall_ranges_by_path: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def add_source_lines(self, line_count: int) -> None:
        """Add physical source lines for one file in scope."""
        self.source_total += max(0, int(line_count))

    def add_symbol_chunks(self, chunks: Iterable[SymbolChunk]) -> None:
        """Record line coverage from embedded symbol chunks only."""
        for chunk in chunks:
            if chunk.embedding_vector is None:
                continue
            start_line, end_line = self.symbol_chunks.add_span(
                chunk.file_path,
                chunk.start_line,
                chunk.end_line,
            )
            self.overall_ranges_by_path.setdefault(chunk.file_path, []).append(
                (start_line, end_line)
            )

    def add_graph_edges(self, edges: Iterable[EdgeRecord]) -> None:
        """Record line coverage from embedded graph-edge vectors only."""
        for edge in edges:
            if edge.embedding_vector is None:
                continue
            line_number = max(1, int(edge.line))
            start_line, end_line = self.graph_edges.add_span(
                edge.file_path,
                line_number,
                line_number,
            )
            self.overall_ranges_by_path.setdefault(edge.file_path, []).append(
                (start_line, end_line)
            )

    def build(self) -> VerboseLineMetrics:
        """Freeze the accumulator into a payload-ready metrics object."""
        symbol_metrics = self.symbol_chunks.build()
        edge_metrics = self.graph_edges.build()
        return VerboseLineMetrics(
            source_total=self.source_total,
            embedded_total=symbol_metrics.line_total + edge_metrics.line_total,
            embedded_unique=_merged_line_count(self.overall_ranges_by_path),
            symbol_chunks=symbol_metrics,
            graph_edges=edge_metrics,
        )


@dataclass(frozen=True)
class FileIndexOutcome:
    """Per-file terminal outcome used to classify index runs."""

    status: str
    symbols_indexed: int = 0
    symbols_added: int = 0
    symbols_updated: int = 0
    symbols_removed: int = 0
    source_line_count: int = 0
    reason: str | None = None
    detail: str | None = None
    retained_chunks: tuple[SymbolChunk, ...] = field(default_factory=tuple)
    retained_edges: tuple[EdgeRecord, ...] = field(default_factory=tuple)


@dataclass
class FileIndexExecution:
    """Detailed single-file execution state used by the CLI index command."""

    outcome: FileIndexOutcome
    prepared: PreparedFileIndex | None = None
    timing: FileTimingTrace | None = None
    verbose_lines: VerboseLineMetrics | None = None


@dataclass
class PreparedFileIndex:
    """Prepared file payload carried from parse phase into embedding/persist phase."""

    path: str
    language: str | None
    content_hash: str
    source: str
    size_bytes: int
    repo_id: str
    commit: str
    snapshot: ParsedFileSnapshot
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
    timing: FileTimingTrace


class Indexer:
    """Incremental repository indexer that hashes files and stores symbols."""

    def __init__(
        self,
        config: GloggurConfig,
        cache: CacheManager | None = None,
        parser_registry: ParserRegistry | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        embedding_ledger: EmbeddingLedger | None = None,
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
        self.embedding_ledger = embedding_ledger or EmbeddingLedger(config.cache_dir)
        self._progress_callback: Callable[[int, int], None] | None = None
        self._extract_progress_callback: Callable[[dict[str, object] | None], None] | None = None
        self._scan_callback: Callable[[int, int, str], None] | None = None
        self._stage_callback: Callable[[str], None] | None = None
        self._allow_partial_failures = False
        self._commit_cache: dict[str, str] = {}

    @staticmethod
    def _integrity_status(
        *,
        name: str,
        status: str,
        reason_codes: list[str] | None = None,
        detail: str | None = None,
    ) -> dict[str, object]:
        """Build a normalized integrity marker payload."""
        return {
            "name": name,
            "status": status,
            "reason_codes": list(reason_codes or []),
            "detail": detail,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def _extract_timeout_seconds(self) -> float:
        """Return the configured extract-symbols watchdog timeout."""
        try:
            timeout_seconds = float(
                getattr(self.config, "extract_symbols_timeout_seconds", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            timeout_seconds = 0.0
        return max(0.0, timeout_seconds)

    def _can_use_extract_worker(self) -> bool:
        """Return whether the current indexer wiring is safe to reproduce in a spawned worker."""
        if self._extract_timeout_seconds() <= 0:
            return False
        if type(self) is not Indexer:
            return False
        if type(self.parser_registry) is not ParserRegistry:
            return False
        if TreeSitterParser.extract_symbols is not _TREE_SITTER_EXTRACT_SYMBOLS_IMPL:
            return False
        prepare_impl = getattr(
            self._prepare_file_for_index,
            "__func__",
            self._prepare_file_for_index,
        )
        if prepare_impl is not Indexer._prepare_file_for_index:
            return False
        build_chunks_impl = getattr(
            self._build_symbol_chunks,
            "__func__",
            self._build_symbol_chunks,
        )
        if build_chunks_impl is not Indexer._build_symbol_chunks:
            return False
        build_edges_impl = getattr(self._build_graph_edges, "__func__", self._build_graph_edges)
        if build_edges_impl is not Indexer._build_graph_edges:
            return False
        return True

    def _build_extract_progress(
        self,
        *,
        path: str,
        subphase: str,
        files_done: int,
        files_total: int,
        started_at: str | None = None,
    ) -> dict[str, object]:
        """Build one normalized extract-progress payload."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            "current_file": path,
            "subphase": subphase,
            "files_done": max(0, int(files_done)),
            "files_total": max(0, int(files_total)),
            "started_at": started_at or timestamp,
            "updated_at": timestamp,
        }

    def _publish_extract_progress(self, progress: dict[str, object] | None) -> None:
        """Send extract-progress updates to the active CLI callback when configured."""
        if self._extract_progress_callback is None:
            return
        if progress is None:
            self._extract_progress_callback(None)
            return
        self._extract_progress_callback(dict(progress))

    @staticmethod
    def _stat_file_details(path: str) -> tuple[int | None, int | None]:
        """Return one file's mtime_ns and size_bytes without failing the scan path."""
        try:
            stat_result = os.stat(path)
        except OSError:
            return None, None
        mtime_ns = int(
            getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))
        )
        return mtime_ns, int(getattr(stat_result, "st_size", 0))

    @staticmethod
    def _checkpoint_matches_stat(
        checkpoint: BuildFileCheckpoint,
        *,
        mtime_ns: int | None,
        size_bytes: int | None,
    ) -> bool:
        """Return whether one checkpoint summary still matches a live file stat fingerprint."""
        return (
            checkpoint.mtime_ns is not None
            and checkpoint.size_bytes is not None
            and mtime_ns is not None
            and size_bytes is not None
            and checkpoint.mtime_ns == mtime_ns
            and checkpoint.size_bytes == size_bytes
        )

    @staticmethod
    def _build_checkpoint_snapshot(
        *,
        checkpoint: BuildFileCheckpoint,
        symbols: list[Symbol],
    ) -> ParsedFileSnapshot:
        """Build a minimal snapshot for resumed files loaded from canonical staged rows."""
        return ParsedFileSnapshot(
            path=checkpoint.path,
            source="",
            content_hash=checkpoint.content_hash,
            mtime_ns=checkpoint.mtime_ns,
            language=checkpoint.language,
            symbols=[symbol.model_copy(deep=True) for symbol in symbols],
            span_index=LineByteSpanIndex.from_bytes(b""),
        )

    def _build_prepared_file_from_checkpoint(
        self,
        checkpoint: BuildFileCheckpoint,
        *,
        metadata: FileMetadata,
    ) -> PreparedFileIndex:
        """Rehydrate one checkpoint summary from canonical staged rows for embed resume."""
        path = checkpoint.path
        symbols = self.cache.list_symbols_for_file(path)
        chunks = self.cache.list_chunks_for_file(path)
        edges = self.cache.list_edges_for_file(path)
        repo_id = ""
        commit = ""
        for collection in (symbols, chunks, edges):
            for item in collection:
                repo_id = str(getattr(item, "repo_id", "") or repo_id)
                commit = str(getattr(item, "commit", "") or commit)
                if repo_id and commit:
                    break
            if repo_id and commit:
                break
        if not repo_id:
            repo_id = self._repo_id(path)
        if not commit:
            commit = self._resolve_commit(path)
        current_symbols = [symbol.model_copy(deep=True) for symbol in symbols]
        current_chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        current_edges = [edge.model_copy(deep=True) for edge in edges]
        previous_chunks = [chunk.model_copy(deep=True) for chunk in current_chunks]
        previous_symbols = [symbol.model_copy(deep=True) for symbol in current_symbols]
        previous_edges = [edge.model_copy(deep=True) for edge in current_edges]
        return PreparedFileIndex(
            path=path,
            language=metadata.language or checkpoint.language,
            content_hash=metadata.content_hash,
            source="",
            size_bytes=checkpoint.size_bytes or 0,
            repo_id=repo_id,
            commit=commit,
            snapshot=self._build_checkpoint_snapshot(checkpoint=checkpoint, symbols=current_symbols),
            symbols=current_symbols,
            chunks=current_chunks,
            edges=current_edges,
            previous_symbol_ids=list(metadata.symbols),
            previous_chunk_ids=[chunk.chunk_id for chunk in previous_chunks],
            previous_metadata=metadata.model_copy(deep=True),
            previous_symbols=previous_symbols,
            previous_chunks=previous_chunks,
            previous_edges=previous_edges,
            symbols_added=checkpoint.symbols_added,
            symbols_updated=checkpoint.symbols_updated,
            symbols_removed=checkpoint.symbols_removed,
            timing=FileTimingTrace(
                path=path,
                status="prepared",
                symbol_count=checkpoint.symbol_count,
                chunk_count=checkpoint.chunk_count,
            ),
        )

    def _persist_extract_checkpoint(self, prepared: PreparedFileIndex) -> BuildFileCheckpoint:
        """Persist one extract-complete file into canonical staged rows plus checkpoint summary."""
        self._persist_prepared_file(prepared, update_vectors=False)
        return self.cache.upsert_build_file_checkpoint(
            path=prepared.path,
            state=BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE,
            content_hash=prepared.content_hash,
            mtime_ns=prepared.snapshot.mtime_ns,
            size_bytes=prepared.size_bytes,
            language=prepared.language,
            symbol_count=len(prepared.symbols),
            chunk_count=len(prepared.chunks),
            edge_count=len(prepared.edges),
            symbols_added=prepared.symbols_added,
            symbols_updated=prepared.symbols_updated,
            symbols_removed=prepared.symbols_removed,
        )

    def _load_checkpoint_prepared_files(
        self,
        *,
        states: Iterable[str],
    ) -> list[PreparedFileIndex]:
        """Return checkpointed prepared files loaded from canonical staged rows."""
        prepared_files: list[PreparedFileIndex] = []
        for checkpoint in self.cache.list_build_file_checkpoints(states=states):
            metadata = self.cache.get_file_metadata(checkpoint.path)
            if metadata is None or metadata.content_hash != checkpoint.content_hash:
                self.cache.delete_build_file_checkpoint(checkpoint.path)
                continue
            prepared_files.append(
                self._build_prepared_file_from_checkpoint(checkpoint, metadata=metadata)
            )
        return prepared_files

    @staticmethod
    def _extract_timeout_detail(
        *,
        path: str,
        subphase: str,
        timeout_seconds: float,
        worker_pid: int | None,
    ) -> str:
        """Build a stable timeout detail string for failed_samples and debug payloads."""
        detail = f"{subphase} timed out after {timeout_seconds:.3f}s while processing {path}"
        if worker_pid is not None:
            detail += f" (worker_pid={worker_pid})"
        return detail

    @staticmethod
    def _source_line_count(path: str) -> int:
        """Best-effort line-count lookup for timeout payloads."""
        try:
            with open(path, "rb") as handle:
                return LineByteSpanIndex.from_bytes(handle.read()).line_count
        except OSError:
            return 0

    def _prepare_file_for_index_with_watchdog(
        self,
        *,
        worker: _ExtractSymbolsWorker | None,
        path: str,
        existing_content_hash: str | None,
        capture_verbose_metrics: bool,
        files_done: int,
        files_total: int,
    ) -> tuple[PreparedFileIndex | None, FileIndexOutcome, FileTimingTrace, bool]:
        """Run prepare_file inline or in the watchdog worker."""
        progress = self._build_extract_progress(
            path=path,
            subphase="prepare_file",
            files_done=files_done,
            files_total=files_total,
        )
        self._publish_extract_progress(progress)
        if worker is None:
            prepared, outcome, timing = self._prepare_file_for_index(
                path,
                existing_content_hash=existing_content_hash,
                capture_verbose_metrics=capture_verbose_metrics,
            )
            return prepared, outcome, timing, False

        next_heartbeat = time.monotonic() + _EXTRACT_PROGRESS_HEARTBEAT_SECONDS
        progress_started_at = str(progress["started_at"])
        started = time.perf_counter()

        def _heartbeat() -> None:
            """Refresh persisted progress while the worker is still alive."""
            nonlocal next_heartbeat, progress
            if time.monotonic() < next_heartbeat:
                return
            progress = self._build_extract_progress(
                path=path,
                subphase="prepare_file",
                files_done=files_done,
                files_total=files_total,
                started_at=progress_started_at,
            )
            self._publish_extract_progress(progress)
            next_heartbeat = time.monotonic() + _EXTRACT_PROGRESS_HEARTBEAT_SECONDS

        try:
            response = worker.run_job(
                kind="prepare_file",
                payload={
                    "path": path,
                    "existing_content_hash": existing_content_hash,
                    "capture_verbose_metrics": capture_verbose_metrics,
                },
                timeout_seconds=self._extract_timeout_seconds(),
                on_poll=_heartbeat,
            )
        except _ExtractWorkerTimeout as exc:
            detail = self._extract_timeout_detail(
                path=path,
                subphase="prepare_file",
                timeout_seconds=exc.timeout_seconds,
                worker_pid=exc.worker_pid,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return (
                None,
                FileIndexOutcome(
                    status="failed",
                    source_line_count=self._source_line_count(path),
                    reason=EXTRACT_SYMBOLS_TIMEOUT_REASON,
                    detail=detail,
                ),
                FileTimingTrace(
                    path=path,
                    status="failed",
                    parse_ms=elapsed_ms,
                    reason=EXTRACT_SYMBOLS_TIMEOUT_REASON,
                    detail=detail,
                ),
                True,
            )
        prepared = response.get("prepared")
        outcome = response.get("outcome")
        timing = response.get("timing")
        if not isinstance(outcome, FileIndexOutcome) or not isinstance(timing, FileTimingTrace):
            raise _ExtractWorkerError("prepare_file worker returned an invalid payload")
        if prepared is not None and not isinstance(prepared, PreparedFileIndex):
            raise _ExtractWorkerError("prepare_file worker returned an invalid prepared payload")
        return prepared, outcome, timing, False

    def _build_edges_with_watchdog(
        self,
        *,
        worker: _ExtractSymbolsWorker | None,
        prepared: PreparedFileIndex,
        candidate_symbols: list[Symbol],
        files_done: int,
        files_total: int,
    ) -> tuple[list[EdgeRecord] | None, int, bool, str | None]:
        """Run build_edges inline or in the watchdog worker."""
        progress = self._build_extract_progress(
            path=prepared.path,
            subphase="build_edges",
            files_done=files_done,
            files_total=files_total,
        )
        self._publish_extract_progress(progress)
        started = time.perf_counter()
        if worker is None:
            edges = self._build_graph_edges(
                path=prepared.path,
                source=prepared.source,
                symbols=prepared.symbols,
                candidate_symbols=candidate_symbols,
                repo_id=prepared.repo_id,
                commit=prepared.commit,
                language=prepared.language or "unknown",
                include_text=self.config.embed_graph_edges,
            )
            return edges, int((time.perf_counter() - started) * 1000), False, None

        next_heartbeat = time.monotonic() + _EXTRACT_PROGRESS_HEARTBEAT_SECONDS
        progress_started_at = str(progress["started_at"])

        def _heartbeat() -> None:
            """Refresh persisted edge-build progress while the worker is still alive."""
            nonlocal next_heartbeat, progress
            if time.monotonic() < next_heartbeat:
                return
            progress = self._build_extract_progress(
                path=prepared.path,
                subphase="build_edges",
                files_done=files_done,
                files_total=files_total,
                started_at=progress_started_at,
            )
            self._publish_extract_progress(progress)
            next_heartbeat = time.monotonic() + _EXTRACT_PROGRESS_HEARTBEAT_SECONDS

        try:
            response = worker.run_job(
                kind="build_edges",
                payload={
                    "path": prepared.path,
                    "source": prepared.source,
                    "symbols": prepared.symbols,
                    "candidate_symbols": candidate_symbols,
                    "repo_id": prepared.repo_id,
                    "commit": prepared.commit,
                    "language": prepared.language or "unknown",
                    "include_text": self.config.embed_graph_edges,
                },
                timeout_seconds=self._extract_timeout_seconds(),
                on_poll=_heartbeat,
            )
        except _ExtractWorkerTimeout as exc:
            detail = self._extract_timeout_detail(
                path=prepared.path,
                subphase="build_edges",
                timeout_seconds=exc.timeout_seconds,
                worker_pid=exc.worker_pid,
            )
            return None, int((time.perf_counter() - started) * 1000), True, detail
        edges = response.get("edges")
        if not isinstance(edges, list):
            raise _ExtractWorkerError("build_edges worker returned an invalid payload")
        return list(edges), int((time.perf_counter() - started) * 1000), False, None

    def index_repository(self, path: str, *, capture_verbose_metrics: bool = False) -> IndexResult:
        """Index all supported files under a repository root."""

        start = time.perf_counter()
        self.cache.delete_index_metadata()
        # Test-only hook to make interruption windows deterministic.
        self._maybe_pause_after_metadata_delete()

        if self._stage_callback is not None:
            self._stage_callback("scan_source")
        scan_source_started = time.perf_counter()
        source_files = list(self._iter_source_files(path))
        scan_source_ms = int((time.perf_counter() - scan_source_started) * 1000)
        checkpoint_by_path = {
            checkpoint.path: checkpoint for checkpoint in self.cache.list_build_file_checkpoints()
        }
        deleted_checkpoint_paths = sorted(set(checkpoint_by_path) - set(source_files))
        for deleted_checkpoint_path in deleted_checkpoint_paths:
            self.cache.delete_build_file_checkpoint(deleted_checkpoint_path)
            checkpoint_by_path.pop(deleted_checkpoint_path, None)
        cached_file_hashes = self.cache.list_file_hashes()
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
        parsed_files: list[ParsedFileSnapshot] = []
        seen_paths: set[str] = set()
        catalog_excluded_paths: set[str] = set(deleted_checkpoint_paths)
        prepared_candidates: list[PreparedFileIndex] = []
        file_timings: list[FileTimingTrace] = []
        file_timings_by_path: dict[str, FileTimingTrace] = {}
        verbose_lines = VerboseLineMetricsAccumulator() if capture_verbose_metrics else None
        chunk_integrity = self._integrity_status(
            name="chunk_span",
            status="passed",
            detail="chunk/span integrity checks passed",
        )
        terminal_extract_files = 0
        extract_timeout_halted = False
        timed_out_symbol_ids: set[str] = set()
        worker = _ExtractSymbolsWorker(self.config) if self._can_use_extract_worker() else None

        try:
            if self._stage_callback is not None:
                self._stage_callback("extract_symbols")
            parse_phase_started = time.perf_counter()
            for file_path in source_files:
                seen_paths.add(file_path)
                files_considered += 1
                checkpoint = checkpoint_by_path.get(file_path)
                if checkpoint is not None:
                    stat_mtime_ns, stat_size_bytes = self._stat_file_details(file_path)
                    if self._checkpoint_matches_stat(
                        checkpoint,
                        mtime_ns=stat_mtime_ns,
                        size_bytes=stat_size_bytes,
                    ):
                        metadata = self.cache.get_file_metadata(file_path)
                        if metadata is not None and metadata.content_hash == checkpoint.content_hash:
                            checkpoint_timing = FileTimingTrace(
                                path=file_path,
                                status=(
                                    "unchanged"
                                    if checkpoint.state == BUILD_FILE_CHECKPOINT_STATE_EMBEDDED_COMPLETE
                                    else "prepared"
                                ),
                                symbol_count=checkpoint.symbol_count,
                                chunk_count=checkpoint.chunk_count,
                            )
                            file_timings.append(checkpoint_timing)
                            file_timings_by_path[file_path] = checkpoint_timing
                            if verbose_lines is not None:
                                verbose_lines.add_source_lines(self._source_line_count(file_path))
                            terminal_extract_files += 1
                            if checkpoint.state == BUILD_FILE_CHECKPOINT_STATE_EMBEDDED_COMPLETE:
                                unchanged_files += 1
                                if verbose_lines is not None:
                                    verbose_lines.add_symbol_chunks(
                                        self.cache.list_chunks_for_file(file_path)
                                    )
                                    verbose_lines.add_graph_edges(
                                        self.cache.list_edges_for_file(file_path)
                                    )
                                if self._scan_callback is not None:
                                    self._scan_callback(files_considered, total_files, "unchanged")
                                continue
                            if self._scan_callback is not None:
                                self._scan_callback(files_considered, total_files, "prepared")
                            continue
                    self.cache.delete_build_file_checkpoint(file_path)
                    checkpoint_by_path.pop(file_path, None)
                prepared, outcome, timing, timed_out = self._prepare_file_for_index_with_watchdog(
                    worker=worker,
                    path=file_path,
                    existing_content_hash=cached_file_hashes.get(file_path),
                    capture_verbose_metrics=capture_verbose_metrics,
                    files_done=terminal_extract_files,
                    files_total=total_files,
                )
                file_timings.append(timing)
                file_timings_by_path[file_path] = timing
                if verbose_lines is not None:
                    verbose_lines.add_source_lines(
                        prepared.snapshot.span_index.line_count
                        if prepared is not None
                        else outcome.source_line_count
                    )
                if prepared is not None:
                    prepared_candidates.append(prepared)
                    parsed_files.append(prepared.snapshot)
                    catalog_excluded_paths.add(file_path)
                    if self._scan_callback is not None:
                        self._scan_callback(files_considered, total_files, "prepared")
                    continue
                terminal_extract_files += 1
                if self._scan_callback is not None:
                    self._scan_callback(files_considered, total_files, outcome.status)
                if outcome.status == "unchanged":
                    unchanged_files += 1
                    if verbose_lines is not None:
                        verbose_lines.add_symbol_chunks(outcome.retained_chunks)
                        verbose_lines.add_graph_edges(outcome.retained_edges)
                    continue

                catalog_excluded_paths.add(file_path)
                failed_files += 1
                reason = outcome.reason or "indexing_error"
                failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                if reason == "chunk_span_integrity_error":
                    chunk_integrity = self._integrity_status(
                        name="chunk_span",
                        status="failed",
                        reason_codes=[reason],
                        detail=outcome.detail,
                    )
                if len(failed_samples) < 5:
                    if outcome.detail:
                        failed_samples.append(f"{file_path}: {outcome.detail}")
                    else:
                        failed_samples.append(file_path)
                if timed_out:
                    if worker is not None and self._allow_partial_failures:
                        worker.restart()
                        continue
                    extract_timeout_halted = True
                    break
            parse_phase_ms = int((time.perf_counter() - parse_phase_started) * 1000)

            prepared_files: list[PreparedFileIndex] = []
            edge_phase_started = time.perf_counter()
            if not extract_timeout_halted and prepared_candidates:
                repo_symbol_catalog = self._build_repo_symbol_catalog(
                    prepared_candidates,
                    excluded_paths=catalog_excluded_paths,
                )
                for prepared in prepared_candidates:
                    edges, edge_ms, timed_out, timeout_detail = self._build_edges_with_watchdog(
                        worker=worker,
                        prepared=prepared,
                        candidate_symbols=repo_symbol_catalog,
                        files_done=terminal_extract_files + len(prepared_files),
                        files_total=total_files,
                    )
                    prepared.timing.edge_ms = edge_ms
                    if timed_out:
                        prepared.timing.status = "failed"
                        prepared.timing.reason = EXTRACT_SYMBOLS_TIMEOUT_REASON
                        prepared.timing.detail = timeout_detail
                        failed_files += 1
                        failed_reasons[EXTRACT_SYMBOLS_TIMEOUT_REASON] = (
                            failed_reasons.get(EXTRACT_SYMBOLS_TIMEOUT_REASON, 0) + 1
                        )
                        if len(failed_samples) < 5:
                            failed_samples.append(
                                f"{prepared.path}: "
                                f"{timeout_detail or EXTRACT_SYMBOLS_TIMEOUT_REASON}"
                            )
                        timed_out_symbol_ids.update(
                            symbol.id for symbol in prepared.symbols if symbol.id
                        )
                        terminal_extract_files += 1
                        if worker is not None and self._allow_partial_failures:
                            worker.restart()
                            continue
                        extract_timeout_halted = True
                        break
                    prepared.edges = edges or []
                    try:
                        self._persist_extract_checkpoint(prepared)
                    except Exception as exc:
                        prepared.timing.status = "failed"
                        prepared.timing.reason = "storage_error"
                        prepared.timing.detail = f"{type(exc).__name__}: {exc}"
                        failed_files += 1
                        failed_reasons["storage_error"] = failed_reasons.get("storage_error", 0) + 1
                        if len(failed_samples) < 5:
                            failed_samples.append(f"{prepared.path}: {type(exc).__name__}: {exc}")
                        extract_timeout_halted = True
                        break
                    prepared_files.append(prepared)
                    terminal_extract_files += 1
                if timed_out_symbol_ids:
                    for prepared in prepared_files:
                        filtered_edges = [
                            edge
                            for edge in prepared.edges
                            if edge.from_id not in timed_out_symbol_ids
                            and edge.to_id not in timed_out_symbol_ids
                        ]
                        if len(filtered_edges) == len(prepared.edges):
                            continue
                        prepared.edges = filtered_edges
                        try:
                            self._persist_extract_checkpoint(prepared)
                        except Exception as exc:
                            prepared.timing.status = "failed"
                            prepared.timing.reason = "storage_error"
                            prepared.timing.detail = f"{type(exc).__name__}: {exc}"
                            failed_files += 1
                            failed_reasons["storage_error"] = (
                                failed_reasons.get("storage_error", 0) + 1
                            )
                            if len(failed_samples) < 5:
                                failed_samples.append(
                                    f"{prepared.path}: {type(exc).__name__}: {exc}"
                                )
                            extract_timeout_halted = True
                            break
            edge_phase_ms = int((time.perf_counter() - edge_phase_started) * 1000)
            extract_symbols_ms = parse_phase_ms + edge_phase_ms

            if extract_timeout_halted:
                duration_ms = int((time.perf_counter() - start) * 1000)
                return IndexResult(
                    files_considered=files_considered,
                    indexed=0,
                    unchanged=unchanged_files,
                    failed=failed_files,
                    indexed_symbols=0,
                    duration_ms=duration_ms,
                    files_changed=0,
                    files_removed=0,
                    symbols_added=0,
                    symbols_updated=0,
                    symbols_removed=0,
                    failed_reasons=failed_reasons,
                    failed_samples=failed_samples,
                    phase_timings_ms={
                        "scan_source": scan_source_ms,
                        "extract_symbols": extract_symbols_ms,
                        "embed_chunks": 0,
                        "persist_cache": 0,
                        "validate_integrity": 0,
                        "parse": parse_phase_ms,
                        "edge": edge_phase_ms,
                        "embed_persist": 0,
                        "cleanup": 0,
                        "consistency_checks": 0,
                    },
                    index_stats=self.cache.get_index_stats(),
                    file_timings=file_timings,
                    parsed_files=parsed_files,
                    source_files=source_files,
                    verbose_lines=verbose_lines.build() if verbose_lines is not None else None,
                )
        finally:
            if worker is not None:
                worker.close()

        prepared_files = self._load_checkpoint_prepared_files(
            states=[BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE]
        )
        for prepared in prepared_files:
            timing = file_timings_by_path.get(prepared.path)
            if timing is not None:
                prepared.timing = timing
        total_embedding_symbols = sum(len(prepared.chunks) for prepared in prepared_files)
        embedded_symbols_done = 0
        if self._stage_callback is not None:
            self._stage_callback("embed_chunks")
        embed_persist_started = time.perf_counter()
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
                embed_started = time.perf_counter()
                prepared.chunks = self._apply_embeddings(
                    prepared.chunks,
                    progress_callback=progress_callback,
                )
                prepared.edges = self._apply_edge_embeddings(prepared.edges)
                prepared.timing.embed_ms = int((time.perf_counter() - embed_started) * 1000)
                persist_started = time.perf_counter()
                self._persist_prepared_file(prepared)
                self.cache.mark_build_file_checkpoint_embedded(prepared.path)
                prepared.timing.persist_ms = int((time.perf_counter() - persist_started) * 1000)
            except Exception as exc:
                failed_files += 1
                reason = (
                    "embedding_provider_error"
                    if isinstance(exc, EmbeddingProviderError)
                    else "storage_error"
                )
                prepared.timing.status = "failed"
                prepared.timing.reason = reason
                prepared.timing.detail = f"{type(exc).__name__}: {exc}"
                failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                if len(failed_samples) < 5:
                    failed_samples.append(f"{prepared.path}: {type(exc).__name__}: {exc}")
                embedded_symbols_done += file_progress_done
                continue

            embedded_symbols_done += file_symbols_total
            prepared.timing.status = "indexed"
            indexed_files += 1
            indexed_symbols += len(prepared.symbols)
            symbols_added += prepared.symbols_added
            symbols_updated += prepared.symbols_updated
            symbols_removed += prepared.symbols_removed
            if verbose_lines is not None:
                verbose_lines.add_symbol_chunks(prepared.chunks)
                verbose_lines.add_graph_edges(prepared.edges)
        embed_persist_ms = int((time.perf_counter() - embed_persist_started) * 1000)
        embed_chunks_ms = sum(timing.embed_ms for timing in file_timings)

        if self._stage_callback is not None:
            self._stage_callback("persist_cache")
        cleanup_started = time.perf_counter()
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
        cleanup_ms = int((time.perf_counter() - cleanup_started) * 1000)
        persist_cache_ms = sum(timing.persist_ms for timing in file_timings) + cleanup_ms
        if self._stage_callback is not None:
            self._stage_callback("validate_integrity")
        consistency_started = time.perf_counter()
        consistency = self._validate_vector_metadata_consistency()
        failed_files += consistency["failed"]
        for reason, count in consistency["failed_reasons"].items():
            failed_reasons[reason] = failed_reasons.get(reason, 0) + count
        for sample in consistency["failed_samples"]:
            if len(failed_samples) >= 5:
                break
            failed_samples.append(sample)
        vector_integrity = consistency.get("integrity")
        if not isinstance(vector_integrity, dict):
            vector_integrity = self._integrity_status(
                name="vector_cache",
                status="missing",
                reason_codes=["vector_integrity_missing"],
                detail="vector/cache integrity marker missing",
            )
        self.cache.set_search_integrity(
            {
                "vector_cache": vector_integrity,
                "chunk_span": chunk_integrity,
            }
        )
        consistency_ms = int((time.perf_counter() - consistency_started) * 1000)

        if failed_files == 0:
            metadata = IndexMetadata(
                version=self.config.index_version,
                total_symbols=self.cache.count_symbols(),
                indexed_files=self.cache.count_files(),
            )
            self.cache.set_index_metadata(metadata)
            self.cache.set_index_profile(self.config.embedding_profile())

        duration_ms = int((time.perf_counter() - start) * 1000)
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
            phase_timings_ms={
                "scan_source": scan_source_ms,
                "extract_symbols": extract_symbols_ms,
                "embed_chunks": embed_chunks_ms,
                "persist_cache": persist_cache_ms,
                "validate_integrity": consistency_ms,
                "parse": parse_phase_ms,
                "edge": edge_phase_ms,
                "embed_persist": embed_persist_ms,
                "cleanup": cleanup_ms,
                "consistency_checks": consistency_ms,
            },
            index_stats=self.cache.get_index_stats(),
            file_timings=file_timings,
            parsed_files=parsed_files,
            source_files=source_files,
            verbose_lines=verbose_lines.build() if verbose_lines is not None else None,
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

    def index_file_with_details(
        self,
        path: str,
        *,
        capture_verbose_metrics: bool = False,
    ) -> FileIndexExecution:
        """Index a file and return outcome plus shared parse/timing state."""

        existing = self.cache.get_file_metadata(path)
        worker = _ExtractSymbolsWorker(self.config) if self._can_use_extract_worker() else None
        try:
            prepared, outcome, timing, _timed_out = self._prepare_file_for_index_with_watchdog(
                worker=worker,
                path=path,
                existing_content_hash=existing.content_hash if existing else None,
                capture_verbose_metrics=capture_verbose_metrics,
                files_done=0,
                files_total=1,
            )
            verbose_lines = VerboseLineMetricsAccumulator() if capture_verbose_metrics else None
            if verbose_lines is not None:
                verbose_lines.add_source_lines(
                    prepared.snapshot.span_index.line_count
                    if prepared is not None
                    else outcome.source_line_count
                )
            if prepared is None:
                if verbose_lines is not None:
                    verbose_lines.add_symbol_chunks(outcome.retained_chunks)
                    verbose_lines.add_graph_edges(outcome.retained_edges)
                return FileIndexExecution(
                    outcome=outcome,
                    timing=timing,
                    verbose_lines=verbose_lines.build() if verbose_lines is not None else None,
                )

            candidate_symbols = [
                symbol for symbol in self.cache.list_symbols() if symbol.file_path != prepared.path
            ]
            candidate_symbols.extend(prepared.symbols)
            edges, edge_ms, timed_out, timeout_detail = self._build_edges_with_watchdog(
                worker=worker,
                prepared=prepared,
                candidate_symbols=candidate_symbols,
                files_done=0,
                files_total=1,
            )
            timing.edge_ms = edge_ms
            if timed_out:
                timing.status = "failed"
                timing.reason = EXTRACT_SYMBOLS_TIMEOUT_REASON
                timing.detail = timeout_detail
                return FileIndexExecution(
                    outcome=FileIndexOutcome(
                        status="failed",
                        source_line_count=prepared.snapshot.span_index.line_count,
                        reason=EXTRACT_SYMBOLS_TIMEOUT_REASON,
                        detail=timeout_detail,
                    ),
                    timing=timing,
                    verbose_lines=verbose_lines.build() if verbose_lines is not None else None,
                )
            prepared.edges = edges or []

            try:
                embed_started = time.perf_counter()
                prepared.chunks = self._apply_embeddings(
                    prepared.chunks,
                    progress_callback=self._progress_callback,
                )
                prepared.edges = self._apply_edge_embeddings(prepared.edges)
                timing.embed_ms = int((time.perf_counter() - embed_started) * 1000)
                persist_started = time.perf_counter()
                self._persist_prepared_file(prepared)
                timing.persist_ms = int((time.perf_counter() - persist_started) * 1000)
            except Exception as exc:
                reason = (
                    "embedding_provider_error"
                    if isinstance(exc, EmbeddingProviderError)
                    else "storage_error"
                )
                timing.status = "failed"
                timing.reason = reason
                timing.detail = f"{type(exc).__name__}: {exc}"
                return FileIndexExecution(
                    outcome=FileIndexOutcome(
                        status="failed",
                        source_line_count=prepared.snapshot.span_index.line_count,
                        reason=reason,
                        detail=f"{type(exc).__name__}: {exc}",
                    ),
                    prepared=prepared,
                    timing=timing,
                    verbose_lines=verbose_lines.build() if verbose_lines is not None else None,
                )

            timing.status = "indexed"
            if verbose_lines is not None:
                verbose_lines.add_symbol_chunks(prepared.chunks)
                verbose_lines.add_graph_edges(prepared.edges)
            return FileIndexExecution(
                outcome=FileIndexOutcome(
                    status="indexed",
                    symbols_indexed=len(prepared.symbols),
                    symbols_added=prepared.symbols_added,
                    symbols_updated=prepared.symbols_updated,
                    symbols_removed=prepared.symbols_removed,
                    source_line_count=prepared.snapshot.span_index.line_count,
                ),
                prepared=prepared,
                timing=timing,
                verbose_lines=verbose_lines.build() if verbose_lines is not None else None,
            )
        finally:
            if worker is not None:
                worker.close()

    def index_file_with_outcome(self, path: str) -> FileIndexOutcome:
        """Index a file and return an explicit terminal outcome."""
        return self.index_file_with_details(path).outcome

    def _prepare_file_for_index(
        self,
        path: str,
        *,
        existing_content_hash: str | None = None,
        capture_verbose_metrics: bool = False,
    ) -> tuple[PreparedFileIndex | None, FileIndexOutcome, FileTimingTrace]:
        """Read/parse a file and compute symbol diff metadata for later persistence."""

        started = time.perf_counter()
        try:
            with open(path, "rb") as handle:
                raw_source = handle.read()
        except OSError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            return (
                None,
                FileIndexOutcome(
                    status="failed",
                    source_line_count=0,
                    reason="read_error",
                    detail=detail,
                ),
                FileTimingTrace(
                    path=path,
                    status="failed",
                    parse_ms=int((time.perf_counter() - started) * 1000),
                    reason="read_error",
                    detail=detail,
                ),
            )
        span_index = LineByteSpanIndex.from_bytes(raw_source)
        try:
            source = raw_source.decode("utf8")
        except UnicodeDecodeError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            return (
                None,
                FileIndexOutcome(
                    status="failed",
                    source_line_count=span_index.line_count,
                    reason="decode_error",
                    detail=detail,
                ),
                FileTimingTrace(
                    path=path,
                    status="failed",
                    parse_ms=int((time.perf_counter() - started) * 1000),
                    reason="decode_error",
                    detail=detail,
                ),
            )

        content_hash = self._hash_content(source)
        existing = self.cache.get_file_metadata(path) if existing_content_hash is None else None
        cached_hash = (
            existing_content_hash
            if existing_content_hash is not None
            else (existing.content_hash if existing else None)
        )
        if cached_hash == content_hash:
            retained_chunks: tuple[SymbolChunk, ...] = ()
            retained_edges: tuple[EdgeRecord, ...] = ()
            if capture_verbose_metrics:
                retained_chunks = tuple(self.cache.list_chunks_for_file(path))
                retained_edges = tuple(self.cache.list_edges_for_file(path))
            return (
                None,
                FileIndexOutcome(
                    status="unchanged",
                    source_line_count=span_index.line_count,
                    retained_chunks=retained_chunks,
                    retained_edges=retained_edges,
                ),
                FileTimingTrace(
                    path=path,
                    status="unchanged",
                    parse_ms=int((time.perf_counter() - started) * 1000),
                ),
            )
        previous_symbols: list[Symbol] = []
        previous_chunks: list[SymbolChunk] = []
        previous_edges: list[EdgeRecord] = []
        existing_symbols_by_id: dict[str, Symbol] = {}

        parser_entry = self.parser_registry.get_parser_for_path(path)
        if not parser_entry:
            detail = "No parser registered for file extension."
            return (
                None,
                FileIndexOutcome(
                    status="failed",
                    source_line_count=span_index.line_count,
                    reason="parser_unavailable",
                    detail=detail,
                ),
                FileTimingTrace(
                    path=path,
                    status="failed",
                    parse_ms=int((time.perf_counter() - started) * 1000),
                    reason="parser_unavailable",
                    detail=detail,
                ),
            )

        try:
            symbols = parser_entry.parser.extract_symbols(path, source)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            return (
                None,
                FileIndexOutcome(
                    status="failed",
                    source_line_count=span_index.line_count,
                    reason="parse_error",
                    detail=detail,
                ),
                FileTimingTrace(
                    path=path,
                    status="failed",
                    parse_ms=int((time.perf_counter() - started) * 1000),
                    reason="parse_error",
                    detail=detail,
                ),
            )

        existing = existing or self.cache.get_file_metadata(path)
        if existing:
            previous_symbols = self.cache.list_symbols_for_file(path)
            previous_chunks = self.cache.list_chunks_for_file(path)
            previous_edges = self.cache.list_edges_for_file(path)
            existing_symbols_by_id = {symbol.id: symbol for symbol in previous_symbols}

        repo_id = self._repo_id(path)
        commit = self._resolve_commit(path)
        for symbol in symbols:
            if not symbol.repo_id:
                symbol.repo_id = repo_id
            symbol.commit = commit

        chunks = self._build_symbol_chunks(
            path=path,
            source=source,
            symbols=symbols,
            commit=commit,
            span_index=span_index,
        )
        chunk_integrity_error = self._validate_chunk_span_integrity(symbols=symbols, chunks=chunks)
        if chunk_integrity_error is not None:
            return (
                None,
                FileIndexOutcome(
                    status="failed",
                    source_line_count=span_index.line_count,
                    reason="chunk_span_integrity_error",
                    detail=chunk_integrity_error,
                ),
                FileTimingTrace(
                    path=path,
                    status="failed",
                    parse_ms=int((time.perf_counter() - started) * 1000),
                    symbol_count=len(symbols),
                    chunk_count=len(chunks),
                    reason="chunk_span_integrity_error",
                    detail=chunk_integrity_error,
                ),
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

        snapshot = ParsedFileSnapshot(
            path=path,
            source=source,
            content_hash=content_hash,
            mtime_ns=self._stat_mtime_ns(path),
            language=parser_entry.language,
            symbols=symbols,
            span_index=span_index,
        )
        timing = FileTimingTrace(
            path=path,
            status="prepared",
            parse_ms=int((time.perf_counter() - started) * 1000),
            symbol_count=len(symbols),
            chunk_count=len(chunks),
        )
        prepared = PreparedFileIndex(
            path=path,
            language=parser_entry.language,
            content_hash=content_hash,
            source=source,
            size_bytes=len(raw_source),
            repo_id=repo_id,
            commit=commit,
            snapshot=snapshot,
            symbols=symbols,
            chunks=chunks,
            edges=[],
            previous_symbol_ids=previous_symbol_ids,
            previous_chunk_ids=previous_chunk_ids,
            previous_metadata=existing.model_copy(deep=True) if existing else None,
            previous_symbols=[symbol.model_copy(deep=True) for symbol in previous_symbols],
            previous_chunks=[chunk.model_copy(deep=True) for chunk in previous_chunks],
            previous_edges=[edge.model_copy(deep=True) for edge in previous_edges],
            symbols_added=symbols_added,
            symbols_updated=symbols_updated,
            symbols_removed=symbols_removed,
            timing=timing,
        )
        return (
            prepared,
            FileIndexOutcome(
                status="prepared",
                symbols_indexed=len(symbols),
                source_line_count=span_index.line_count,
            ),
            timing,
        )

    @staticmethod
    def _validate_chunk_span_integrity(
        *,
        symbols: list[Symbol],
        chunks: list[SymbolChunk],
    ) -> str | None:
        """Return first deterministic chunk/span integrity error, if any."""
        symbols_by_id = {symbol.id: symbol for symbol in symbols}
        chunks_by_symbol: dict[str, list[SymbolChunk]] = {}
        for chunk in chunks:
            symbol = symbols_by_id.get(chunk.symbol_id)
            if symbol is None:
                return f"chunk {chunk.chunk_id} references unknown symbol {chunk.symbol_id}"
            if chunk.start_line < symbol.start_line or chunk.end_line > symbol.end_line:
                return (
                    f"chunk {chunk.chunk_id} span {chunk.start_line}-{chunk.end_line} escapes "
                    f"symbol {symbol.id} span {symbol.start_line}-{symbol.end_line}"
                )
            if chunk.end_line < chunk.start_line:
                return (
                    f"chunk {chunk.chunk_id} has descending span "
                    f"{chunk.start_line}-{chunk.end_line}"
                )
            if symbol.end_line >= symbol.start_line and not chunk.text.strip():
                return f"chunk {chunk.chunk_id} has empty text " f"for non-empty symbol {symbol.id}"
            chunks_by_symbol.setdefault(chunk.symbol_id, []).append(chunk)

        ordered_symbols = sorted(
            symbols,
            key=lambda item: (item.start_line, item.end_line, item.id),
        )
        for symbol in ordered_symbols:
            symbol_chunks = sorted(
                chunks_by_symbol.get(symbol.id, []),
                key=lambda item: (
                    item.start_line,
                    item.end_line,
                    item.chunk_part_index,
                    item.chunk_id,
                ),
            )
            previous_end = symbol.start_line - 1
            for chunk in symbol_chunks:
                if chunk.start_line <= previous_end:
                    return (
                        f"chunk {chunk.chunk_id} starts before prior chunk end "
                        f"for symbol {symbol.id} ({chunk.start_line} <= {previous_end})"
                    )
                previous_end = chunk.end_line
        return None

    def _persist_prepared_file(
        self,
        prepared: PreparedFileIndex,
        *,
        update_vectors: bool = True,
    ) -> None:
        """Persist one prepared file into cache and optionally refresh vector rows."""
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
            if update_vectors and self.vector_store and prepared.chunks:
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
        previous_embedded_chunks = [
            chunk for chunk in prepared.previous_chunks if chunk.embedding_vector is not None
        ]
        if previous_embedded_chunks:
            self.vector_store.upsert_vectors(previous_embedded_chunks)

    def index_file(self, path: str) -> int | None:
        """Index a file and return symbol count when indexed, else None."""

        outcome = self.index_file_with_outcome(path)
        if outcome.status == "indexed":
            return outcome.symbols_indexed
        return None

    def _build_repo_symbol_catalog(
        self,
        prepared_files: list[PreparedFileIndex],
        *,
        excluded_paths: Iterable[str] = (),
    ) -> list[Symbol]:
        """Return one per-run symbol catalog reused for edge extraction."""
        changed_paths = {prepared.path for prepared in prepared_files}
        changed_paths.update(path for path in excluded_paths if path)
        catalog = [
            symbol for symbol in self.cache.list_symbols() if symbol.file_path not in changed_paths
        ]
        for prepared in prepared_files:
            catalog.extend(prepared.symbols)
        return catalog

    @staticmethod
    def _stat_mtime_ns(path: str) -> int | None:
        """Return mtime_ns when available without failing the index path."""
        try:
            stat_result = os.stat(path)
        except OSError:
            return None
        return int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))

    def _iter_source_files(self, root: str) -> Iterable[str]:
        """Yield supported source files under a root directory."""

        for current_root, dirs, files in os.walk(root):
            dirs[:] = filter_index_walk_dirs(
                current_root,
                dirs,
                excluded_dirs=self.config.excluded_dirs,
            )
            for filename in files:
                full_path = os.path.join(current_root, filename)
                if self._is_supported_file(full_path):
                    yield full_path

    def _is_supported_file(self, path: str) -> bool:
        """Return whether a file path is in scope for indexing."""

        return is_indexable_source_path(
            path,
            supported_extensions=self.config.supported_extensions,
            excluded_dirs=self.config.excluded_dirs,
            include_minified_js=self.config.include_minified_js,
        )

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
            return {
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
                "integrity": self._integrity_status(
                    name="vector_cache",
                    status="missing",
                    reason_codes=["vector_integrity_missing"],
                    detail=(
                        "vector/cache integrity unavailable without "
                        "embedding provider or vector store"
                    ),
                ),
            }
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
                "integrity": self._integrity_status(
                    name="vector_cache",
                    status="missing",
                    reason_codes=["vector_consistency_unverifiable"],
                    detail="vector store does not expose list_symbol_ids()",
                ),
            }
        try:
            vector_symbol_ids = {str(symbol_id) for symbol_id in list_symbol_ids()}
        except Exception as exc:
            return {
                "failed": 1,
                "failed_reasons": {"vector_metadata_mismatch": 1},
                "failed_samples": [f"vector metadata check failed: {type(exc).__name__}: {exc}"],
                "integrity": self._integrity_status(
                    name="vector_cache",
                    status="failed",
                    reason_codes=["vector_metadata_mismatch"],
                    detail=f"vector metadata check failed: {type(exc).__name__}: {exc}",
                ),
            }

        pending_embed_paths = {
            checkpoint.path
            for checkpoint in self.cache.list_build_file_checkpoints(
                states=[BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE]
            )
        }
        cache_chunks = self.cache.list_chunks()
        pending_embed_chunk_ids = {
            chunk.chunk_id for chunk in cache_chunks if chunk.file_path in pending_embed_paths
        }
        cache_chunk_ids = {
            chunk.chunk_id for chunk in cache_chunks if chunk.file_path not in pending_embed_paths
        }
        vector_symbol_ids.difference_update(pending_embed_chunk_ids)
        missing_vectors = sorted(cache_chunk_ids - vector_symbol_ids)
        stale_vectors = sorted(vector_symbol_ids - cache_chunk_ids)
        if not missing_vectors and not stale_vectors:
            return {
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
                "integrity": self._integrity_status(
                    name="vector_cache",
                    status="passed",
                    detail="vector/cache integrity checks passed",
                ),
            }
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
            "integrity": self._integrity_status(
                name="vector_cache",
                status="failed",
                reason_codes=["vector_metadata_mismatch"],
                detail=sample,
            ),
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
        embedding_profile = self.config.embedding_profile()
        text_hashes = [embedding_text_hash(text) for text in texts]
        cached_vectors = self.embedding_ledger.get_vectors(
            embedding_profile=embedding_profile,
            record_kind="chunk",
            text_hashes=text_hashes,
        )
        chunk_size = getattr(self.embedding_provider, "_chunk_size", 50)
        missing_hashes: dict[str, list[int]] = {}
        missing_items: list[tuple[str, str]] = []
        if progress_callback is not None:
            progress_callback(0, total)
        completed = 0
        chunk_inputs = zip(chunks, text_hashes, texts, strict=True)
        for index, (chunk_row, text_hash, text) in enumerate(chunk_inputs):
            cached_vector = cached_vectors.get(text_hash)
            if cached_vector is not None:
                chunk_row.embedding_vector = list(cached_vector)
                completed += 1
                continue
            positions = missing_hashes.setdefault(text_hash, [])
            positions.append(index)
            if len(positions) == 1:
                missing_items.append((text_hash, text))
        if progress_callback is not None and completed > 0:
            progress_callback(completed, total)
        for i in range(0, len(missing_items), chunk_size):
            batch_items = missing_items[i : i + chunk_size]
            chunk_texts = [text for _text_hash, text in batch_items]
            try:
                vectors = self.embedding_provider.embed_batch(chunk_texts)
                if len(vectors) != len(batch_items):
                    raise RuntimeError(
                        "embedding provider returned "
                        f"{len(vectors)} vectors for {len(batch_items)} "
                        "chunks during indexing"
                    )
            except Exception as exc:
                raise wrap_embedding_error(
                    exc,
                    provider=self.config.embedding_provider,
                    operation="embed chunk batch for indexing",
                ) from exc
            ledger_entries: list[tuple[str, list[float]]] = []
            batch_completed = 0
            for (text_hash, _text), vector in zip(batch_items, vectors, strict=True):
                ledger_entries.append((text_hash, vector))
                for index in missing_hashes.get(text_hash, []):
                    chunks[index].embedding_vector = vector
                    batch_completed += 1
            self.embedding_ledger.upsert_vectors(
                embedding_profile=embedding_profile,
                record_kind="chunk",
                entries=ledger_entries,
            )
            completed += batch_completed
            if progress_callback is not None:
                progress_callback(completed, total)
        return chunks

    def _apply_edge_embeddings(self, edges: list[EdgeRecord]) -> list[EdgeRecord]:
        """Attach embedding vectors to edge fact text when a provider is available."""
        if not self.config.embed_graph_edges:
            return edges
        if not self.embedding_provider:
            return edges
        texts = [edge.text or "" for edge in edges]
        if not texts:
            return edges
        embedding_profile = self.config.embedding_profile()
        text_hashes = [embedding_text_hash(text) for text in texts]
        cached_vectors = self.embedding_ledger.get_vectors(
            embedding_profile=embedding_profile,
            record_kind="edge",
            text_hashes=text_hashes,
        )
        batch_size = getattr(self.embedding_provider, "_chunk_size", 50)
        missing_hashes: dict[str, list[int]] = {}
        missing_items: list[tuple[str, str]] = []
        edge_inputs = zip(edges, text_hashes, texts, strict=True)
        for index, (edge, text_hash, text) in enumerate(edge_inputs):
            cached_vector = cached_vectors.get(text_hash)
            if cached_vector is not None:
                edge.embedding_vector = list(cached_vector)
                continue
            positions = missing_hashes.setdefault(text_hash, [])
            positions.append(index)
            if len(positions) == 1:
                missing_items.append((text_hash, text))
        for offset in range(0, len(missing_items), batch_size):
            batch_items = missing_items[offset : offset + batch_size]
            text_batch = [text for _text_hash, text in batch_items]
            try:
                vectors = self.embedding_provider.embed_batch(text_batch)
                if len(vectors) != len(batch_items):
                    raise RuntimeError(
                        "embedding provider returned "
                        f"{len(vectors)} vectors for {len(batch_items)} edges during indexing"
                    )
            except Exception as exc:
                raise wrap_embedding_error(
                    exc,
                    provider=self.config.embedding_provider,
                    operation="embed edge batch for indexing",
                ) from exc
            ledger_entries: list[tuple[str, list[float]]] = []
            for (text_hash, _text), vector in zip(batch_items, vectors, strict=True):
                ledger_entries.append((text_hash, vector))
                for index in missing_hashes.get(text_hash, []):
                    edges[index].embedding_vector = vector
            self.embedding_ledger.upsert_vectors(
                embedding_profile=embedding_profile,
                record_kind="edge",
                entries=ledger_entries,
            )
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
        include_text: bool,
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
            include_text=include_text,
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
        span_index: LineByteSpanIndex,
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
                        start_byte=span_index.span_for_lines(part_start, part_end)[0],
                        end_byte=span_index.span_for_lines(part_start, part_end)[1],
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
