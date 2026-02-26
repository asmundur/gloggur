from __future__ import annotations

import errno
import json
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.concurrency import cache_write_lock
from gloggur.indexer.indexer import Indexer
from gloggur.models import IndexMetadata
from gloggur.parsers.registry import ParserRegistry
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def is_process_running(pid: Optional[int]) -> bool:
    """Return True when a PID appears alive in the OS process table.

    `os.kill(pid, 0)` can raise `PermissionError` (`EPERM`) for live processes
    owned by other users; treat that case as alive so callers do not
    misclassify the process as stopped.
    """

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        if exc.errno == errno.EPERM:
            return True
        if exc.errno == errno.ESRCH:
            return False
        return False
    return True


def load_watch_state(path: str) -> Dict[str, object]:
    """Load watcher state JSON from disk."""

    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return {}


@dataclass
class BatchResult:
    """Counters for one processed change batch."""

    changed_files: int = 0
    deleted_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    error_count: int = 0
    indexed_symbols: int = 0
    last_error: Optional[str] = None
    files_considered: int = 0
    failed_reasons: Dict[str, int] = field(default_factory=dict)
    failed_samples: List[str] = field(default_factory=list)

    def record_failed(self, reason: str, sample: str) -> None:
        """Track a failed file outcome with a stable reason and sample."""

        self.error_count += 1
        self.failed_reasons[reason] = self.failed_reasons.get(reason, 0) + 1
        if len(self.failed_samples) < 5:
            self.failed_samples.append(sample)
        self.last_error = sample

    def as_dict(self) -> Dict[str, object]:
        """Convert to a JSON-friendly mapping."""

        return {
            "changed_files": self.changed_files,
            "deleted_files": self.deleted_files,
            "files_considered": self.files_considered,
            "indexed": self.indexed_files,
            "unchanged": self.skipped_files,
            "failed": self.error_count,
            "failed_reasons": dict(self.failed_reasons),
            "failed_samples": list(self.failed_samples),
            "indexed_files": self.indexed_files,
            "skipped_files": self.skipped_files,
            "error_count": self.error_count,
            "indexed_symbols": self.indexed_symbols,
            "last_error": self.last_error,
        }


class WatchService:
    """File watcher that incrementally updates cache and vectors on save."""

    def __init__(
        self,
        config: GloggurConfig,
        embedding_provider: Optional[EmbeddingProvider] = None,
        cache: Optional[CacheManager] = None,
        vector_store: Optional[VectorStore] = None,
        parser_registry: Optional[ParserRegistry] = None,
    ) -> None:
        """Initialize cache, vector store, and indexer dependencies."""

        self.config = config
        self.cache = cache or CacheManager(CacheConfig(config.cache_dir))
        self.vector_store = vector_store or VectorStore(VectorStoreConfig(config.cache_dir))
        self.parser_registry = parser_registry or ParserRegistry()
        self.indexer = Indexer(
            config=config,
            cache=self.cache,
            parser_registry=self.parser_registry,
            embedding_provider=embedding_provider,
            vector_store=self.vector_store,
        )
        self._supported_extensions = set(config.supported_extensions)
        self._stop_event = Event()
        self._total_files_considered = 0
        self._total_indexed_files = 0
        self._total_indexed_symbols = 0
        self._total_skipped_files = 0
        self._total_errors = 0
        self._total_failed_reasons: Dict[str, int] = {}
        self._total_failed_samples: List[str] = []

    def run_forever(self, path: str) -> Dict[str, object]:
        """Watch filesystem changes and process until stopped."""

        watch_root = os.path.abspath(path)
        watch_target = watch_root if os.path.isdir(watch_root) else os.path.dirname(watch_root)
        watch_file = watch_root if os.path.isfile(watch_root) else None

        self._write_state(
            running=True,
            watch_path=watch_root,
            started_at=utc_now_iso(),
            last_heartbeat=utc_now_iso(),
            last_batch={},
            status="running",
        )

        old_sigterm = signal.signal(signal.SIGTERM, self._handle_stop_signal)
        old_sigint = signal.signal(signal.SIGINT, self._handle_stop_signal)
        try:
            for changes in self._watch_changes(watch_target):
                if self._stop_event.is_set():
                    break
                batch = self.process_batch(changes, watch_root=watch_root, watch_file=watch_file)
                self._total_files_considered += batch.files_considered
                self._total_indexed_files += batch.indexed_files
                self._total_indexed_symbols += batch.indexed_symbols
                self._total_skipped_files += batch.skipped_files
                self._total_errors += batch.error_count
                for reason, count in batch.failed_reasons.items():
                    self._total_failed_reasons[reason] = (
                        self._total_failed_reasons.get(reason, 0) + count
                    )
                for sample in batch.failed_samples:
                    if len(self._total_failed_samples) >= 5:
                        break
                    self._total_failed_samples.append(sample)
                self._write_state(
                    running=True,
                    watch_path=watch_root,
                    last_heartbeat=utc_now_iso(),
                    last_batch=batch.as_dict(),
                    files_considered=self._total_files_considered,
                    indexed=self._total_indexed_files,
                    unchanged=self._total_skipped_files,
                    failed=self._total_errors,
                    failed_reasons=dict(self._total_failed_reasons),
                    failed_samples=list(self._total_failed_samples),
                    indexed_files=self._total_indexed_files,
                    indexed_symbols=self._total_indexed_symbols,
                    skipped_files=self._total_skipped_files,
                    error_count=self._total_errors,
                    last_error=batch.last_error,
                    status="running_with_errors" if self._total_errors else "running",
                )
        finally:
            signal.signal(signal.SIGTERM, old_sigterm)
            signal.signal(signal.SIGINT, old_sigint)
            self._write_state(
                running=False,
                watch_path=watch_root,
                stopped_at=utc_now_iso(),
                last_heartbeat=utc_now_iso(),
                files_considered=self._total_files_considered,
                indexed=self._total_indexed_files,
                unchanged=self._total_skipped_files,
                failed=self._total_errors,
                failed_reasons=dict(self._total_failed_reasons),
                failed_samples=list(self._total_failed_samples),
                indexed_files=self._total_indexed_files,
                indexed_symbols=self._total_indexed_symbols,
                skipped_files=self._total_skipped_files,
                error_count=self._total_errors,
                status="stopped",
            )

        return {
            "watch_path": watch_root,
            "files_considered": self._total_files_considered,
            "indexed": self._total_indexed_files,
            "unchanged": self._total_skipped_files,
            "failed": self._total_errors,
            "failed_reasons": dict(self._total_failed_reasons),
            "failed_samples": list(self._total_failed_samples),
            "indexed_files": self._total_indexed_files,
            "indexed_symbols": self._total_indexed_symbols,
            "skipped_files": self._total_skipped_files,
            "error_count": self._total_errors,
        }

    def process_batch(
        self,
        changes: Iterable[Tuple[object, str]],
        watch_root: str,
        watch_file: Optional[str] = None,
    ) -> BatchResult:
        """Process one batch of file-system events."""

        operations: Dict[str, str] = {}
        for change, raw_path in changes:
            path = os.path.abspath(raw_path)
            if not self._in_scope(path, watch_root=watch_root, watch_file=watch_file):
                continue
            if self._is_excluded(path):
                continue
            is_deleted = self._is_deleted_change(change)
            if is_deleted:
                operations[path] = "deleted"
                continue
            if self._is_supported_file(path):
                if operations.get(path) != "deleted":
                    operations[path] = "changed"

        result = BatchResult(files_considered=len(operations))
        metadata_invalidated = False

        def _invalidate_metadata() -> None:
            nonlocal metadata_invalidated
            if metadata_invalidated:
                return
            self.cache.delete_index_metadata()
            metadata_invalidated = True

        if operations:
            with cache_write_lock(self.config.cache_dir):
                for path, operation in operations.items():
                    if operation == "deleted":
                        self._process_deleted(
                            path,
                            result,
                            invalidate_metadata=_invalidate_metadata,
                        )
                    else:
                        self._process_changed(
                            path,
                            result,
                            invalidate_metadata=_invalidate_metadata,
                        )

                if metadata_invalidated:
                    self.vector_store.save()
                    metadata = IndexMetadata(
                        version=self.config.index_version,
                        total_symbols=len(self.cache.list_symbols()),
                        indexed_files=self.cache.count_files(),
                    )
                    self.cache.set_index_metadata(metadata)
                    self.cache.set_index_profile(self.config.embedding_profile())
        return result

    def _watch_changes(self, watch_target: str):
        """Yield raw change batches from the watch backend."""

        try:
            from watchfiles import watch
        except ImportError as exc:
            raise RuntimeError(
                "watchfiles is required for watch mode. Install with `pip install watchfiles`."
            ) from exc

        yield from watch(
            watch_target,
            debounce=max(0, int(self.config.watch_debounce_ms)),
            stop_event=self._stop_event,
            yield_on_timeout=False,
        )

    def _process_deleted(
        self,
        path: str,
        result: BatchResult,
        *,
        invalidate_metadata: Callable[[], None],
    ) -> None:
        """Delete stale symbols/vectors for a removed file."""

        existing = self.cache.get_file_metadata(path)
        if not existing:
            result.skipped_files += 1
            return
        try:
            invalidate_metadata()
            if existing.symbols:
                self.vector_store.remove_ids(existing.symbols)
            self.cache.delete_symbols_for_file(path)
            self.cache.delete_file_metadata(path)
        except Exception as exc:
            result.record_failed("delete_error", f"{path}: {type(exc).__name__}: {exc}")
            return
        result.deleted_files += 1
        result.indexed_files += 1

    def _process_changed(
        self,
        path: str,
        result: BatchResult,
        *,
        invalidate_metadata: Callable[[], None],
    ) -> None:
        """Index a changed file and classify the terminal outcome."""

        outcome = self.indexer.index_file_with_outcome(path)
        if outcome.status == "unchanged":
            result.skipped_files += 1
            return
        if outcome.status == "failed":
            invalidate_metadata()
            reason = outcome.reason or "indexing_error"
            detail = outcome.detail or "indexing failed"
            result.record_failed(reason, f"{path}: {detail}")
            return

        invalidate_metadata()
        result.changed_files += 1
        result.indexed_files += 1
        result.indexed_symbols += outcome.symbols_indexed

    def _in_scope(self, path: str, watch_root: str, watch_file: Optional[str]) -> bool:
        """Return True if a changed path belongs to the active watch scope."""

        if watch_file:
            return path == watch_file
        try:
            return os.path.commonpath([watch_root, path]) == watch_root
        except ValueError:
            return False

    def _is_supported_file(self, path: str) -> bool:
        """Return True for configured source file extensions."""

        return any(path.endswith(ext) for ext in self._supported_extensions)

    def _is_excluded(self, path: str) -> bool:
        """Return True if a path includes an excluded directory segment."""

        segments = set(os.path.normpath(path).split(os.sep))
        return any(excluded in segments for excluded in self.config.excluded_dirs)

    @staticmethod
    def _is_deleted_change(change: object) -> bool:
        """Return True if change value represents file deletion."""

        name = getattr(change, "name", "")
        if str(name).lower() == "deleted":
            return True
        try:
            return int(change) == 3
        except (TypeError, ValueError):
            return False

    def _handle_stop_signal(self, signum: int, _frame: object) -> None:
        """Signal handler that stops the watch loop."""

        _ = signum
        self._stop_event.set()

    def _write_state(self, **fields: object) -> None:
        """Merge and persist watch state payload."""

        path = self.config.watch_state_file
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = load_watch_state(path)
        payload.update(fields)
        with open(path, "w", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2)
