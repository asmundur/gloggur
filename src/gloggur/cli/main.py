from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
import yaml

from gloggur.audit.docstring_audit import audit_docstrings
from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import (
    EmbeddingProviderError,
    format_embedding_error_message,
    wrap_embedding_error,
)
from gloggur.embeddings.factory import create_embedding_provider
from gloggur.indexer.cache import CacheConfig, CacheManager, CacheRecoveryError
from gloggur.indexer.concurrency import cache_write_lock
from gloggur.io_failures import StorageIOError, format_io_error_message, wrap_io_error
from gloggur.indexer.indexer import Indexer
from gloggur.models import AuditFileMetadata, IndexMetadata
from gloggur.parsers.registry import ParserRegistry
from gloggur.search.hybrid_search import HybridSearch
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.watch.service import WatchService, is_process_running, load_watch_state, utc_now_iso


@click.group()
def cli() -> None:
    """Gloggur CLI for indexing, search, and docstring inspection."""


def _emit(payload: Dict[str, object], as_json: bool) -> None:
    """Print payload as JSON or raw text."""
    if as_json:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(payload)


def _resolve_as_json(kwargs: dict[str, object]) -> bool:
    """Resolve --json flag for command wrappers."""
    if "as_json" in kwargs:
        return bool(kwargs["as_json"])
    context = click.get_current_context(silent=True)
    if context is None:
        return False
    return bool(context.params.get("as_json"))


def _with_io_failure_handling(
    callback: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap CLI callbacks with structured I/O error output and non-zero exit."""

    @wraps(callback)
    def _wrapped(*args: object, **kwargs: object) -> Any:
        as_json = _resolve_as_json(kwargs)
        try:
            return callback(*args, **kwargs)
        except StorageIOError as exc:
            click.echo(format_io_error_message(exc), err=True)
            if as_json:
                _emit(exc.to_payload(), as_json=True)
            raise click.exceptions.Exit(1) from exc
        except EmbeddingProviderError as exc:
            click.echo(format_embedding_error_message(exc), err=True)
            if as_json:
                _emit(exc.to_payload(), as_json=True)
            raise click.exceptions.Exit(1) from exc

    return _wrapped


def _load_config(config_path: Optional[str]) -> GloggurConfig:
    """Load configuration from file/env."""
    load_path = _normalize_config_path(config_path)
    error_path = "<auto-discovery>"
    if load_path:
        error_path = load_path
    else:
        for candidate in (".gloggur.yaml", ".gloggur.yml", ".gloggur.json"):
            if os.path.exists(candidate):
                error_path = os.path.abspath(candidate)
                break
    try:
        return GloggurConfig.load(path=load_path)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read gloggur config",
            path=error_path,
        ) from exc
    except (json.JSONDecodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise StorageIOError(
            category="unknown_io_error",
            operation="read gloggur config",
            path=error_path,
            probable_cause=(
                "The gloggur config file is malformed or uses an unsupported "
                "top-level structure."
            ),
            remediation=[
                f"Fix config syntax and top-level mapping structure in {error_path}.",
                "Or pass --config <path> to a valid .gloggur.yaml/.gloggur.json file.",
            ],
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


def _normalize_config_path(config_path: Optional[str]) -> Optional[str]:
    """Return an absolute config path when provided."""
    if not config_path:
        return None
    return os.path.abspath(os.path.expanduser(config_path))


def _resolve_relative_to(base_dir: str, value: str) -> str:
    """Resolve value relative to base_dir when it is not absolute."""
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(base_dir, expanded))


def _normalize_watch_paths(config: GloggurConfig, config_path: Optional[str]) -> GloggurConfig:
    """Resolve watch path fields relative to config file directory."""
    if not config_path:
        return config
    config_dir = os.path.dirname(config_path)
    config.watch_path = _resolve_relative_to(config_dir, config.watch_path)
    config.watch_state_file = _resolve_relative_to(config_dir, config.watch_state_file)
    config.watch_pid_file = _resolve_relative_to(config_dir, config.watch_pid_file)
    config.watch_log_file = _resolve_relative_to(config_dir, config.watch_log_file)
    return config


def _hash_content(source: str) -> str:
    """Hash content to detect changes."""
    return hashlib.sha256(source.encode("utf8")).hexdigest()


def _profile_reindex_reason(
    metadata_present: bool,
    cached_profile: Optional[str],
    expected_profile: str,
) -> Optional[str]:
    """Return a reason why cached index data should be rebuilt."""
    if cached_profile is None:
        if metadata_present:
            return "cached embedding profile is unknown"
        return None
    if cached_profile != expected_profile:
        return (
            "embedding profile changed "
            f"(cached={cached_profile}, current={expected_profile})"
        )
    return None


def _metadata_reindex_reason(metadata_present: bool) -> Optional[str]:
    """Return reason when index metadata is missing/incomplete."""
    if metadata_present:
        return None
    return "index metadata missing (index build in progress, interrupted, or never completed)"


def _resolve_config_file_path(config_path: Optional[str]) -> str:
    """Resolve config file path for watch init updates."""
    if config_path:
        return config_path
    for candidate in (".gloggur.yaml", ".gloggur.yml", ".gloggur.json"):
        if os.path.exists(candidate):
            return candidate
    return ".gloggur.yaml"


def _read_config_payload(path: str) -> Dict[str, object]:
    """Load config file payload (yaml/json), returning empty dict if missing."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf8") as handle:
            if path.endswith(".json"):
                payload = json.load(handle)
            else:
                payload = yaml.safe_load(handle) or {}
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise wrap_io_error(
            exc,
            operation="read watch config payload",
            path=path,
        ) from exc
    if isinstance(payload, dict):
        return payload
    raise wrap_io_error(
        ValueError("watch config payload must be a mapping"),
        operation="read watch config payload",
        path=path,
    )


def _write_config_payload(path: str, payload: Dict[str, object]) -> None:
    """Persist config payload using yaml/json by file extension."""
    directory = os.path.dirname(path)
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf8") as handle:
            if path.endswith(".json"):
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                return
            yaml.safe_dump(payload, handle, sort_keys=False)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise wrap_io_error(
            exc,
            operation="write watch config payload",
            path=path,
        ) from exc


def _read_pid_file(path: str) -> Optional[int]:
    """Read PID from pid file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf8") as handle:
            value = handle.read().strip()
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read watch pid file",
            path=path,
        ) from exc
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise wrap_io_error(
            exc,
            operation="read watch pid file",
            path=path,
        ) from exc


def _write_pid_file(path: str, pid: int) -> None:
    """Write PID to file."""
    directory = os.path.dirname(path)
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf8") as handle:
            handle.write(f"{pid}\n")
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="write watch pid file",
            path=path,
        ) from exc


def _remove_file(path: str) -> None:
    """Best-effort file removal helper."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="delete watch runtime file",
            path=path,
        ) from exc


def _write_watch_state(path: str, updates: Dict[str, object]) -> None:
    """Merge watcher state updates and persist JSON."""
    directory = os.path.dirname(path)
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = load_watch_state(path)
        payload.update(updates)
        with open(path, "w", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        raise wrap_io_error(
            exc,
            operation="write watch state file",
            path=path,
        ) from exc


def _terminate_watch_process(process: object) -> None:
    """Best-effort daemon cleanup for partially initialized watch starts."""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return
    poll = getattr(process, "poll", None)
    if callable(poll):
        try:
            if poll() is not None:
                return
        except Exception:
            pass
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        wait(timeout=0.2)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return
    poll = getattr(process, "poll", None)
    if callable(poll):
        try:
            if poll() is not None:
                return
        except Exception:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _read_watch_state_for_status(path: str) -> Dict[str, object]:
    """Read watch status state file with deterministic failure semantics."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise wrap_io_error(
            exc,
            operation="read watch state file",
            path=path,
        ) from exc
    if isinstance(payload, dict):
        return payload
    raise wrap_io_error(
        ValueError("watch state payload must be a mapping"),
        operation="read watch state file",
        path=path,
    )


def _create_runtime(
    config_path: Optional[str],
    embedding_provider: Optional[str] = None,
    rebuild_on_profile_change: bool = False,
    write_locked: bool = False,
) -> tuple[GloggurConfig, CacheManager, VectorStore]:
    """Create config/cache/vector runtime and apply profile rebuild logic."""
    resolved_config_path = _normalize_config_path(config_path)
    config = _load_config(resolved_config_path)
    # Apply CLI provider override in-memory to avoid a second config-file read.
    if embedding_provider:
        config.embedding_provider = embedding_provider
    config = _normalize_watch_paths(config, resolved_config_path)
    expected_profile = config.embedding_profile()
    cache = _create_cache_manager(config.cache_dir)
    vector_store = VectorStore(VectorStoreConfig(config.cache_dir))
    if cache.last_reset_reason:
        vector_store.clear()
    metadata_present = cache.get_index_metadata() is not None
    cached_profile = cache.get_index_profile()
    reindex_reason = _profile_reindex_reason(metadata_present, cached_profile, expected_profile)
    if reindex_reason and rebuild_on_profile_change:
        click.echo(
            "Embedding settings changed; rebuilding cache at "
            f"{config.cache_dir} ({reindex_reason}).",
            err=True,
        )
        if write_locked:
            cache.clear()
            vector_store.clear()
        else:
            with cache_write_lock(config.cache_dir):
                cache.clear()
                vector_store.clear()
    return config, cache, vector_store


def _create_cache_manager(cache_dir: str) -> CacheManager:
    """Create cache manager with user-facing error mapping for unrecoverable corruption."""
    try:
        return CacheManager(CacheConfig(cache_dir))
    except CacheRecoveryError as exc:
        raise StorageIOError(
            category="unknown_io_error",
            operation="recover corrupted cache database",
            path=os.path.join(cache_dir, "index.db"),
            probable_cause=(
                "Automatic cache corruption recovery failed due to filesystem constraints "
                "or artifact access issues."
            ),
            remediation=[
                "Fix permissions for the cache path and remove corrupted cache artifacts manually.",
                "Retry the command after cleanup, or run `gloggur clear-cache --json`.",
            ],
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


def _is_transient_status_race_error(error: StorageIOError) -> bool:
    """Return True when status hit a transient table-missing race during concurrent recovery."""
    detail = error.detail.lower()
    if error.operation == "execute cache database transaction" and (
        "no such table" in detail
        or "database schema has changed" in detail
    ):
        return True
    if error.operation == "configure cache database pragmas":
        return True
    return False


def _remap_status_recovery_error(error: StorageIOError) -> StorageIOError:
    """Remap transient status races to a stable recovery operation contract."""
    return StorageIOError(
        category=error.category,
        operation="recover corrupted cache database",
        path=error.path,
        probable_cause=error.probable_cause,
        remediation=list(error.remediation),
        detail=error.detail,
    )


def _build_status_payload(config: GloggurConfig, cache: CacheManager) -> Dict[str, object]:
    """Build status payload from cache metadata/profile state."""
    expected_profile = config.embedding_profile()
    metadata = cache.get_index_metadata()
    schema_version = cache.get_schema_version()
    cached_profile = cache.get_index_profile()
    metadata_reason = _metadata_reindex_reason(metadata is not None)
    profile_reason = _profile_reindex_reason(
        metadata_present=metadata is not None,
        cached_profile=cached_profile,
        expected_profile=expected_profile,
    )
    reindex_reason = metadata_reason or profile_reason
    if cache.last_reset_reason:
        reset_label = "cache schema rebuilt"
        if "cache corruption detected" in cache.last_reset_reason:
            reset_label = "cache corruption recovered"
        reindex_reason = (
            f"{reset_label} "
            f"({cache.last_reset_reason})"
        )
    return {
        "cache_dir": config.cache_dir,
        "metadata": metadata.model_dump(mode="json") if metadata else None,
        "schema_version": schema_version,
        "expected_index_profile": expected_profile,
        "cached_index_profile": cached_profile,
        "needs_reindex": metadata is None or reindex_reason is not None,
        "reindex_reason": reindex_reason,
        "total_symbols": len(cache.list_symbols()),
    }


def _create_status_payload(config: GloggurConfig) -> Dict[str, object]:
    """Create cache manager and build status payload."""
    cache = _create_cache_manager(config.cache_dir)
    return _build_status_payload(config, cache)


def _create_embedding_provider_for_command(
    config: GloggurConfig,
    *,
    require_provider: bool = False,
) -> Optional[EmbeddingProvider]:
    """Create embedding provider with deterministic error mapping."""
    if not config.embedding_provider:
        if require_provider:
            raise wrap_embedding_error(
                ValueError("embedding provider is not configured"),
                provider="unknown",
                operation="initialize embedding provider",
            )
        return None
    try:
        return create_embedding_provider(config)
    except Exception as exc:
        raise wrap_embedding_error(
            exc,
            provider=config.embedding_provider,
            operation="initialize embedding provider",
        ) from exc


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--embedding-provider", type=str, default=None)
@click.option(
    "--allow-partial",
    is_flag=True,
    default=False,
    help="Exit zero even when some files fail to index.",
)
@_with_io_failure_handling
def index(
    path: str,
    config_path: Optional[str],
    as_json: bool,
    embedding_provider: Optional[str],
    allow_partial: bool,
) -> None:
    """Load config/runtime, index path, and emit summary counts."""
    resolved_config_path = _normalize_config_path(config_path)
    lock_config = _load_config(resolved_config_path)
    with cache_write_lock(lock_config.cache_dir):
        config, cache, vector_store = _create_runtime(
            config_path=resolved_config_path,
            embedding_provider=embedding_provider,
            rebuild_on_profile_change=True,
            write_locked=True,
        )
        click.echo("Indexing...", err=True)
        embedding = _create_embedding_provider_for_command(
            config,
            require_provider=True,
        )
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=embedding,
            vector_store=vector_store,
        )
        if os.path.isdir(path):
            result = indexer.index_repository(path)
            payload = result.as_payload()
            _emit(payload, as_json)
            if result.failed > 0 and not allow_partial:
                raise click.exceptions.Exit(1)
            return

        files_considered = 1
        if any(path.endswith(ext) for ext in config.supported_extensions):
            segments = set(os.path.normpath(os.path.abspath(path)).split(os.sep))
            if any(excluded in segments for excluded in config.excluded_dirs):
                files_considered = 0
        else:
            files_considered = 0
        if files_considered:
            cache.delete_index_metadata()
        outcome = indexer.index_file_with_outcome(path) if files_considered else None
        if vector_store and outcome and outcome.status == "indexed":
            vector_store.save()
        if outcome and outcome.status != "failed":
            metadata = IndexMetadata(
                version=config.index_version,
                total_symbols=len(cache.list_symbols()),
                indexed_files=cache.count_files(),
            )
            cache.set_index_metadata(metadata)
            cache.set_index_profile(config.embedding_profile())
        failed_reasons: Dict[str, int] = {}
        failed_samples: List[str] = []
        indexed = 0
        unchanged = 0
        failed = 0
        indexed_symbols = 0
        if outcome:
            if outcome.status == "indexed":
                indexed = 1
                indexed_symbols = outcome.symbols_indexed
            elif outcome.status == "unchanged":
                unchanged = 1
            else:
                failed = 1
                reason = outcome.reason or "indexing_error"
                failed_reasons[reason] = 1
                detail = outcome.detail or "indexing failed"
                failed_samples.append(f"{path}: {detail}")
        result = {
            "files_considered": files_considered,
            "indexed": indexed,
            "unchanged": unchanged,
            "failed": failed,
            "failed_reasons": failed_reasons,
            "failed_samples": failed_samples,
            "indexed_files": indexed,
            "skipped_files": unchanged,
            "indexed_symbols": indexed_symbols,
            "duration_ms": 0,
        }
        _emit(result, as_json)
        if failed > 0 and not allow_partial:
            raise click.exceptions.Exit(1)


@cli.command()
@click.argument("query", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--kind", type=str, default=None)
@click.option("--file", "file_path", type=str, default=None)
@click.option("--top-k", type=int, default=10)
@click.option("--stream", is_flag=True, default=False)
@_with_io_failure_handling
def search(
    query: str,
    config_path: Optional[str],
    as_json: bool,
    kind: Optional[str],
    file_path: Optional[str],
    top_k: int,
    stream: bool,
) -> None:
    """Search indexed symbols with optional filters."""
    config, cache, vector_store = _create_runtime(config_path=config_path)
    expected_profile = config.embedding_profile()
    metadata_present = cache.get_index_metadata() is not None
    cached_profile = cache.get_index_profile()
    metadata_reason = _metadata_reindex_reason(metadata_present)
    profile_reason = _profile_reindex_reason(metadata_present, cached_profile, expected_profile)
    reindex_reason = metadata_reason or profile_reason
    if reindex_reason is not None:
        payload = {
            "query": query,
            "results": [],
            "metadata": {
                "total_results": 0,
                "search_time_ms": 0,
                "needs_reindex": True,
                "reindex_reason": reindex_reason,
                "expected_index_profile": expected_profile,
                "cached_index_profile": cached_profile,
            },
        }
        _emit(payload, as_json)
        return
    embedding = _create_embedding_provider_for_command(
        config,
        require_provider=True,
    )
    metadata_store = MetadataStore(MetadataStoreConfig(config.cache_dir))
    searcher = HybridSearch(embedding, vector_store, metadata_store)
    filters = {}
    if kind:
        filters["kind"] = kind
    if file_path:
        filters["file"] = file_path
    result = searcher.search(query, filters=filters, top_k=top_k)
    if stream and as_json:
        for item in result["results"]:
            click.echo(json.dumps(item))
        return
    _emit(result, as_json)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Reinspect even if unchanged since last run.",
)
@click.option(
    "--symbol-id",
    "symbol_ids",
    multiple=True,
    help="Inspect only the specified symbol id(s). Can be repeated.",
)
@click.option(
    "--allow-partial",
    is_flag=True,
    default=False,
    help="Exit zero even when some files fail inspection.",
)
@_with_io_failure_handling
def inspect(
    path: str,
    config_path: Optional[str],
    as_json: bool,
    force: bool,
    symbol_ids: tuple[str, ...],
    allow_partial: bool,
) -> None:
    """Run docstring inspection and emit warnings/reports."""
    config = _load_config(config_path)
    cache = _create_cache_manager(config.cache_dir)
    parser_registry = ParserRegistry()
    embedding = _create_embedding_provider_for_command(config)
    symbols = []
    code_texts: Dict[str, str] = {}
    processed_files: List[Tuple[str, str]] = []
    files_considered = 0
    inspected_files = 0
    failed_files = 0
    failed_reasons: Dict[str, int] = {}
    failed_samples: List[str] = []
    skipped_files = 0
    paths = [path]
    if os.path.isdir(path):
        paths = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in config.excluded_dirs]
            for filename in files:
                full_path = os.path.join(root, filename)
                if any(full_path.endswith(ext) for ext in config.supported_extensions):
                    paths.append(full_path)
    elif not any(path.endswith(ext) for ext in config.supported_extensions):
        paths = []
    else:
        segments = set(os.path.normpath(os.path.abspath(path)).split(os.sep))
        if any(excluded in segments for excluded in config.excluded_dirs):
            paths = []
    for file_path in paths:
        files_considered += 1
        try:
            with open(file_path, "r", encoding="utf8") as handle:
                source = handle.read()
        except UnicodeDecodeError as exc:
            failed_files += 1
            failed_reasons["decode_error"] = failed_reasons.get("decode_error", 0) + 1
            if len(failed_samples) < 5:
                failed_samples.append(f"{file_path}: {type(exc).__name__}: {exc}")
            continue
        except OSError as exc:
            failed_files += 1
            failed_reasons["read_error"] = failed_reasons.get("read_error", 0) + 1
            if len(failed_samples) < 5:
                failed_samples.append(f"{file_path}: {type(exc).__name__}: {exc}")
            continue
        content_hash = _hash_content(source)
        if not force:
            existing = cache.get_audit_file_metadata(file_path)
            if existing and existing.content_hash == content_hash:
                skipped_files += 1
                continue
        parser_entry = parser_registry.get_parser_for_path(file_path)
        if not parser_entry:
            failed_files += 1
            failed_reasons["parser_unavailable"] = failed_reasons.get("parser_unavailable", 0) + 1
            if len(failed_samples) < 5:
                failed_samples.append(f"{file_path}: No parser registered for file extension.")
            continue
        try:
            file_symbols = parser_entry.parser.extract_symbols(file_path, source)
        except Exception as exc:
            failed_files += 1
            failed_reasons["parse_error"] = failed_reasons.get("parse_error", 0) + 1
            if len(failed_samples) < 5:
                failed_samples.append(f"{file_path}: {type(exc).__name__}: {exc}")
            continue
        if symbol_ids:
            file_symbols = [symbol for symbol in file_symbols if symbol.id in symbol_ids]
            if not file_symbols:
                continue
        lines = source.splitlines()
        for symbol in file_symbols:
            snippet_start = max(0, symbol.start_line - 1)
            snippet_end = min(len(lines), symbol.end_line)
            code_texts[symbol.id] = "\n".join(lines[snippet_start:snippet_end])
        symbols.extend(file_symbols)
        processed_files.append((file_path, content_hash))
        inspected_files += 1
    reports = audit_docstrings(
        symbols,
        code_texts=code_texts,
        embedding_provider=embedding,
        semantic_threshold=config.docstring_semantic_threshold,
        semantic_min_chars=config.docstring_semantic_min_chars,
        semantic_max_chars=config.docstring_semantic_max_chars,
    )
    for report in reports:
        cache.set_audit_warnings(report.symbol_id, report.warnings)
    for file_path, content_hash in processed_files:
        cache.upsert_audit_file_metadata(
            AuditFileMetadata(path=file_path, content_hash=content_hash)
        )
    warning_reports = [report for report in reports if report.warnings]
    payload = {
        "path": path,
        "symbol_ids": list(symbol_ids) if symbol_ids else None,
        "warnings": [report.__dict__ for report in warning_reports],
        "total": len(warning_reports),
        "reports": [report.__dict__ for report in reports],
        "reports_total": len(reports),
        "files_considered": files_considered,
        "indexed": inspected_files,
        "unchanged": skipped_files,
        "failed": failed_files,
        "failed_reasons": failed_reasons,
        "failed_samples": failed_samples,
        "inspected_files": inspected_files,
        "skipped_files": skipped_files,
    }
    _emit(payload, as_json)
    if failed_files > 0 and not allow_partial:
        raise click.exceptions.Exit(1)


@cli.group()
def watch() -> None:
    """Manage save-triggered incremental indexing."""


@watch.command("init")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@_with_io_failure_handling
def watch_init(path: str, config_path: Optional[str], as_json: bool) -> None:
    """Enable watch mode defaults in config for a repository path."""

    config_file = _normalize_config_path(_resolve_config_file_path(config_path))
    assert config_file is not None
    payload = _read_config_payload(config_file)
    payload["watch_enabled"] = True
    payload["watch_path"] = os.path.abspath(path)
    payload.setdefault("watch_debounce_ms", 300)
    payload.setdefault("watch_mode", "daemon")
    payload.setdefault("watch_state_file", ".gloggur-cache/watch_state.json")
    payload.setdefault("watch_pid_file", ".gloggur-cache/watch.pid")
    payload.setdefault("watch_log_file", ".gloggur-cache/watch.log")
    _write_config_payload(config_file, payload)
    response = {
        "initialized": True,
        "config_file": config_file,
        "watch_path": payload["watch_path"],
        "watch_mode": payload["watch_mode"],
        "next_steps": ["gloggur watch start", "gloggur watch status"],
    }
    _emit(response, as_json)


@watch.command("start")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--foreground", "force_foreground", is_flag=True, default=False)
@click.option("--daemon", "force_daemon", is_flag=True, default=False)
@click.option(
    "--allow-partial",
    is_flag=True,
    default=False,
    help="Foreground mode: exit zero even when some files fail.",
)
@_with_io_failure_handling
def watch_start(
    config_path: Optional[str],
    as_json: bool,
    force_foreground: bool,
    force_daemon: bool,
    allow_partial: bool,
) -> None:
    """Start watcher in foreground or daemon mode."""

    if force_foreground and force_daemon:
        raise click.ClickException("Use only one of --foreground or --daemon.")

    resolved_config_path = _normalize_config_path(config_path)
    config, cache, vector_store = _create_runtime(
        config_path=resolved_config_path,
        rebuild_on_profile_change=True,
    )
    watch_path = os.path.abspath(config.watch_path)
    if not os.path.exists(watch_path):
        raise click.ClickException(f"Watch path does not exist: {watch_path}")

    mode = config.watch_mode
    if force_foreground:
        mode = "foreground"
    if force_daemon:
        mode = "daemon"
    if mode not in {"foreground", "daemon"}:
        raise click.ClickException(f"Unsupported watch mode: {mode}")

    pid_path = config.watch_pid_file
    pid = _read_pid_file(pid_path)
    daemon_child = os.getenv("GLOGGUR_WATCH_DAEMON_CHILD") == "1"
    if daemon_child and pid == os.getpid():
        pid = None
    if is_process_running(pid):
        _emit({"started": False, "reason": "already_running", "pid": pid}, as_json)
        return

    embedding = _create_embedding_provider_for_command(config)
    service = WatchService(
        config=config,
        embedding_provider=embedding,
        cache=cache,
        vector_store=vector_store,
    )

    if mode == "daemon":
        log_file = config.watch_log_file
        log_dir = os.path.dirname(log_file)
        try:
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="prepare watch log directory",
                path=log_dir or log_file,
            ) from exc
        cmd = [sys.executable, "-m", "gloggur.cli.main", "watch", "start", "--foreground"]
        if resolved_config_path:
            cmd.extend(["--config", resolved_config_path])
        if allow_partial:
            cmd.append("--allow-partial")
        child_env = os.environ.copy()
        child_env["GLOGGUR_WATCH_DAEMON_CHILD"] = "1"
        try:
            log_handle = open(log_file, "a", encoding="utf8")
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="open watch log file",
                path=log_file,
            ) from exc
        try:
            with log_handle:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=log_handle,
                    start_new_session=True,
                    env=child_env,
                )
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="spawn watch daemon process",
                path=sys.executable,
            ) from exc
        time.sleep(0.05)
        daemon_exit_code = process.poll()
        if daemon_exit_code is not None:
            try:
                _remove_file(pid_path)
            except StorageIOError:
                pass
            try:
                _write_watch_state(
                    config.watch_state_file,
                    {
                        "running": False,
                        "status": "failed_startup",
                        "pid": process.pid,
                        "watch_path": watch_path,
                        "stopped_at": utc_now_iso(),
                        "last_error": (
                            f"watch daemon exited early with code {daemon_exit_code}"
                        ),
                    },
                )
            except StorageIOError:
                pass
            raise StorageIOError(
                category="unknown_io_error",
                operation="verify watch daemon startup",
                path=log_file,
                probable_cause="Watch daemon exited before reporting a running state.",
                remediation=[
                    "Inspect the watch log file for startup errors and traceback details.",
                    "Fix configuration/dependency issues and rerun `gloggur watch start --daemon --json`.",
                ],
                detail=f"RuntimeError: watch daemon exited early with code {daemon_exit_code}",
            )
        try:
            _write_pid_file(pid_path, process.pid)
            _write_watch_state(
                config.watch_state_file,
                {
                    "running": True,
                    "status": "starting",
                    "pid": process.pid,
                    "watch_path": watch_path,
                    "last_heartbeat": utc_now_iso(),
                },
            )
        except Exception:
            _terminate_watch_process(process)
            try:
                _remove_file(pid_path)
            except StorageIOError:
                pass
            raise
        daemon_exit_code = process.poll()
        if daemon_exit_code is not None:
            try:
                _remove_file(pid_path)
            except StorageIOError:
                pass
            try:
                _write_watch_state(
                    config.watch_state_file,
                    {
                        "running": False,
                        "status": "failed_startup",
                        "pid": process.pid,
                        "watch_path": watch_path,
                        "stopped_at": utc_now_iso(),
                        "last_error": (
                            f"watch daemon exited early with code {daemon_exit_code}"
                        ),
                    },
                )
            except StorageIOError:
                pass
            raise StorageIOError(
                category="unknown_io_error",
                operation="verify watch daemon startup",
                path=log_file,
                probable_cause="Watch daemon exited before reporting a stable running state.",
                remediation=[
                    "Inspect the watch log file for startup errors and traceback details.",
                    "Fix configuration/dependency issues and rerun `gloggur watch start --daemon --json`.",
                ],
                detail=f"RuntimeError: watch daemon exited early with code {daemon_exit_code}",
            )
        _emit(
            {
                "started": True,
                "mode": "daemon",
                "pid": process.pid,
                "watch_path": watch_path,
                "log_file": log_file,
            },
            as_json,
        )
        return

    _write_pid_file(pid_path, os.getpid())
    try:
        result = service.run_forever(watch_path)
    finally:
        _remove_file(pid_path)
    _emit(
        {
            "started": True,
            "mode": "foreground",
            "pid": os.getpid(),
            **result,
        },
        as_json,
    )
    if int(result.get("failed", result.get("error_count", 0))) > 0 and not allow_partial:
        raise click.exceptions.Exit(1)


@watch.command("stop")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@_with_io_failure_handling
def watch_stop(config_path: Optional[str], as_json: bool) -> None:
    """Stop watcher process identified by pid file."""

    resolved_config_path = _normalize_config_path(config_path)
    config = _normalize_watch_paths(_load_config(resolved_config_path), resolved_config_path)
    pid = _read_pid_file(config.watch_pid_file)
    if not is_process_running(pid):
        _remove_file(config.watch_pid_file)
        _write_watch_state(
            config.watch_state_file,
            {
                "running": False,
                "status": "stopped",
                "stopped_at": utc_now_iso(),
            },
        )
        _emit({"stopped": False, "running": False, "pid": pid}, as_json)
        return

    assert pid is not None
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="signal watch process",
            path=config.watch_pid_file,
        ) from exc
    for _ in range(30):
        if not is_process_running(pid):
            break
        time.sleep(0.1)

    running = is_process_running(pid)
    if not running:
        _remove_file(config.watch_pid_file)
    _write_watch_state(
        config.watch_state_file,
        {
            "running": running,
            "status": "stopped" if not running else "stopping",
            "stopped_at": utc_now_iso() if not running else None,
            "pid": pid,
        },
    )
    _emit({"stopped": not running, "running": running, "pid": pid}, as_json)


@watch.command("status")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@_with_io_failure_handling
def watch_status(config_path: Optional[str], as_json: bool) -> None:
    """Show watcher process and heartbeat status."""

    resolved_config_path = _normalize_config_path(config_path)
    config = _normalize_watch_paths(_load_config(resolved_config_path), resolved_config_path)
    pid = _read_pid_file(config.watch_pid_file)
    running = is_process_running(pid)
    state = _read_watch_state_for_status(config.watch_state_file)
    payload: Dict[str, object] = {
        "watch_enabled": config.watch_enabled,
        "watch_path": os.path.abspath(config.watch_path),
        "mode": config.watch_mode,
        "pid": pid,
        "running": running,
        "state_file": config.watch_state_file,
        "log_file": config.watch_log_file,
    }
    payload.update(state)
    payload["running"] = running
    _emit(payload, as_json)


@cli.command()
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@_with_io_failure_handling
def status(config_path: Optional[str], as_json: bool) -> None:
    """Show index statistics and metadata."""
    config = _load_config(config_path)
    try:
        payload = _create_status_payload(config)
    except StorageIOError as error:
        if not _is_transient_status_race_error(error):
            raise
        try:
            payload = _create_status_payload(config)
        except StorageIOError as retry_error:
            if _is_transient_status_race_error(retry_error):
                raise _remap_status_recovery_error(retry_error) from retry_error
            raise
    _emit(payload, as_json)


@cli.command("clear-cache")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@_with_io_failure_handling
def clear_cache(config_path: Optional[str], as_json: bool) -> None:
    """Clear the index cache."""
    resolved_config_path = _normalize_config_path(config_path)
    config = _load_config(resolved_config_path)
    with cache_write_lock(config.cache_dir):
        cache = _create_cache_manager(config.cache_dir)
        cache.clear()
        vector_store = VectorStore(
            VectorStoreConfig(config.cache_dir),
            load_existing=False,
        )
        vector_store.clear()
    _emit({"cleared": True, "cache_dir": config.cache_dir}, as_json)


if __name__ == "__main__":
    cli()
