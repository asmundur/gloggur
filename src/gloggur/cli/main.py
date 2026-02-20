from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import click
import yaml

from gloggur.audit.docstring_audit import audit_docstrings
from gloggur.config import GloggurConfig
from gloggur.embeddings.factory import create_embedding_provider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.indexer import Indexer
from gloggur.models import AuditFileMetadata
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


def _load_config(config_path: Optional[str]) -> GloggurConfig:
    """Load configuration from file/env."""
    return GloggurConfig.load(path=config_path)


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
    with open(path, "r", encoding="utf8") as handle:
        if path.endswith(".json"):
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle) or {}
    if isinstance(payload, dict):
        return payload
    return {}


def _write_config_payload(path: str, payload: Dict[str, object]) -> None:
    """Persist config payload using yaml/json by file extension."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        if path.endswith(".json"):
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            return
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_pid_file(path: str) -> Optional[int]:
    """Read PID from pid file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf8") as handle:
            value = handle.read().strip()
        if not value:
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def _write_pid_file(path: str, pid: int) -> None:
    """Write PID to file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        handle.write(f"{pid}\n")


def _remove_file(path: str) -> None:
    """Best-effort file removal helper."""
    if os.path.exists(path):
        os.remove(path)


def _write_watch_state(path: str, updates: Dict[str, object]) -> None:
    """Merge watcher state updates and persist JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = load_watch_state(path)
    payload.update(updates)
    with open(path, "w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2)


def _create_runtime(
    config_path: Optional[str],
    embedding_provider: Optional[str] = None,
    rebuild_on_profile_change: bool = False,
) -> tuple[GloggurConfig, CacheManager, VectorStore]:
    """Create config/cache/vector runtime and apply profile rebuild logic."""
    resolved_config_path = _normalize_config_path(config_path)
    overrides: Dict[str, str] = {}
    if embedding_provider:
        overrides["embedding_provider"] = embedding_provider
    config = _load_config(resolved_config_path)
    if overrides:
        config = GloggurConfig.load(path=resolved_config_path, overrides=overrides)
    config = _normalize_watch_paths(config, resolved_config_path)
    expected_profile = config.embedding_profile()
    cache = CacheManager(CacheConfig(config.cache_dir))
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
        cache.clear()
        vector_store.clear()
    return config, cache, vector_store


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--embedding-provider", type=str, default=None)
def index(
    path: str,
    config_path: Optional[str],
    as_json: bool,
    embedding_provider: Optional[str],
) -> None:
    """Load config/runtime, index path, and emit summary counts."""
    config, cache, vector_store = _create_runtime(
        config_path=config_path,
        embedding_provider=embedding_provider,
        rebuild_on_profile_change=True,
    )
    click.echo("Indexing...", err=True)
    embedding = create_embedding_provider(config) if config.embedding_provider else None
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=embedding,
        vector_store=vector_store,
    )
    if os.path.isdir(path):
        result = indexer.index_repository(path)
    else:
        count = indexer.index_file(path) or 0
        result = {
            "indexed_files": 1 if count else 0,
            "indexed_symbols": count,
            "skipped_files": 0 if count else 1,
            "duration_ms": 0,
        }
        _emit(result, as_json)
        return
    _emit(result.__dict__, as_json)


@cli.command()
@click.argument("query", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--kind", type=str, default=None)
@click.option("--file", "file_path", type=str, default=None)
@click.option("--top-k", type=int, default=10)
@click.option("--stream", is_flag=True, default=False)
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
    reindex_reason = _profile_reindex_reason(metadata_present, cached_profile, expected_profile)
    if reindex_reason:
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
    embedding = create_embedding_provider(config)
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
def inspect(
    path: str,
    config_path: Optional[str],
    as_json: bool,
    force: bool,
    symbol_ids: tuple[str, ...],
) -> None:
    """Run docstring inspection and emit warnings/reports."""
    config = _load_config(config_path)
    cache = CacheManager(CacheConfig(config.cache_dir))
    parser_registry = ParserRegistry()
    embedding = create_embedding_provider(config) if config.embedding_provider else None
    symbols = []
    code_texts: Dict[str, str] = {}
    processed_files: List[Tuple[str, str]] = []
    skipped_files = 0
    inspected_files = 0
    paths = [path]
    if os.path.isdir(path):
        paths = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in config.excluded_dirs]
            for filename in files:
                full_path = os.path.join(root, filename)
                if any(full_path.endswith(ext) for ext in config.supported_extensions):
                    paths.append(full_path)
    for file_path in paths:
        try:
            with open(file_path, "r", encoding="utf8") as handle:
                source = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        content_hash = _hash_content(source)
        if not force:
            existing = cache.get_audit_file_metadata(file_path)
            if existing and existing.content_hash == content_hash:
                skipped_files += 1
                continue
        parser_entry = parser_registry.get_parser_for_path(file_path)
        if not parser_entry:
            continue
        file_symbols = parser_entry.parser.extract_symbols(file_path, source)
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
        "inspected_files": inspected_files,
        "skipped_files": skipped_files,
    }
    _emit(payload, as_json)


@cli.group()
def watch() -> None:
    """Manage save-triggered incremental indexing."""


@watch.command("init")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
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
def watch_start(
    config_path: Optional[str],
    as_json: bool,
    force_foreground: bool,
    force_daemon: bool,
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
    if is_process_running(pid):
        _emit({"started": False, "reason": "already_running", "pid": pid}, as_json)
        return

    embedding = create_embedding_provider(config) if config.embedding_provider else None
    service = WatchService(
        config=config,
        embedding_provider=embedding,
        cache=cache,
        vector_store=vector_store,
    )

    if mode == "daemon":
        log_file = config.watch_log_file
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        cmd = [sys.executable, "-m", "gloggur.cli.main", "watch", "start", "--foreground"]
        if resolved_config_path:
            cmd.extend(["--config", resolved_config_path])
        with open(log_file, "a", encoding="utf8") as log_handle:
            process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
            )
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


@watch.command("stop")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
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
    os.kill(pid, signal.SIGTERM)
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
def watch_status(config_path: Optional[str], as_json: bool) -> None:
    """Show watcher process and heartbeat status."""

    resolved_config_path = _normalize_config_path(config_path)
    config = _normalize_watch_paths(_load_config(resolved_config_path), resolved_config_path)
    pid = _read_pid_file(config.watch_pid_file)
    running = is_process_running(pid)
    state = load_watch_state(config.watch_state_file)
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
def status(config_path: Optional[str], as_json: bool) -> None:
    """Show index statistics and metadata."""
    config = _load_config(config_path)
    expected_profile = config.embedding_profile()
    cache = CacheManager(CacheConfig(config.cache_dir))
    metadata = cache.get_index_metadata()
    schema_version = cache.get_schema_version()
    cached_profile = cache.get_index_profile()
    reindex_reason = _profile_reindex_reason(
        metadata_present=metadata is not None,
        cached_profile=cached_profile,
        expected_profile=expected_profile,
    )
    if cache.last_reset_reason:
        reindex_reason = (
            "cache schema rebuilt "
            f"({cache.last_reset_reason})"
        )
    payload = {
        "cache_dir": config.cache_dir,
        "metadata": metadata.model_dump(mode="json") if metadata else None,
        "schema_version": schema_version,
        "expected_index_profile": expected_profile,
        "cached_index_profile": cached_profile,
        "needs_reindex": metadata is None or reindex_reason is not None,
        "reindex_reason": reindex_reason,
        "total_symbols": len(cache.list_symbols()),
    }
    _emit(payload, as_json)


@cli.command("clear-cache")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
def clear_cache(config_path: Optional[str], as_json: bool) -> None:
    """Clear the index cache."""
    config = _load_config(config_path)
    cache = CacheManager(CacheConfig(config.cache_dir))
    cache.clear()
    vector_store = VectorStore(VectorStoreConfig(config.cache_dir))
    vector_store.clear()
    _emit({"cleared": True, "cache_dir": config.cache_dir}, as_json)


if __name__ == "__main__":
    cli()
