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

from gloggur import __version__ as GLOGGUR_VERSION
from gloggur.audit.docstring_audit import audit_docstrings
from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import (
    EmbeddingProviderError,
    format_embedding_error_message,
    wrap_embedding_error,
)
from gloggur.embeddings.factory import create_embedding_provider
from gloggur.indexer.cache import (
    CACHE_SCHEMA_VERSION,
    CacheConfig,
    CacheManager,
    CacheRecoveryError,
)
from gloggur.indexer.concurrency import cache_write_lock
from gloggur.io_failures import StorageIOError, format_io_error_message, wrap_io_error
from gloggur.indexer.indexer import FAILURE_REMEDIATION, Indexer
from gloggur.models import AuditFileMetadata, IndexMetadata
from gloggur.parsers.registry import ParserRegistry
from gloggur.search.hybrid_search import HybridSearch
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.watch.service import WatchService, is_process_running, load_watch_state, utc_now_iso


@click.group()
def cli() -> None:
    """Gloggur CLI for indexing, search, and docstring inspection."""


INSPECT_PAYLOAD_SCHEMA_VERSION = "1"
DEFAULT_INDEX_FAILURE_REMEDIATION = (
    "Inspect failed_samples and rerun indexing after resolving the underlying error."
)
DEFAULT_WATCH_STATUS_FAILURE_REMEDIATION = (
    "Inspect watch-state counters and rerun `gloggur watch stop --json` then `gloggur watch start --json`."
)
WATCH_STATUS_FAILURE_REMEDIATION: Dict[str, List[str]] = {
    "watch_state_inconsistent": [
        "Watch state reports failures without reason codes; restart watch and verify state-file updates.",
        "If this recurs, run `gloggur index . --json` to re-establish deterministic cache state.",
    ],
    "watch_last_batch_inconsistent": [
        "Watch last_batch reports failures but reason codes are missing; restart watch and verify daemon state writes.",
        "Run `gloggur index . --json` if inconsistent batch-state reporting persists.",
    ],
}
DEFAULT_RESUME_REMEDIATION = (
    "Inspect resume_reason_details and rerun `gloggur index . --json` after resolving the issue."
)
RESUME_REMEDIATION: Dict[str, List[str]] = {
    "missing_index_metadata": [
        "Run `gloggur index . --json` to rebuild missing metadata.",
        "Avoid reusing cache state until the rebuild completes successfully.",
    ],
    "index_interrupted": [
        "A previous index run appears interrupted; rerun `gloggur index . --json` to completion.",
        "If interruption repeats, inspect process termination signals and cache write-lock timing.",
    ],
    "missing_cached_profile": [
        "Rebuild the index so cache metadata records the active embedding profile.",
    ],
    "embedding_profile_changed": [
        "Run `gloggur index . --json` using the current embedding profile to refresh vectors.",
    ],
    "tool_version_changed": [
        "Reindex with the current CLI/tool version before relying on cached retrieval.",
    ],
    "cache_corruption_recovered": [
        "Rebuild the index after corruption recovery to restore full symbol coverage.",
    ],
    "cache_schema_rebuilt": [
        "Run a full `gloggur index . --json` after schema rebuild before search operations.",
    ],
}


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


def _metadata_reindex_signal(
    *,
    metadata_present: bool,
    has_last_success_marker: bool,
) -> Optional[Tuple[str, str]]:
    """Return machine-readable metadata reindex signal with interruption disambiguation."""
    if metadata_present:
        return None
    if has_last_success_marker:
        return (
            "index_interrupted",
            "index metadata missing after a previous successful index (index run interrupted or failed before completion)",
        )
    reason = _metadata_reindex_reason(metadata_present=False)
    return ("missing_index_metadata", reason or "index metadata missing")


def _tool_version_reindex_reason(
    *,
    last_success_tool_version: Optional[str],
    current_tool_version: str,
) -> Optional[str]:
    """Return reason when tool-version drift invalidates previously successful cache state."""
    if last_success_tool_version is None:
        return None
    if last_success_tool_version == current_tool_version:
        return None
    return (
        "tool version changed "
        f"(cached={last_success_tool_version}, current={current_tool_version})"
    )


def _stable_fingerprint(payload: Dict[str, object]) -> str:
    """Return a stable SHA256 fingerprint for JSON-serializable payloads."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _hash_content(serialized)


def _index_metadata_digest(metadata: Optional[IndexMetadata]) -> Optional[str]:
    """Return a deterministic digest of index metadata fields used for resume checks."""
    if metadata is None:
        return None
    payload: Dict[str, object] = {
        "version": metadata.version,
        "last_updated": metadata.last_updated.isoformat(),
        "total_symbols": metadata.total_symbols,
        "indexed_files": metadata.indexed_files,
    }
    return _stable_fingerprint(payload)


def _reset_reindex_signal(reset_reason: Optional[str]) -> Optional[Tuple[str, str]]:
    """Map cache reset reason text to a stable machine-readable code + detail."""
    if not reset_reason:
        return None
    if "cache corruption detected" in reset_reason:
        return ("cache_corruption_recovered", f"cache corruption recovered ({reset_reason})")
    return ("cache_schema_rebuilt", f"cache schema rebuilt ({reset_reason})")


def _build_resume_contract(
    *,
    metadata: Optional[IndexMetadata],
    schema_version: Optional[str],
    expected_profile: str,
    cached_profile: Optional[str],
    reset_reason: Optional[str],
    needs_reindex: bool,
    last_success_resume_fingerprint: Optional[str],
    last_success_resume_at: Optional[str],
    tool_version: str = GLOGGUR_VERSION,
    last_success_tool_version: Optional[str] = None,
) -> Dict[str, object]:
    """Build deterministic resume/fingerprint metadata for status and search JSON payloads."""
    metadata_present = metadata is not None
    metadata_signal = _metadata_reindex_signal(
        metadata_present=metadata_present,
        has_last_success_marker=(
            last_success_resume_fingerprint is not None or last_success_resume_at is not None
        ),
    )
    profile_reason = _profile_reindex_reason(metadata_present, cached_profile, expected_profile)
    tool_version_reason = _tool_version_reindex_reason(
        last_success_tool_version=last_success_tool_version,
        current_tool_version=tool_version,
    )
    reset_signal = _reset_reindex_signal(reset_reason)

    reason_codes: List[str] = []
    reason_details: List[str] = []
    if metadata_signal is not None:
        reason_codes.append(metadata_signal[0])
        reason_details.append(metadata_signal[1])
        if metadata_signal[0] != "missing_index_metadata":
            reason_codes.append("missing_index_metadata")
            metadata_reason = _metadata_reindex_reason(metadata_present=False)
            reason_details.append(metadata_reason or "index metadata missing")
    if profile_reason is not None:
        code = "missing_cached_profile" if cached_profile is None else "embedding_profile_changed"
        reason_codes.append(code)
        reason_details.append(profile_reason)
    if tool_version_reason is not None:
        reason_codes.append("tool_version_changed")
        reason_details.append(tool_version_reason)
    if reset_signal is not None:
        reason_codes.append(reset_signal[0])
        reason_details.append(reset_signal[1])
    effective_needs_reindex = needs_reindex or tool_version_reason is not None

    workspace_path_hash = _hash_content(os.path.abspath(os.getcwd()))
    metadata_digest = _index_metadata_digest(metadata)
    cached_tool_version = last_success_tool_version or tool_version
    expected_resume_fingerprint = _stable_fingerprint(
        {
            "workspace_path_hash": workspace_path_hash,
            "schema_version": schema_version or CACHE_SCHEMA_VERSION,
            "index_profile": expected_profile,
            "metadata_digest": metadata_digest,
            "tool_version": tool_version,
        }
    )
    cached_resume_fingerprint = _stable_fingerprint(
        {
            "workspace_path_hash": workspace_path_hash,
            "schema_version": schema_version,
            "index_profile": cached_profile,
            "metadata_digest": metadata_digest,
            "tool_version": cached_tool_version,
        }
    )
    resume_remediation = {
        code: RESUME_REMEDIATION.get(code, [DEFAULT_RESUME_REMEDIATION])
        for code in reason_codes
    }

    return {
        "resume_decision": "reindex_required" if effective_needs_reindex else "resume_ok",
        "resume_reason_codes": reason_codes,
        "resume_reason_details": reason_details,
        "resume_remediation": resume_remediation,
        "workspace_path_hash": workspace_path_hash,
        "expected_resume_fingerprint": expected_resume_fingerprint,
        "cached_resume_fingerprint": cached_resume_fingerprint,
        "resume_fingerprint_match": expected_resume_fingerprint == cached_resume_fingerprint,
        "last_success_resume_fingerprint": last_success_resume_fingerprint,
        "last_success_resume_at": last_success_resume_at,
        "tool_version": tool_version,
        "last_success_tool_version": last_success_tool_version,
        "last_success_tool_version_match": (
            last_success_tool_version == tool_version
            if last_success_tool_version is not None
            else None
        ),
        "last_success_resume_fingerprint_match": (
            last_success_resume_fingerprint == expected_resume_fingerprint
            if last_success_resume_fingerprint is not None
            else None
        ),
    }


def _persist_last_success_resume_state(config: GloggurConfig, cache: CacheManager) -> None:
    """Persist last-success resume fingerprint/timestamp when index state is reusable."""
    metadata = cache.get_index_metadata()
    if metadata is None:
        return
    resume_contract = _build_resume_contract(
        metadata=metadata,
        schema_version=cache.get_schema_version(),
        expected_profile=config.embedding_profile(),
        cached_profile=cache.get_index_profile(),
        reset_reason=cache.last_reset_reason,
        needs_reindex=False,
        last_success_resume_fingerprint=cache.get_last_success_resume_fingerprint(),
        last_success_resume_at=cache.get_last_success_resume_at(),
        tool_version=GLOGGUR_VERSION,
        last_success_tool_version=cache.get_last_success_tool_version(),
    )
    if resume_contract["resume_decision"] != "resume_ok":
        return
    fingerprint = resume_contract["expected_resume_fingerprint"]
    if isinstance(fingerprint, str):
        cache.set_last_success_resume_fingerprint(fingerprint)
    cache.set_last_success_resume_at(metadata.last_updated.isoformat())
    cache.set_last_success_tool_version(GLOGGUR_VERSION)


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


def _normalize_reason_counts(payload: object) -> Dict[str, int]:
    """Normalize reason-count payloads into a stable positive-int mapping."""
    normalized: Dict[str, int] = {}
    if not isinstance(payload, dict):
        return normalized
    for raw_reason, raw_count in payload.items():
        reason = str(raw_reason)
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        normalized[reason] = normalized.get(reason, 0) + count
    return normalized


def _read_failed_count(payload: Dict[str, object]) -> int:
    """Read failed/error_count from a watch payload with safe int coercion."""
    raw_failed = payload.get("failed", payload.get("error_count", 0))
    try:
        return int(raw_failed)
    except (TypeError, ValueError):
        return 0


def _collect_watch_failure_signals(state: Dict[str, object]) -> tuple[int, Dict[str, int]]:
    """Collect fail-closed failure counters/reasons from watch state + last_batch."""
    normalized_reasons = _normalize_reason_counts(state.get("failed_reasons"))
    failed_count = _read_failed_count(state)

    last_batch_payload = state.get("last_batch")
    if isinstance(last_batch_payload, dict):
        last_batch_reasons = _normalize_reason_counts(last_batch_payload.get("failed_reasons"))
        for reason, count in last_batch_reasons.items():
            normalized_reasons[reason] = normalized_reasons.get(reason, 0) + count
        last_batch_failed = _read_failed_count(last_batch_payload)
        if failed_count <= 0:
            failed_count = last_batch_failed
        if failed_count > 0 and not normalized_reasons:
            normalized_reasons["watch_last_batch_inconsistent"] = failed_count

    if failed_count > 0 and not normalized_reasons:
        normalized_reasons["watch_state_inconsistent"] = failed_count
    elif failed_count <= 0 and normalized_reasons:
        failed_count = sum(normalized_reasons.values())

    return failed_count, normalized_reasons


def _normalize_watch_status(running: bool, state: Dict[str, object]) -> str:
    """Return a status label consistent with observed liveness."""

    raw_status = state.get("status")
    status = raw_status if isinstance(raw_status, str) else ""
    normalized = status.strip().lower()

    if running:
        failed_count, _normalized_reasons = _collect_watch_failure_signals(state)
        if failed_count > 0:
            return "running_with_errors"
        if normalized in {"running", "running_with_errors", "starting"}:
            return normalized
        return "running"

    if normalized in {"failed_startup", "stopped"}:
        return normalized
    return "stopped"


def _build_watch_failure_contract(state: Dict[str, object]) -> Dict[str, object]:
    """Build deterministic watch failure codes/guidance from state counters."""
    _failed_count, normalized_reasons = _collect_watch_failure_signals(state)
    if not normalized_reasons:
        return {}

    return {
        "failed_reasons": normalized_reasons,
        "failure_codes": sorted(normalized_reasons),
        "failure_guidance": {
            reason: FAILURE_REMEDIATION.get(
                reason,
                WATCH_STATUS_FAILURE_REMEDIATION.get(
                    reason,
                    [DEFAULT_WATCH_STATUS_FAILURE_REMEDIATION],
                ),
            )
            for reason in sorted(normalized_reasons)
        },
    }


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
    last_success_tool_version = cache.get_last_success_tool_version()
    metadata_reason = _metadata_reindex_reason(metadata is not None)
    profile_reason = _profile_reindex_reason(
        metadata_present=metadata is not None,
        cached_profile=cached_profile,
        expected_profile=expected_profile,
    )
    tool_version_reason = _tool_version_reindex_reason(
        last_success_tool_version=last_success_tool_version,
        current_tool_version=GLOGGUR_VERSION,
    )
    reindex_reason = metadata_reason or profile_reason or tool_version_reason
    if cache.last_reset_reason:
        reset_label = "cache schema rebuilt"
        if "cache corruption detected" in cache.last_reset_reason:
            reset_label = "cache corruption recovered"
        reindex_reason = (
            f"{reset_label} "
            f"({cache.last_reset_reason})"
        )
    needs_reindex = metadata is None or reindex_reason is not None
    resume_contract = _build_resume_contract(
        metadata=metadata,
        schema_version=schema_version,
        expected_profile=expected_profile,
        cached_profile=cached_profile,
        reset_reason=cache.last_reset_reason,
        needs_reindex=needs_reindex,
        last_success_resume_fingerprint=cache.get_last_success_resume_fingerprint(),
        last_success_resume_at=cache.get_last_success_resume_at(),
        tool_version=GLOGGUR_VERSION,
        last_success_tool_version=last_success_tool_version,
    )
    return {
        "cache_dir": config.cache_dir,
        "metadata": metadata.model_dump(mode="json") if metadata else None,
        "schema_version": schema_version,
        "expected_index_profile": expected_profile,
        "cached_index_profile": cached_profile,
        "needs_reindex": needs_reindex,
        "reindex_reason": reindex_reason,
        "total_symbols": len(cache.list_symbols()),
        **resume_contract,
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


def _profile_matches_filter(cached_profile: Optional[str], profile_filter: str) -> bool:
    """Return whether a cached profile matches a user-provided clear-cache filter."""
    if not cached_profile:
        return False
    normalized_filter = profile_filter.strip().lower()
    if not normalized_filter:
        return False
    normalized_profile = cached_profile.strip().lower()
    if ":" in normalized_filter:
        return normalized_profile == normalized_filter
    return normalized_filter in normalized_profile


def _build_index_failure_contract(failed_reasons: Dict[str, int]) -> Dict[str, object]:
    """Build deterministic machine-readable failure codes and remediation for index payloads."""
    if not failed_reasons:
        return {}
    return {
        "failure_codes": sorted(failed_reasons),
        "failure_guidance": {
            reason: FAILURE_REMEDIATION.get(reason, [DEFAULT_INDEX_FAILURE_REMEDIATION])
            for reason in sorted(failed_reasons)
        },
    }


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
        if not as_json:
            def _scan(done: int, total: int, status: str) -> None:
                _ = status
                click.echo(f"\rScanning: {done}/{total} files    ", nl=False, err=True)

            indexer._scan_callback = _scan

            if embedding is not None:
                def _progress(done: int, total: int) -> None:
                    click.echo(
                        f"\rEmbedding: {done}/{total} symbols    ",
                        nl=False,
                        err=True,
                    )
                indexer._progress_callback = _progress

        if os.path.isdir(path):
            result = indexer.index_repository(path)
            if not as_json:
                click.echo("", err=True)  # newline after final progress line
            if result.failed == 0:
                _persist_last_success_resume_state(config, cache)
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
        stale_cleanup = (
            indexer.prune_missing_file_entries()
            if files_considered
            else {
                "files_removed": 0,
                "symbols_removed": 0,
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
            }
        )
        if vector_store and files_considered:
            vector_store.save()
        consistency = (
            indexer.validate_vector_metadata_consistency()
            if files_considered
            else {"failed": 0, "failed_reasons": {}, "failed_samples": []}
        )
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
        files_removed = int(stale_cleanup.get("files_removed", 0))
        symbols_removed = int(stale_cleanup.get("symbols_removed", 0))
        stale_failed = int(stale_cleanup.get("failed", 0))
        failed += stale_failed
        stale_failed_reasons = stale_cleanup.get("failed_reasons", {})
        if isinstance(stale_failed_reasons, dict):
            for reason, count in stale_failed_reasons.items():
                reason_key = str(reason)
                reason_count = int(count)
                failed_reasons[reason_key] = failed_reasons.get(reason_key, 0) + reason_count
        stale_failed_samples = stale_cleanup.get("failed_samples", [])
        if isinstance(stale_failed_samples, list):
            for sample in stale_failed_samples:
                if len(failed_samples) >= 5:
                    break
                failed_samples.append(str(sample))
        failed += int(consistency.get("failed", 0))
        consistency_failed_reasons = consistency.get("failed_reasons", {})
        if isinstance(consistency_failed_reasons, dict):
            for reason, count in consistency_failed_reasons.items():
                reason_key = str(reason)
                reason_count = int(count)
                failed_reasons[reason_key] = failed_reasons.get(reason_key, 0) + reason_count
        consistency_failed_samples = consistency.get("failed_samples", [])
        if isinstance(consistency_failed_samples, list):
            for sample in consistency_failed_samples:
                if len(failed_samples) >= 5:
                    break
                failed_samples.append(str(sample))
        if outcome and outcome.status != "failed" and failed == 0:
            metadata = IndexMetadata(
                version=config.index_version,
                total_symbols=len(cache.list_symbols()),
                indexed_files=cache.count_files(),
            )
            cache.set_index_metadata(metadata)
            cache.set_index_profile(config.embedding_profile())
            _persist_last_success_resume_state(config, cache)
        result = {
            "files_considered": files_considered,
            "indexed": indexed,
            "unchanged": unchanged,
            "failed": failed,
            "failed_reasons": failed_reasons,
            "failed_samples": failed_samples,
            "files_changed": indexed,
            "files_removed": files_removed,
            "symbols_added": outcome.symbols_added if outcome else 0,
            "symbols_updated": outcome.symbols_updated if outcome else 0,
            "symbols_removed": (outcome.symbols_removed if outcome else 0) + symbols_removed,
            "indexed_files": indexed,
            "skipped_files": unchanged,
            "indexed_symbols": indexed_symbols,
            "duration_ms": 0,
        }
        result.update(_build_index_failure_contract(failed_reasons))
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
    metadata = cache.get_index_metadata()
    metadata_present = metadata is not None
    cached_profile = cache.get_index_profile()
    last_success_tool_version = cache.get_last_success_tool_version()
    metadata_reason = _metadata_reindex_reason(metadata_present)
    profile_reason = _profile_reindex_reason(metadata_present, cached_profile, expected_profile)
    tool_version_reason = _tool_version_reindex_reason(
        last_success_tool_version=last_success_tool_version,
        current_tool_version=GLOGGUR_VERSION,
    )
    reindex_reason = metadata_reason or profile_reason or tool_version_reason
    needs_reindex = reindex_reason is not None
    resume_contract = _build_resume_contract(
        metadata=metadata,
        schema_version=cache.get_schema_version(),
        expected_profile=expected_profile,
        cached_profile=cached_profile,
        reset_reason=cache.last_reset_reason,
        needs_reindex=needs_reindex,
        last_success_resume_fingerprint=cache.get_last_success_resume_fingerprint(),
        last_success_resume_at=cache.get_last_success_resume_at(),
        tool_version=GLOGGUR_VERSION,
        last_success_tool_version=last_success_tool_version,
    )
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
                **resume_contract,
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
    metadata_payload = result.get("metadata")
    if isinstance(metadata_payload, dict):
        metadata_payload.update(resume_contract)
    if stream and as_json:
        for item in result["results"]:
            click.echo(json.dumps(item))
        return
    _emit(result, as_json)


def _inspect_path_class(path: str) -> str:
    """Classify a path into src/tests/scripts/other buckets for inspect summaries."""
    normalized = os.path.normpath(os.path.abspath(path))
    segments = {segment for segment in normalized.split(os.sep) if segment}
    if "src" in segments:
        return "src"
    if "tests" in segments:
        return "tests"
    if "scripts" in segments:
        return "scripts"
    return "other"


def _should_include_inspect_path(
    path: str,
    *,
    include_tests: bool,
    include_scripts: bool,
) -> bool:
    """Return True when a path should be included in inspect traversal."""
    path_class = _inspect_path_class(path)
    if path_class == "tests":
        return include_tests
    if path_class == "scripts":
        return include_scripts
    return True


def _warning_type(warning: str) -> str:
    """Normalize a warning string into a stable warning-type key."""
    return warning.split(" (", 1)[0]


def _symbol_id_file_path(symbol_id: str) -> Optional[str]:
    """Best-effort extraction of file path from symbol id format path:start:name."""
    parts = symbol_id.rsplit(":", 2)
    if len(parts) != 3:
        return None
    return parts[0]


def _build_inspect_warning_summary(
    warning_reports: List[Dict[str, object]],
    *,
    symbol_file_paths: Dict[str, str],
) -> Dict[str, object]:
    """Build deterministic warning summaries for inspect JSON payloads."""
    warning_counts_by_type: Dict[str, int] = {}
    warning_counts_by_path_class: Dict[str, int] = {
        "src": 0,
        "tests": 0,
        "scripts": 0,
        "other": 0,
    }
    report_counts_by_path_class: Dict[str, int] = {
        "src": 0,
        "tests": 0,
        "scripts": 0,
        "other": 0,
    }
    warning_counts_by_file: Dict[str, int] = {}
    total_warnings = 0

    for report in warning_reports:
        symbol_id = report.get("symbol_id")
        if not isinstance(symbol_id, str):
            continue
        report_warnings = report.get("warnings")
        if not isinstance(report_warnings, list):
            continue
        file_path = symbol_file_paths.get(symbol_id) or _symbol_id_file_path(symbol_id)
        path_class = _inspect_path_class(file_path) if file_path else "other"
        report_counts_by_path_class[path_class] += 1
        if file_path:
            warning_counts_by_file[file_path] = warning_counts_by_file.get(file_path, 0) + len(
                report_warnings
            )
        for warning in report_warnings:
            if not isinstance(warning, str):
                continue
            total_warnings += 1
            warning_type = _warning_type(warning)
            warning_counts_by_type[warning_type] = warning_counts_by_type.get(warning_type, 0) + 1
            warning_counts_by_path_class[path_class] += 1

    top_files = sorted(
        warning_counts_by_file.items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]
    return {
        "total_warnings": total_warnings,
        "by_warning_type": dict(sorted(warning_counts_by_type.items())),
        "by_path_class": warning_counts_by_path_class,
        "reports_by_path_class": report_counts_by_path_class,
        "top_files": [
            {
                "file": file_path,
                "warnings": count,
                "path_class": _inspect_path_class(file_path),
            }
            for file_path, count in top_files
        ],
    }


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
    "--include-tests",
    is_flag=True,
    default=False,
    help="Include `tests/` paths when inspecting directories.",
)
@click.option(
    "--include-scripts",
    is_flag=True,
    default=False,
    help="Include `scripts/` paths when inspecting directories.",
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
    include_tests: bool,
    include_scripts: bool,
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
    symbol_file_paths: Dict[str, str] = {}
    include_tests_effective = include_tests
    include_scripts_effective = include_scripts
    paths = [path]
    if os.path.isdir(path):
        root_class = _inspect_path_class(path)
        include_tests_effective = include_tests or root_class == "tests"
        include_scripts_effective = include_scripts or root_class == "scripts"
        paths = []
        for root, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()
            dirs[:] = [d for d in dirs if d not in config.excluded_dirs]
            for filename in files:
                full_path = os.path.join(root, filename)
                if not any(full_path.endswith(ext) for ext in config.supported_extensions):
                    continue
                if not _should_include_inspect_path(
                    full_path,
                    include_tests=include_tests_effective,
                    include_scripts=include_scripts_effective,
                ):
                    continue
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
            symbol_file_paths[symbol.id] = file_path
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
        semantic_min_code_chars=config.docstring_semantic_min_code_chars,
        kind_thresholds=config.docstring_semantic_kind_thresholds,
    )
    for report in reports:
        cache.set_audit_warnings(report.symbol_id, report.warnings)
    for file_path, content_hash in processed_files:
        cache.upsert_audit_file_metadata(
            AuditFileMetadata(path=file_path, content_hash=content_hash)
        )
    warning_reports = [report for report in reports if report.warnings]
    warning_report_payloads = [report.__dict__ for report in warning_reports]
    payload = {
        "path": path,
        "symbol_ids": list(symbol_ids) if symbol_ids else None,
        "warnings": warning_report_payloads,
        "total": len(warning_reports),
        "reports": [report.__dict__ for report in reports],
        "reports_total": len(reports),
        "inspect_payload_schema_version": INSPECT_PAYLOAD_SCHEMA_VERSION,
        "inspect_scope": {
            "default_src_focus": not include_tests and not include_scripts,
            "include_tests": include_tests_effective,
            "include_scripts": include_scripts_effective,
        },
        "warning_summary": _build_inspect_warning_summary(
            warning_report_payloads,
            symbol_file_paths=symbol_file_paths,
        ),
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
    payload.update(_build_watch_failure_contract(payload))
    payload["running"] = running
    payload["status"] = _normalize_watch_status(running=running, state=payload)
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
@click.option(
    "--profile-filter",
    type=str,
    default=None,
    help=(
        "Only clear when cached_index_profile matches this exact profile "
        "(provider:model) or contains this substring."
    ),
)
@_with_io_failure_handling
def clear_cache(config_path: Optional[str], as_json: bool, profile_filter: Optional[str]) -> None:
    """Clear the index cache."""
    resolved_config_path = _normalize_config_path(config_path)
    config = _load_config(resolved_config_path)
    with cache_write_lock(config.cache_dir):
        cache = _create_cache_manager(config.cache_dir)
        cached_profile = cache.get_index_profile()
        if profile_filter and not _profile_matches_filter(cached_profile, profile_filter):
            _emit(
                {
                    "cleared": False,
                    "cache_dir": config.cache_dir,
                    "cached_index_profile": cached_profile,
                    "profile_filter": profile_filter,
                    "reason": "profile_filter_miss",
                },
                as_json,
            )
            return
        cache.clear()
        vector_store = VectorStore(
            VectorStoreConfig(config.cache_dir),
            load_existing=False,
        )
        vector_store.clear()
    _emit(
        {
            "cleared": True,
            "cache_dir": config.cache_dir,
            "cached_index_profile": cached_profile,
            "profile_filter": profile_filter,
        },
        as_json,
    )


if __name__ == "__main__":
    cli()
