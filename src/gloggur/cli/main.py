from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import unquote, urlparse
from uuid import uuid4

import click
import yaml

from gloggur import __version__ as GLOGGUR_VERSION
from gloggur.adapters.registry import AdapterResolutionError
from gloggur.archive_utils import (
    ArchiveFileSource,
    create_deterministic_tar_gz,
)
from gloggur.archive_utils import (
    sha256_bytes as archive_sha256_bytes,
)
from gloggur.archive_utils import (
    sha256_file as archive_sha256_file,
)
from gloggur.audit.docstring_audit import DocstringAuditReport, audit_docstrings
from gloggur.byte_spans import (
    LineByteSpanIndex,
    RepoPathResolutionError,
    discover_repo_root,
    is_path_absolute,
    resolve_repo_relative_path,
)
from gloggur.config import GloggurConfig
from gloggur.coverage_importers import (
    CoverageImportError,
    create_coverage_importer,
    list_coverage_importers,
)
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import (
    EmbeddingProviderError,
    format_embedding_error_message,
    wrap_embedding_error,
)
from gloggur.embeddings.factory import create_embedding_provider, list_embedding_provider_adapters
from gloggur.graph.service import GraphService
from gloggur.indexer.cache import (
    CACHE_SCHEMA_VERSION,
    CacheConfig,
    CacheManager,
    CacheRecoveryError,
)
from gloggur.indexer.concurrency import cache_write_lock
from gloggur.indexer.indexer import FAILURE_REMEDIATION, Indexer
from gloggur.indexer.shared import FileTimingTrace, ParsedFileSnapshot
from gloggur.io_failures import StorageIOError, format_io_error_message, wrap_io_error
from gloggur.models import AuditFileMetadata, IndexMetadata
from gloggur.parsers.coverage import CoverageIngester
from gloggur.parsers.registry import ParserRegistry
from gloggur.runtime.hosts import create_runtime_host, list_runtime_hosts
from gloggur.search import hybrid_search as hybrid_search_module
from gloggur.search.hybrid_search import HybridSearch
from gloggur.search.router import (
    SearchConstraints,
    SearchRouter,
    load_search_router_config,
)
from gloggur.storage.backends import create_storage_backend, list_storage_backends
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.support import (
    SupportCallbacks,
    SupportContractError,
)
from gloggur.support import (
    collect_support_bundle as collect_support_bundle_impl,
)
from gloggur.support import (
    run_support_command as run_support_command_impl,
)
from gloggur.symbol_index.indexer import SymbolIndexer
from gloggur.symbol_index.store import SymbolIndexStore, SymbolIndexStoreConfig
from gloggur.watch.service import (
    DEFAULT_WATCH_FAILURE_REMEDIATION,
    WatchService,
    is_process_running,
    load_watch_state,
    utc_now_iso,
)


@click.group()
def cli() -> None:
    """Gloggur CLI for indexing, search, coverage, and docstring inspection."""


@cli.group()
def coverage() -> None:
    """Manage dynamic coverage mappings (test code to production capabilities)."""


@cli.group()
def adapters() -> None:
    """Inspect adapter registry state and discoverability."""


@adapters.command("list")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
def adapters_list(config_path: str | None, as_json: bool) -> None:
    """List discoverable adapters across parser/coverage/embedding/storage/runtime."""
    _with_io_failure_handling(_adapters_list_impl)(
        config_path=config_path,
        as_json=as_json,
    )


def _adapters_list_impl(config_path: str | None, as_json: bool) -> None:
    """List discoverable adapters across parser/coverage/embedding/storage/runtime."""
    config = _load_config(config_path)
    payload = {
        "active": {
            "embedding_provider": config.embedding_provider,
            "storage_backend": config.storage_backend(),
            "runtime_host": config.runtime_host(),
        },
        "available": {
            "parsers": ParserRegistry.available_adapters(),
            "coverage_importers": list_coverage_importers(),
            "embedding_providers": list_embedding_provider_adapters(),
            "storage_backends": list_storage_backends(),
            "runtime_hosts": list_runtime_hosts(),
        },
    }
    _emit(payload, as_json=as_json)


INSPECT_PAYLOAD_SCHEMA_VERSION = "1"
INSPECT_PAYLOAD_SCHEMA_POLICY_VERSION = "1"
INSPECT_PAYLOAD_SCHEMA_BUMP_POLICY: dict[str, object] = {
    "policy_version": INSPECT_PAYLOAD_SCHEMA_POLICY_VERSION,
    "bump_required_for": [
        "remove_or_rename_existing_field",
        "change_existing_field_type",
        "change_existing_field_semantics",
        "remove_or_rename_existing_failure_reason_code",
    ],
    "bump_not_required_for": [
        "add_new_optional_field",
        "add_new_failure_reason_code",
    ],
}
DEFAULT_INSPECT_FAILURE_REMEDIATION = (
    "Inspect failed_samples, then rerun "
    "`gloggur inspect <path> --json --force` "
    "after resolving the file-level failure."
)
DEFAULT_CLI_FAILURE_REMEDIATION = (
    "Resolve the CLI precondition failure and rerun the command "
    "with --json for machine-readable diagnostics."
)
ARTIFACT_MANIFEST_SCHEMA_VERSION = "1"
DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_MAX_REQUERY_ATTEMPTS = 1
DEFAULT_SEARCH_CONTEXT_RADIUS = 12
DEFAULT_EVIDENCE_MIN_CONFIDENCE = 0.6
DEFAULT_EVIDENCE_MIN_ITEMS = 1
MAX_REQUERY_TOP_K = 64
REQUERY_STRATEGY_TOP_K_EXPANSION = "top_k_expansion"
_HF_CACHE_MODEL_SEGMENT_RE = re.compile(r"^models--(?P<org>[^/\\]+)--(?P<repo>[^/\\]+)$")
CLI_FAILURE_REMEDIATION: dict[str, list[str]] = {
    "cli_usage_error": [
        "Fix command arguments/options and rerun with `--help` for usage details.",
    ],
    "watch_mode_conflict": [
        "Use exactly one mode override: `--foreground` or `--daemon`.",
    ],
    "watch_path_missing": [
        "Set `watch_path` to an existing path via config or run `gloggur watch init <path>`.",
    ],
    "watch_mode_invalid": [
        "Set watch mode to `foreground` or `daemon` in config, or pass an explicit mode flag.",
    ],
    "allow_tool_version_drift_env_invalid": [
        "Set GLOGGUR_ALLOW_TOOL_VERSION_DRIFT to one of: 1, true, yes, on, 0, false, no, off.",
        "Or unset GLOGGUR_ALLOW_TOOL_VERSION_DRIFT to rely on CLI flags only.",
    ],
    "local_fallback_env_unsupported": [
        "Unset GLOGGUR_LOCAL_FALLBACK; deterministic local fallback embeddings "
        "are no longer supported.",
        "For deterministic test-only embeddings, set GLOGGUR_EMBEDDING_PROVIDER=test.",
    ],
    "support_command_invalid": [
        "Pass a Gloggur subcommand after `--`, for example `gloggur support run -- status --json`.",
        "Do not pass executable paths or recursive `support` child commands.",
    ],
    "support_session_missing": [
        "List or inspect the existing session ids under `.gloggur/support/sessions`.",
        "Retry with a valid `--session <id>` value or omit `--session` for a fresh snapshot.",
    ],
    "support_session_invalid": [
        "Use a session id created by `gloggur support run` or `gloggur support collect`.",
        "If the session metadata is corrupted, reproduce the failure and collect a fresh bundle.",
    ],
    "support_destination_exists": [
        "Choose a new bundle destination path or pass `--overwrite` "
        "to replace the existing archive.",
    ],
    "artifact_source_missing": [
        "Set --source to an existing cache directory or run "
        "`gloggur index . --json` to create one.",
    ],
    "artifact_source_not_directory": [
        "Set --source to a directory path containing cache artifacts.",
    ],
    "artifact_source_uninitialized": [
        "Run `gloggur index . --json` before publishing so metadata and vectors are present.",
    ],
    "coverage_file_missing": [
        "Set path to a valid coverage file that exists on disk.",
    ],
    "coverage_file_invalid": [
        "Coverage JSON schema is malformed or invalid.",
    ],
    "coverage_sqlite_invalid": [
        "Coverage SQLite schema is missing or unreadable.",
    ],
    "artifact_destination_unsupported": [
        "Use a local path or file:// URI for destination in this command variant.",
    ],
    "artifact_destination_exists": [
        "Choose a new destination path or pass --overwrite to replace the existing artifact.",
    ],
    "artifact_destination_inside_source": [
        "Publish to a path outside the source cache directory to avoid self-referential artifacts.",
    ],
    "artifact_path_missing": [
        "Set --artifact to an existing .tar.gz artifact path "
        "created by `gloggur artifact publish`.",
    ],
    "artifact_path_not_file": [
        "Set --artifact to a regular file path (not a directory).",
    ],
    "extract_path_invalid": [
        "Pass a repo-relative file path under the active workspace root.",
        "Do not use absolute paths or paths that escape with `..`.",
    ],
    "extract_file_missing": [
        "Pass a repo-relative path to an existing file under the active workspace root.",
    ],
    "extract_byte_range_invalid": [
        "Set byte bounds to non-negative integers with end_byte >= start_byte.",
    ],
    "extract_range_out_of_bounds": [
        "Choose byte bounds within the file size reported on disk.",
    ],
    "artifact_archive_invalid": [
        "Artifact is not a readable tar.gz archive; rebuild and republish the artifact.",
    ],
    "artifact_manifest_missing": [
        "Artifact is missing manifest.json; republish with `gloggur artifact publish --json`.",
    ],
    "artifact_manifest_invalid": [
        "manifest.json is malformed or missing required fields; "
        "republish artifact from a healthy cache.",
    ],
    "artifact_manifest_schema_unsupported": [
        "Artifact manifest schema is unsupported by this CLI "
        "version; rebuild with a compatible gloggur version.",
    ],
    "artifact_manifest_file_mismatch": [
        "Artifact file checksums/sizes do not match manifest "
        "entries; treat artifact as corrupted and republish.",
    ],
    "artifact_manifest_totals_mismatch": [
        "Manifest aggregate totals do not match file entries; "
        "republish artifact from source cache.",
    ],
    "artifact_restore_destination_exists": [
        "Choose a new restore destination or pass --overwrite "
        "to replace the existing cache directory.",
    ],
    "artifact_restore_destination_not_directory": [
        "Set --destination to a directory path (not an existing file).",
    ],
    "artifact_uploader_command_invalid": [
        "Set --uploader-command to a valid argv-style template "
        "using supported placeholders such as "
        "{artifact_path} and {destination}.",
    ],
    "artifact_uploader_failed": [
        "Inspect uploader stderr/stdout and exit code, then "
        "rerun after fixing the external uploader command "
        "or destination permissions.",
    ],
    "artifact_uploader_timeout": [
        "Increase --uploader-timeout-seconds or fix the remote "
        "uploader path so the command completes within "
        "the expected time.",
    ],
    "artifact_http_upload_failed": [
        "Inspect the HTTP status/body and destination URL, "
        "then rerun after fixing remote auth, permissions, "
        "or presigned URL configuration.",
    ],
    "artifact_http_upload_timeout": [
        "Increase --uploader-timeout-seconds or fix the remote "
        "upload endpoint so the HTTP upload completes "
        "within the expected time.",
    ],
    "search_top_k_invalid": [
        "Set --top-k to a positive integer (>= 1).",
    ],
    "search_confidence_threshold_invalid": [
        "Set --confidence-threshold to a float between 0.0 and 1.0.",
    ],
    "search_max_requery_attempts_invalid": [
        "Set --max-requery-attempts to a non-negative integer.",
    ],
    "search_result_payload_invalid": [
        "Search backend returned malformed results; verify search response contract and rerun.",
    ],
    "search_contract_v1_removed": [
        "Search JSON output now uses ContextPack v2 only (schema_version=2).",
        "Remove deprecated v1-only flags and parse `hits[]` + `summary` "
        "instead of `results` + `metadata`.",
    ],
    "search_cache_not_ready": [
        "Run `gloggur index . --json` and wait for the build to finish before searching.",
        "Inspect `status --json` for resume/build-state details "
        "when the cache remains unavailable.",
    ],
    "search_router_backends_failed": [
        "All retrieval backends failed or returned no usable evidence "
        "within the configured budget.",
        "Retry with `--mode exact` or increase `--time-budget-ms`, "
        "then inspect `debug.backend_errors`.",
    ],
    "search_max_files_invalid": [
        "Set --max-files to a positive integer (>= 1).",
    ],
    "search_max_snippets_invalid": [
        "Set --max-snippets to a positive integer (>= 1).",
    ],
    "search_time_budget_invalid": [
        "Set --time-budget-ms to a positive integer (>= 1).",
    ],
    "search_evidence_min_confidence_invalid": [
        "Set --evidence-min-confidence to a float between 0.0 and 1.0.",
    ],
    "search_evidence_min_items_invalid": [
        "Set --evidence-min-items to an integer >= 1.",
    ],
    "search_evidence_trace_invalid": [
        "Search evidence trace payload was malformed; verify result schema and rerun.",
    ],
    "search_grounding_validation_failed": [
        "Grounding validation failed; retry with broader "
        "query/top-k or adjust evidence thresholds explicitly.",
    ],
    "search_stream_contract_conflict": [
        "Disable --stream when requesting evidence trace/grounding validation payloads.",
    ],
}
INSPECT_FAILURE_REMEDIATION: dict[str, list[str]] = {
    "decode_error": [
        "File contents could not be decoded as UTF-8; convert "
        "the file encoding or exclude it from inspect scope.",
        "Rerun `gloggur inspect <path> --json --force` after normalizing file encoding.",
    ],
    "read_error": [
        "File could not be read from disk; verify file "
        "permissions/path availability and rerun inspect.",
    ],
    "parser_unavailable": [
        "No parser is registered for this file extension; "
        "inspect currently supports configured language "
        "extensions only.",
    ],
    "parse_error": [
        "Parser failed on file contents; inspect syntax "
        "validity and parser compatibility, then rerun "
        "inspect.",
    ],
}
DEFAULT_INDEX_FAILURE_REMEDIATION = (
    "Inspect failed_samples and rerun indexing after resolving the underlying error."
)
DEFAULT_WATCH_STATUS_FAILURE_REMEDIATION = (
    "Inspect watch-state counters and rerun "
    "`gloggur watch stop --json` then "
    "`gloggur watch start --json`."
)
WATCH_STATUS_FAILURE_REMEDIATION: dict[str, list[str]] = {
    "watch_state_inconsistent": [
        "Watch state reports failures without reason codes; "
        "restart watch and verify state-file updates.",
        "If this recurs, run `gloggur index . --json` to re-establish deterministic cache state.",
    ],
    "watch_last_batch_inconsistent": [
        "Watch last_batch reports failures but reason codes "
        "are missing; restart watch and verify daemon "
        "state writes.",
        "Run `gloggur index . --json` if inconsistent batch-state reporting persists.",
    ],
}
DEFAULT_RESUME_REMEDIATION = (
    "Inspect resume_reason_details and rerun `gloggur index . --json` after resolving the issue."
)
RESUME_REMEDIATION: dict[str, list[str]] = {
    "missing_index_metadata": [
        "Run `gloggur index . --json` to rebuild missing metadata.",
        "Avoid reusing cache state until the rebuild completes successfully.",
    ],
    "stale_build_state": [
        "Recorded index build state is stale (the saved build PID is no longer running).",
        "Run `gloggur index . --json` to clear stale build markers and rebuild cleanly.",
    ],
    "build_in_progress": [
        "Wait for the active index build to finish before relying on this cache.",
        "If the build appears stuck, inspect the writer process "
        "and rerun `gloggur index . --json`.",
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
    "tool_version_changed_override": [
        "Override is active: verify retrieval correctness in "
        "this runtime and schedule a full reindex "
        "when possible.",
        "Disable override and rerun `gloggur index . --json` before normal operations.",
    ],
    "cache_corruption_recovered": [
        "Rebuild the index after corruption recovery to restore full symbol coverage.",
    ],
    "cache_schema_rebuilt": [
        "Run a full `gloggur index . --json` after schema rebuild before search operations.",
    ],
}

INDEX_STAGE_ORDER = (
    "bootstrap_model",
    "scan_source",
    "extract_symbols",
    "embed_chunks",
    "persist_cache",
    "validate_integrity",
    "update_symbol_index",
    "commit_metadata",
)

INDEX_STAGE_REASON_CODES: dict[str, tuple[str, ...]] = {
    "extract_symbols": (
        "decode_error",
        "read_error",
        "parser_unavailable",
        "parse_error",
        "chunk_span_integrity_error",
    ),
    "embed_chunks": ("embedding_provider_error",),
    "persist_cache": ("storage_error", "stale_cleanup_error"),
    "validate_integrity": ("vector_metadata_mismatch", "vector_consistency_unverifiable"),
}


@dataclass
class IndexStageRecorder:
    """Deterministic recorder for top-level index stages."""

    order: tuple[str, ...] = INDEX_STAGE_ORDER
    _entries: dict[str, dict[str, object]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Pre-seed every known stage as not-run so output order stays stable."""
        self._entries = {
            name: {
                "name": name,
                "status": "not_run",
                "duration_ms": 0,
                "counts": {},
            }
            for name in self.order
        }

    def record(
        self,
        name: str,
        *,
        status: str,
        duration_ms: int,
        counts: dict[str, object] | None = None,
    ) -> None:
        """Update one stage entry while preserving deterministic field layout."""
        if name not in self._entries:
            raise ValueError(f"unknown index stage: {name}")
        self._entries[name] = {
            "name": name,
            "status": status,
            "duration_ms": max(0, int(duration_ms)),
            "counts": dict(counts or {}),
        }

    def as_payload(self) -> list[dict[str, object]]:
        """Return stage entries in stable order for JSON output."""
        return [dict(self._entries[name]) for name in self.order]


class CLIContractError(click.ClickException):
    """Click exception with stable machine-readable failure code/guidance payload."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        remediation: list[str] | None = None,
    ) -> None:
        """Capture a stable machine-readable code and optional remediation override."""
        super().__init__(message)
        self.error_code = error_code
        self.remediation = remediation

    def to_payload(self) -> dict[str, object]:
        """Return deterministic JSON payload for CLI-contract failures."""
        guidance = self.remediation or CLI_FAILURE_REMEDIATION.get(
            self.error_code,
            [DEFAULT_CLI_FAILURE_REMEDIATION],
        )
        return {
            "error": {
                "type": "cli_contract_error",
                "code": self.error_code,
                "detail": self.message,
                "probable_cause": "Command precondition or argument validation failed.",
                "remediation": guidance,
            },
            "failure_codes": [self.error_code],
            "failure_guidance": {self.error_code: guidance},
        }


def _json_error_envelope(
    *,
    error_code: str,
    error: str,
    stage: str,
    compatibility: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build top-level JSON error envelope for wrapper/CLI contracts."""
    payload: dict[str, object] = {
        "ok": False,
        "error_code": error_code,
        "error": error,
        "stage": stage,
    }
    if compatibility is not None:
        payload["compatibility"] = compatibility
    return payload


def _emit_json_error(payload: dict[str, object]) -> None:
    """Emit a single-line JSON error payload."""
    click.echo(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))


def _emit(payload: dict[str, object], as_json: bool) -> None:
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
        """Normalize CLI/runtime failures into deterministic non-zero command exits."""
        as_json = _resolve_as_json(kwargs)
        try:
            return callback(*args, **kwargs)
        except CLIContractError as exc:
            if as_json:
                _emit_json_error(exc.to_payload())
                raise click.exceptions.Exit(exc.exit_code) from exc
            raise
        except click.ClickException as exc:
            if as_json:
                error_code = "cli_usage_error"
                guidance = CLI_FAILURE_REMEDIATION.get(
                    error_code,
                    [DEFAULT_CLI_FAILURE_REMEDIATION],
                )
                _emit_json_error(
                    {
                        "error": {
                            "type": "cli_usage_error",
                            "code": error_code,
                            "detail": exc.message,
                            "probable_cause": (
                                "Command arguments/options were invalid for this CLI path."
                            ),
                            "remediation": guidance,
                        },
                        "failure_codes": [error_code],
                        "failure_guidance": {error_code: guidance},
                    }
                )
                raise click.exceptions.Exit(exc.exit_code) from exc
            raise
        except StorageIOError as exc:
            if as_json:
                _emit_json_error(exc.to_payload())
            else:
                click.echo(format_io_error_message(exc), err=True)
            raise click.exceptions.Exit(1) from exc
        except EmbeddingProviderError as exc:
            if as_json:
                _emit_json_error(exc.to_payload())
            else:
                click.echo(format_embedding_error_message(exc), err=True)
            raise click.exceptions.Exit(1) from exc

    return _wrapped


def _load_config(config_path: str | None) -> GloggurConfig:
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
    _validate_legacy_local_fallback_env()
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
                "The gloggur config file is malformed or uses an unsupported top-level structure."
            ),
            remediation=[
                f"Fix config syntax and top-level mapping structure in {error_path}.",
                "Or pass --config <path> to a valid .gloggur.yaml/.gloggur.json file.",
            ],
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


def _normalize_config_path(config_path: str | None) -> str | None:
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


def _normalize_watch_paths(config: GloggurConfig, config_path: str | None) -> GloggurConfig:
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


def _canonicalize_hf_snapshot_model(model: str) -> str:
    """Canonicalize HuggingFace cache/snapshot paths to stable org/repo model ids."""
    normalized = model.strip().replace("\\", "/")
    if not normalized:
        return model.strip()
    segments = [segment for segment in normalized.split("/") if segment]
    for segment in segments:
        match = _HF_CACHE_MODEL_SEGMENT_RE.fullmatch(segment)
        if not match:
            continue
        org = match.group("org")
        repo = match.group("repo")
        if org and repo:
            return f"{org}/{repo}"
    return model.strip()


def _canonicalize_embedding_profile(profile: str | None) -> str | None:
    """Return canonical embedding profile for compatibility checks/fingerprinting."""
    if profile is None:
        return None
    normalized = profile.strip()
    if not normalized:
        return normalized
    provider, separator, model = normalized.partition(":")
    if not separator:
        return normalized
    normalized_provider = provider.strip()
    normalized_model = model.strip()
    if normalized_provider.lower() in {"local", "test"}:
        normalized_model = _canonicalize_hf_snapshot_model(normalized_model)
    return f"{normalized_provider}:{normalized_model}"


def _profile_reindex_reason(
    metadata_present: bool,
    cached_profile: str | None,
    expected_profile: str,
) -> str | None:
    """Return a reason why cached index data should be rebuilt."""
    normalized_expected = _canonicalize_embedding_profile(expected_profile) or expected_profile
    normalized_cached = _canonicalize_embedding_profile(cached_profile)
    if cached_profile is None:
        if metadata_present:
            return "cached embedding profile is unknown"
        return None
    if normalized_cached != normalized_expected:
        return (
            "embedding profile changed "
            f"(cached={normalized_cached}, current={normalized_expected})"
        )
    return None


def _normalize_build_state_payload(build_state: object) -> dict[str, object] | None:
    """Normalize build-state payloads from cache/test doubles into a stable shape."""
    if not isinstance(build_state, dict):
        return None
    state_raw = build_state.get("state")
    state = str(state_raw).strip() if state_raw is not None else ""
    if state not in {"building", "interrupted"}:
        return None
    payload: dict[str, object] = {
        "state": state,
        "build_id": build_state.get("build_id"),
        "pid": build_state.get("pid"),
        "started_at": build_state.get("started_at"),
        "updated_at": build_state.get("updated_at"),
        "stage": build_state.get("stage"),
        "cleanup_pending": bool(build_state.get("cleanup_pending")),
    }
    return payload


def _build_state_pid(build_state: dict[str, object] | None) -> int | None:
    """Return normalized positive PID from build-state payload when present."""
    if not isinstance(build_state, dict):
        return None
    raw_pid = build_state.get("pid")
    try:
        pid = int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        return None
    if pid is None or pid <= 0:
        return None
    return pid


def _classify_build_state_for_health(
    build_state: object,
) -> tuple[dict[str, object] | None, bool]:
    """Return normalized build-state plus stale flag for dead-PID in-progress writers."""
    normalized = _normalize_build_state_payload(build_state)
    if normalized is None:
        return None, False
    if str(normalized.get("state")) != "building":
        return normalized, False
    pid = _build_state_pid(normalized)
    if pid is None:
        return normalized, False
    if is_process_running(pid):
        normalized["pid"] = pid
        return normalized, False
    stale_payload = dict(normalized)
    stale_payload["state"] = "interrupted"
    stale_payload["pid"] = pid
    stale_payload["cleanup_pending"] = True
    return stale_payload, True


def _metadata_reindex_reason(
    metadata_present: bool,
    *,
    build_state: dict[str, object] | None = None,
    stale_build_state: bool = False,
) -> str | None:
    """Return reason when index metadata is missing/incomplete."""
    if metadata_present:
        return None
    normalized_build_state = _normalize_build_state_payload(build_state)
    if stale_build_state:
        return "index build state is stale because recorded build pid is no longer running"
    if normalized_build_state is not None:
        state = normalized_build_state["state"]
        if state == "building":
            return "index build in progress and metadata has not been committed yet"
        if state == "interrupted":
            return "index build was interrupted before metadata commit"
    return "index metadata missing (index build in progress, interrupted, or never completed)"


def _metadata_reindex_signal(
    *,
    metadata_present: bool,
    has_last_success_marker: bool,
    build_state: dict[str, object] | None = None,
    stale_build_state: bool = False,
) -> tuple[str, str] | None:
    """Return machine-readable metadata reindex signal with interruption disambiguation."""
    if metadata_present:
        return None
    normalized_build_state = _normalize_build_state_payload(build_state)
    if normalized_build_state is not None:
        if stale_build_state:
            return (
                "stale_build_state",
                "index build state is stale because the recorded build pid is no longer running",
            )
        state = str(normalized_build_state["state"])
        if state == "building":
            return (
                "build_in_progress",
                "index metadata is not committed yet because an index build is still running",
            )
        if state == "interrupted":
            return (
                "index_interrupted",
                "index build was interrupted before metadata commit",
            )
    if has_last_success_marker:
        return (
            "index_interrupted",
            "index metadata missing after a previous successful "
            "index (index run interrupted or failed "
            "before completion)",
        )
    reason = _metadata_reindex_reason(
        metadata_present=False,
        build_state=normalized_build_state,
        stale_build_state=stale_build_state,
    )
    return ("missing_index_metadata", reason or "index metadata missing")


def _tool_version_reindex_reason(
    *,
    last_success_tool_version: str | None,
    current_tool_version: str,
) -> str | None:
    """Return reason when tool-version drift invalidates previously successful cache state."""
    if last_success_tool_version is None:
        return None
    if last_success_tool_version == current_tool_version:
        return None
    return (
        f"tool version changed (cached={last_success_tool_version}, current={current_tool_version})"
    )


def _stable_fingerprint(payload: dict[str, object]) -> str:
    """Return a stable SHA256 fingerprint for JSON-serializable payloads."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _hash_content(serialized)


def _index_metadata_digest(metadata: IndexMetadata | None) -> str | None:
    """Return a deterministic digest of index metadata fields used for resume checks."""
    if metadata is None:
        return None
    payload: dict[str, object] = {
        "version": metadata.version,
        "total_symbols": metadata.total_symbols,
        "indexed_files": metadata.indexed_files,
    }
    return _stable_fingerprint(payload)


def _reset_reindex_signal(reset_reason: str | None) -> tuple[str, str] | None:
    """Map cache reset reason text to a stable machine-readable code + detail."""
    if not reset_reason:
        return None
    if "cache corruption detected" in reset_reason:
        return ("cache_corruption_recovered", f"cache corruption recovered ({reset_reason})")
    return ("cache_schema_rebuilt", f"cache schema rebuilt ({reset_reason})")


def _build_resume_contract(
    *,
    metadata: IndexMetadata | None,
    build_state: dict[str, object] | None = None,
    schema_version: str | None,
    expected_profile: str,
    cached_profile: str | None,
    reset_reason: str | None,
    needs_reindex: bool,
    last_success_resume_fingerprint: str | None,
    last_success_resume_at: str | None,
    tool_version: str = GLOGGUR_VERSION,
    last_success_tool_version: str | None = None,
    allow_tool_version_drift: bool = False,
    stale_build_state: bool = False,
) -> dict[str, object]:
    """Build deterministic resume/fingerprint metadata for status and search JSON payloads."""
    normalized_expected_profile = (
        _canonicalize_embedding_profile(expected_profile) or expected_profile
    )
    normalized_cached_profile = _canonicalize_embedding_profile(cached_profile)
    normalized_build_state = _normalize_build_state_payload(build_state)
    metadata_present = metadata is not None
    metadata_signal = _metadata_reindex_signal(
        metadata_present=metadata_present,
        has_last_success_marker=(
            last_success_resume_fingerprint is not None or last_success_resume_at is not None
        ),
        build_state=normalized_build_state,
        stale_build_state=stale_build_state,
    )
    profile_reason = _profile_reindex_reason(
        metadata_present,
        normalized_cached_profile,
        normalized_expected_profile,
    )
    tool_version_reason = _tool_version_reindex_reason(
        last_success_tool_version=last_success_tool_version,
        current_tool_version=tool_version,
    )
    tool_version_drift_detected = tool_version_reason is not None
    tool_version_override_applied = allow_tool_version_drift and tool_version_drift_detected
    reset_signal = _reset_reindex_signal(reset_reason)

    reason_codes: list[str] = []
    reason_details: list[str] = []
    if metadata_signal is not None:
        reason_codes.append(metadata_signal[0])
        reason_details.append(metadata_signal[1])
        if metadata_signal[0] != "missing_index_metadata":
            reason_codes.append("missing_index_metadata")
            metadata_reason = _metadata_reindex_reason(
                metadata_present=False,
                build_state=normalized_build_state,
                stale_build_state=stale_build_state,
            )
            reason_details.append(metadata_reason or "index metadata missing")
    if profile_reason is not None:
        code = "missing_cached_profile" if cached_profile is None else "embedding_profile_changed"
        reason_codes.append(code)
        reason_details.append(profile_reason)
    if tool_version_drift_detected:
        if tool_version_override_applied:
            reason_codes.append("tool_version_changed_override")
            reason_details.append(
                "tool version drift override enabled "
                f"(cached={last_success_tool_version}, current={tool_version})"
            )
        else:
            reason_codes.append("tool_version_changed")
            reason_details.append(tool_version_reason)
    if reset_signal is not None:
        reason_codes.append(reset_signal[0])
        reason_details.append(reset_signal[1])
    effective_needs_reindex = needs_reindex or (
        tool_version_drift_detected and not allow_tool_version_drift
    )

    workspace_path_hash = _hash_content(os.path.abspath(os.getcwd()))
    metadata_digest = _index_metadata_digest(metadata)
    cached_tool_version = last_success_tool_version or tool_version
    expected_resume_fingerprint = _stable_fingerprint(
        {
            "workspace_path_hash": workspace_path_hash,
            "schema_version": schema_version or CACHE_SCHEMA_VERSION,
            "index_profile": normalized_expected_profile,
            "metadata_digest": metadata_digest,
            "tool_version": tool_version,
        }
    )
    cached_resume_fingerprint = _stable_fingerprint(
        {
            "workspace_path_hash": workspace_path_hash,
            "schema_version": schema_version,
            "index_profile": normalized_cached_profile,
            "metadata_digest": metadata_digest,
            "tool_version": cached_tool_version,
        }
    )
    resume_remediation = {
        code: RESUME_REMEDIATION.get(code, [DEFAULT_RESUME_REMEDIATION]) for code in reason_codes
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
        "tool_version_drift_detected": tool_version_drift_detected,
        "allow_tool_version_drift": allow_tool_version_drift,
        "tool_version_drift_override_applied": tool_version_override_applied,
        "last_success_resume_fingerprint_match": (
            last_success_resume_fingerprint == expected_resume_fingerprint
            if last_success_resume_fingerprint is not None
            else None
        ),
    }


def _integrity_marker(
    *,
    name: str,
    status: str,
    reason_codes: list[str] | None = None,
    detail: str | None = None,
    checked_at: str | None = None,
) -> dict[str, object]:
    """Build one normalized integrity marker payload."""
    return {
        "name": name,
        "status": status,
        "reason_codes": list(reason_codes or []),
        "detail": detail,
        "checked_at": checked_at,
    }


def _default_search_integrity() -> dict[str, object]:
    """Return default missing integrity markers for older caches."""
    return {
        "vector_cache": _integrity_marker(
            name="vector_cache",
            status="missing",
            reason_codes=["vector_integrity_missing"],
            detail="vector/cache integrity marker missing",
        ),
        "chunk_span": _integrity_marker(
            name="chunk_span",
            status="missing",
            reason_codes=["chunk_span_integrity_missing"],
            detail="chunk/span integrity marker missing",
        ),
    }


def _normalize_search_integrity(raw: object) -> dict[str, object]:
    """Normalize persisted integrity markers into a stable payload shape."""
    payload = _default_search_integrity()
    if not isinstance(raw, dict):
        return payload
    for key in ("vector_cache", "chunk_span"):
        candidate = raw.get(key)
        if not isinstance(candidate, dict):
            continue
        status = str(candidate.get("status") or payload[key]["status"])
        reason_codes_raw = candidate.get("reason_codes")
        reason_codes = (
            [str(item) for item in reason_codes_raw if str(item)]
            if isinstance(reason_codes_raw, list)
            else list(payload[key]["reason_codes"])
        )
        payload[key] = _integrity_marker(
            name=key,
            status=status,
            reason_codes=reason_codes,
            detail=(
                str(candidate.get("detail"))
                if candidate.get("detail") is not None
                else str(payload[key]["detail"])
            ),
            checked_at=(
                str(candidate.get("checked_at"))
                if candidate.get("checked_at") is not None
                else None
            ),
        )
    return payload


def _build_search_health_snapshot(
    config: GloggurConfig,
    cache: CacheManager,
    *,
    entrypoint: str,
    contract_version: str,
    allow_tool_version_drift: bool = False,
) -> dict[str, object]:
    """Build shared search-health payload for status, CLI search, and legacy searcher calls."""
    expected_profile = _canonicalize_embedding_profile(config.embedding_profile()) or ""
    metadata = cache.get_index_metadata()
    get_build_state = getattr(cache, "get_build_state", None)
    raw_build_state = get_build_state() if callable(get_build_state) else None
    build_state, stale_build_state = _classify_build_state_for_health(raw_build_state)
    cached_profile = _canonicalize_embedding_profile(cache.get_index_profile())
    last_success_tool_version = cache.get_last_success_tool_version()
    metadata_reason = _metadata_reindex_reason(
        metadata is not None,
        build_state=build_state,
        stale_build_state=stale_build_state,
    )
    profile_reason = _profile_reindex_reason(
        metadata_present=metadata is not None,
        cached_profile=cached_profile,
        expected_profile=expected_profile,
    )
    tool_version_reason = _tool_version_reindex_reason(
        last_success_tool_version=last_success_tool_version,
        current_tool_version=GLOGGUR_VERSION,
    )
    effective_tool_version_reason = tool_version_reason
    if allow_tool_version_drift and tool_version_reason is not None:
        effective_tool_version_reason = None
    reindex_reason = metadata_reason or profile_reason or effective_tool_version_reason
    if cache.last_reset_reason:
        reset_label = "cache schema rebuilt"
        if "cache corruption detected" in cache.last_reset_reason:
            reset_label = "cache corruption recovered"
        reindex_reason = f"{reset_label} ({cache.last_reset_reason})"
    needs_reindex = metadata is None or reindex_reason is not None
    resume_contract = _build_resume_contract(
        metadata=metadata,
        build_state=build_state,
        schema_version=cache.get_schema_version(),
        expected_profile=expected_profile,
        cached_profile=cached_profile,
        reset_reason=cache.last_reset_reason,
        needs_reindex=needs_reindex,
        last_success_resume_fingerprint=cache.get_last_success_resume_fingerprint(),
        last_success_resume_at=cache.get_last_success_resume_at(),
        tool_version=GLOGGUR_VERSION,
        last_success_tool_version=last_success_tool_version,
        allow_tool_version_drift=allow_tool_version_drift,
        stale_build_state=stale_build_state,
    )
    get_search_integrity = getattr(cache, "get_search_integrity", None)
    raw_integrity = get_search_integrity() if callable(get_search_integrity) else None
    search_integrity = _normalize_search_integrity(raw_integrity)
    warning_codes: list[str] = []
    if contract_version == "legacy":
        warning_codes.append("legacy_search_contract")
    if entrypoint == "hybrid_search_legacy":
        warning_codes.append("legacy_search_surface")
    if needs_reindex:
        warning_codes.append("reindex_required")
    if build_state is not None:
        state = str(build_state.get("state"))
        if stale_build_state and "stale_build_state" not in warning_codes:
            warning_codes.append("stale_build_state")
        elif state == "building" and "build_in_progress" not in warning_codes:
            warning_codes.append("build_in_progress")
        elif state == "interrupted" and "index_interrupted" not in warning_codes:
            warning_codes.append("index_interrupted")

    semantic_search_allowed = not needs_reindex
    for key, missing_code, failed_code in (
        ("vector_cache", "vector_integrity_missing", "vector_integrity_failed"),
        ("chunk_span", "chunk_span_integrity_missing", "chunk_span_integrity_failed"),
    ):
        marker = search_integrity[key]
        status = str(marker.get("status") or "missing")
        if status != "passed":
            semantic_search_allowed = False
        if status == "failed":
            warning_codes.append(failed_code)
        elif status != "passed":
            warning_codes.append(missing_code)
        for code in marker.get("reason_codes", []):
            if isinstance(code, str) and code and code not in warning_codes:
                warning_codes.append(code)

    return {
        "entrypoint": entrypoint,
        "contract_version": contract_version,
        "needs_reindex": needs_reindex,
        "reindex_reason": reindex_reason,
        "resume_reason_codes": list(resume_contract["resume_reason_codes"]),
        "warning_codes": warning_codes,
        "semantic_search_allowed": semantic_search_allowed,
        "expected_index_profile": expected_profile,
        "cached_index_profile": cached_profile,
        "search_integrity": search_integrity,
        "build_state": build_state,
        "resume_contract": resume_contract,
    }


def _persist_last_success_resume_state(config: GloggurConfig, cache: CacheManager) -> None:
    """Persist last-success resume fingerprint/timestamp when index state is reusable."""
    metadata = cache.get_index_metadata()
    if metadata is None:
        return
    resume_contract = _build_resume_contract(
        metadata=metadata,
        build_state=None,
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
    if not isinstance(fingerprint, str):
        return
    # Skip all writes when the fingerprint is unchanged — an unchanged re-index must not
    # advance last_success_resume_at (or any other stored state), because doing so would
    # change observable session state even when no indexed content has changed.
    if fingerprint == cache.get_last_success_resume_fingerprint():
        return
    cache.set_last_success_resume_fingerprint(fingerprint)
    cache.set_last_success_resume_at(metadata.last_updated.isoformat())
    cache.set_last_success_tool_version(GLOGGUR_VERSION)


def _new_build_id() -> str:
    """Return a short deterministic-enough build id for staged cache lifecycles."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


def _write_cache_build_state(
    cache: CacheManager,
    *,
    state: str,
    build_id: str,
    started_at: str,
    stage: str | None,
    cleanup_pending: bool = False,
) -> dict[str, object]:
    """Persist cache build-state sidecar updates with stable timestamps."""
    updated_at = datetime.now(timezone.utc).isoformat()
    return cache.write_build_state(
        {
            "state": state,
            "build_id": build_id,
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": updated_at,
            "stage": stage,
            "cleanup_pending": cleanup_pending,
        }
    )


def _warm_embedding_provider(embedding: EmbeddingProvider | None) -> dict[str, object]:
    """Force local-model bootstrap before taking the cache write lock."""
    if embedding is None:
        return {}
    if getattr(embedding, "provider", None) != "local":
        return {}
    return {"embedding_dimension": int(embedding.get_dimension())}


def _current_resource_tracker_pid() -> int | None:
    """Return the stdlib multiprocessing resource_tracker PID when available."""
    try:
        from multiprocessing import resource_tracker
    except ImportError:
        return None
    tracker = getattr(resource_tracker, "_resource_tracker", None)
    pid = getattr(tracker, "_pid", None)
    return pid if isinstance(pid, int) and pid > 0 else None


def _terminate_pid(pid: int, *, sig: int) -> None:
    """Best-effort process termination helper that ignores already-exited targets."""
    try:
        os.kill(pid, sig)
    except OSError:
        return


def _terminate_index_children() -> dict[str, object]:
    """Terminate multiprocessing children and the resource tracker during interrupt cleanup."""
    terminated_children: list[int] = []
    for child in multiprocessing.active_children():
        pid = getattr(child, "pid", None)
        if isinstance(pid, int) and pid > 0:
            terminated_children.append(pid)
        try:
            child.terminate()
        except Exception:
            continue
        join = getattr(child, "join", None)
        if callable(join):
            try:
                join(timeout=0.2)
            except Exception:
                pass
        exitcode = getattr(child, "exitcode", None)
        if exitcode is None and isinstance(pid, int) and pid > 0:
            _terminate_pid(pid, sig=signal.SIGKILL)

    resource_tracker_pid = _current_resource_tracker_pid()
    if isinstance(resource_tracker_pid, int) and resource_tracker_pid > 0:
        _terminate_pid(resource_tracker_pid, sig=signal.SIGTERM)

    return {
        "terminated_child_pids": terminated_children,
        "resource_tracker_pid": resource_tracker_pid,
    }


@contextmanager
def _index_signal_guard(on_interrupt: Callable[[str], None]) -> Iterator[None]:
    """Install temporary SIGINT/SIGTERM handlers that mark interrupted builds and clean up."""
    previous_handlers: dict[int, Any] = {}
    interrupted = False

    def _handler(signum: int, _frame: object) -> None:
        """Handle one termination signal by marking the build interrupted exactly once."""
        nonlocal interrupted
        signal_name = signal.Signals(signum).name
        if not interrupted:
            interrupted = True
            on_interrupt(signal_name)
        raise KeyboardInterrupt(signal_name)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handler)
    try:
        yield
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _stage_status_from_reasons(
    reason_counts: dict[str, int],
    stage_name: str,
) -> str:
    """Return failed/completed for one stage based on categorized failure reasons."""
    relevant_reasons = INDEX_STAGE_REASON_CODES.get(stage_name, ())
    if any(int(reason_counts.get(reason, 0)) > 0 for reason in relevant_reasons):
        return "failed"
    return "completed"


def _record_repository_index_stages(
    recorder: IndexStageRecorder,
    result: object,
) -> None:
    """Record repository indexing stages from IndexResult timings and counters."""
    phase_timings = result.phase_timings_ms
    recorder.record(
        "scan_source",
        status="completed",
        duration_ms=int(phase_timings.get("scan_source", 0) or 0),
        counts={"source_files": len(result.source_files)},
    )
    recorder.record(
        "extract_symbols",
        status=_stage_status_from_reasons(result.failed_reasons, "extract_symbols"),
        duration_ms=int(phase_timings.get("extract_symbols", 0) or 0),
        counts={
            "files_considered": result.files_considered,
            "parsed_files": len(result.parsed_files),
        },
    )
    recorder.record(
        "embed_chunks",
        status=_stage_status_from_reasons(result.failed_reasons, "embed_chunks"),
        duration_ms=int(phase_timings.get("embed_chunks", 0) or 0),
        counts={"indexed_symbols": result.indexed_symbols},
    )
    recorder.record(
        "persist_cache",
        status=_stage_status_from_reasons(result.failed_reasons, "persist_cache"),
        duration_ms=int(phase_timings.get("persist_cache", 0) or 0),
        counts={
            "files_changed": result.files_changed,
            "files_removed": result.files_removed,
            "symbols_removed": result.symbols_removed,
        },
    )
    recorder.record(
        "validate_integrity",
        status=_stage_status_from_reasons(result.failed_reasons, "validate_integrity"),
        duration_ms=int(phase_timings.get("validate_integrity", 0) or 0),
        counts={
            "failed_reasons": sum(
                int(result.failed_reasons.get(reason, 0))
                for reason in INDEX_STAGE_REASON_CODES["validate_integrity"]
            )
        },
    )


def _record_symbol_index_stage(
    recorder: IndexStageRecorder,
    symbol_payload: dict[str, object],
) -> None:
    """Record symbol-index stage outcome."""
    failed = int(symbol_payload.get("failed", 0) or 0)
    recorder.record(
        "update_symbol_index",
        status="failed" if failed > 0 else "completed",
        duration_ms=int(symbol_payload.get("duration_ms", 0) or 0),
        counts={
            "files_considered": int(symbol_payload.get("files_considered", 0) or 0),
            "defs_indexed": int(symbol_payload.get("defs_indexed", 0) or 0),
            "refs_indexed": int(symbol_payload.get("refs_indexed", 0) or 0),
            "failed": failed,
        },
    )


def _record_commit_metadata_stage(
    recorder: IndexStageRecorder,
    *,
    status: str,
    duration_ms: int,
    total_symbols: int = 0,
    indexed_files: int = 0,
) -> None:
    """Record commit/publish stage outcome."""
    recorder.record(
        "commit_metadata",
        status=status,
        duration_ms=duration_ms,
        counts={
            "total_symbols": total_symbols,
            "indexed_files": indexed_files,
        },
    )


def _build_search_not_ready_payload(
    *,
    query: str,
    health: dict[str, object],
) -> dict[str, object]:
    """Build a stable structured error payload for search requests on non-ready caches."""
    resume_contract = health["resume_contract"]
    assert isinstance(resume_contract, dict)
    guidance = CLI_FAILURE_REMEDIATION["search_cache_not_ready"]
    metadata: dict[str, object] = {
        "total_results": 0,
        "needs_reindex": bool(health["needs_reindex"]),
        "reindex_reason": health["reindex_reason"],
        "entrypoint": health["entrypoint"],
        "contract_version": health["contract_version"],
        "warning_codes": list(health["warning_codes"]),
        "semantic_search_allowed": health["semantic_search_allowed"],
        "search_integrity": health["search_integrity"],
        "build_state": health["build_state"],
        "expected_index_profile": health["expected_index_profile"],
        "cached_index_profile": health["cached_index_profile"],
    }
    metadata.update(resume_contract)
    return {
        "query": query,
        "results": [],
        "metadata": metadata,
        "error": {
            "type": "search_unavailable",
            "code": "search_cache_not_ready",
            "category": "cache_not_ready",
            "detail": "Semantic search is unavailable because the cache is not ready.",
            "probable_cause": (
                "Index metadata or integrity markers are missing, interrupted, "
                "or still being built."
            ),
            "remediation": guidance,
        },
        "failure_codes": ["search_cache_not_ready"],
        "failure_guidance": {"search_cache_not_ready": guidance},
    }


def _resolve_config_file_path(config_path: str | None) -> str:
    """Resolve config file path for watch init updates."""
    if config_path:
        return config_path
    for candidate in (".gloggur.yaml", ".gloggur.yml", ".gloggur.json"):
        if os.path.exists(candidate):
            return candidate
    return ".gloggur.yaml"


def _read_config_payload(path: str) -> dict[str, object]:
    """Load config file payload (yaml/json), returning empty dict if missing."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf8") as handle:
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


def _write_config_payload(path: str, payload: dict[str, object]) -> None:
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


def _read_pid_file(path: str) -> int | None:
    """Read PID from pid file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf8") as handle:
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


def _write_watch_state(path: str, updates: dict[str, object]) -> None:
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


def _read_watch_state_for_status(path: str) -> dict[str, object]:
    """Read watch status state file with deterministic failure semantics."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf8") as handle:
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


def _normalize_reason_counts(payload: object) -> dict[str, int]:
    """Normalize reason-count payloads into a stable positive-int mapping."""
    normalized: dict[str, int] = {}
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


def _normalize_failure_codes(payload: object) -> list[str]:
    """Normalize a failure-code payload into a stable de-duplicated list."""
    if not isinstance(payload, list):
        return []

    normalized: list[str] = []
    for raw_code in payload:
        code = str(raw_code).strip()
        if not code or code in normalized:
            continue
        normalized.append(code)
    return normalized


def _read_failed_count(payload: dict[str, object]) -> int:
    """Read failed/error_count from a watch payload with safe int coercion."""
    raw_failed = payload.get("failed", payload.get("error_count", 0))
    try:
        return int(raw_failed)
    except (TypeError, ValueError):
        return 0


def _collect_watch_failure_signals(state: dict[str, object]) -> tuple[int, dict[str, int]]:
    """Collect fail-closed failure counters/reasons from watch state + last_batch."""
    normalized_reasons = _normalize_reason_counts(state.get("failed_reasons"))
    failed_count = _read_failed_count(state)
    raw_status = state.get("status")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else ""

    last_batch_payload = state.get("last_batch")
    if isinstance(last_batch_payload, dict):
        last_batch_reasons = _normalize_reason_counts(last_batch_payload.get("failed_reasons"))
        if not last_batch_reasons:
            last_batch_codes = _normalize_failure_codes(last_batch_payload.get("failure_codes"))
            last_batch_reasons = {code: 1 for code in last_batch_codes}
        for reason, count in last_batch_reasons.items():
            normalized_reasons[reason] = normalized_reasons.get(reason, 0) + count
        last_batch_failed = _read_failed_count(last_batch_payload)
        if failed_count <= 0:
            if last_batch_failed > 0:
                failed_count = last_batch_failed
            elif last_batch_reasons:
                failed_count = sum(last_batch_reasons.values())
        if failed_count > 0 and not normalized_reasons:
            normalized_reasons["watch_last_batch_inconsistent"] = failed_count

    if failed_count > 0 and not normalized_reasons:
        normalized_reasons["watch_state_inconsistent"] = failed_count
    elif failed_count <= 0 and normalized_reasons:
        failed_count = sum(normalized_reasons.values())
    elif failed_count <= 0 and not normalized_reasons and status == "running_with_errors":
        # Fail closed when status indicates unhealthy watch state but counters/codes drift away.
        failed_count = 1
        normalized_reasons["watch_state_inconsistent"] = failed_count

    return failed_count, normalized_reasons


def _normalize_watch_status(running: bool, state: dict[str, object]) -> str:
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


def _build_watch_failure_contract(state: dict[str, object]) -> dict[str, object]:
    """Build deterministic watch failure codes/guidance from state counters."""
    _failed_count, normalized_reasons = _collect_watch_failure_signals(state)
    if not normalized_reasons:
        return {}

    failure_codes = sorted(normalized_reasons)
    failure_guidance = {
        reason: FAILURE_REMEDIATION.get(
            reason,
            WATCH_STATUS_FAILURE_REMEDIATION.get(
                reason,
                [DEFAULT_WATCH_STATUS_FAILURE_REMEDIATION],
            ),
        )
        for reason in failure_codes
    }
    contract: dict[str, object] = {
        "failed_reasons": normalized_reasons,
        "failure_codes": failure_codes,
        "failure_guidance": failure_guidance,
    }

    last_batch_payload = state.get("last_batch")
    if isinstance(last_batch_payload, dict):
        patched_batch = dict(last_batch_payload)
        patched = False
        batch_failure_codes = _normalize_failure_codes(patched_batch.get("failure_codes"))
        if not batch_failure_codes:
            patched_batch["failure_codes"] = failure_codes
            patched = True
        batch_failure_guidance = patched_batch.get("failure_guidance")
        if not (isinstance(batch_failure_guidance, dict) and batch_failure_guidance):
            patched_batch["failure_guidance"] = failure_guidance
            patched = True
        if not _normalize_reason_counts(patched_batch.get("failed_reasons")):
            patched_batch["failed_reasons"] = normalized_reasons
            patched = True
        if _read_failed_count(patched_batch) <= 0 and _failed_count > 0:
            patched_batch["failed"] = _failed_count
            patched_batch["error_count"] = _failed_count
            patched = True
        if patched:
            contract["last_batch"] = patched_batch
    else:
        contract["last_batch"] = {
            "failed": _failed_count,
            "failed_reasons": normalized_reasons,
            "failure_codes": failure_codes,
            "failure_guidance": failure_guidance,
        }

    return contract


def _watch_starting_state_payload(*, watch_path: str, pid: int) -> dict[str, object]:
    """Return fail-closed state reset payload for daemon startup transitions."""
    return {
        "running": True,
        "status": "starting",
        "pid": pid,
        "watch_path": watch_path,
        "last_heartbeat": utc_now_iso(),
        # Clear stale failure/batch counters from prior daemon runs so status is
        # scoped to the current process lifecycle.
        "last_batch": {},
        "failed": 0,
        "error_count": 0,
        "failed_reasons": {},
        "failed_samples": [],
        "failure_codes": [],
        "failure_guidance": {},
        "files_considered": 0,
        "indexed": 0,
        "unchanged": 0,
        "indexed_files": 0,
        "indexed_symbols": 0,
        "skipped_files": 0,
        "last_error": None,
    }


def _uses_default_storage_backend(config: GloggurConfig) -> bool:
    """Return True when storage should use the built-in sqlite/faiss classes."""
    backend_id = config.storage_backend()
    override = config.adapter_module_override("storage", backend_id)
    return backend_id == "sqlite_faiss" and not override


def _create_vector_store(config: GloggurConfig, *, load_existing: bool = True):
    """Create vector backend with legacy-compatible default class behavior."""
    if _uses_default_storage_backend(config):
        vector_config = VectorStoreConfig(config.cache_dir)
        if load_existing:
            return VectorStore(vector_config)
        return VectorStore(vector_config, load_existing=False)
    storage_backend = create_storage_backend(config)
    return storage_backend.create_vector_store(config.cache_dir, load_existing=load_existing)


def _create_metadata_store(config: GloggurConfig):
    """Create metadata backend with legacy-compatible default class behavior."""
    if _uses_default_storage_backend(config):
        return MetadataStore(MetadataStoreConfig(config.cache_dir))
    storage_backend = create_storage_backend(config)
    return storage_backend.create_metadata_store(config.cache_dir)


def _create_watch_service(
    *,
    config: GloggurConfig,
    embedding_provider: EmbeddingProvider | None,
    cache: CacheManager,
    vector_store: object,
):
    """Create watch service with default-class compatibility and runtime-host extension."""
    parser_registry = ParserRegistry(
        extension_map=config.parser_extension_map,
        adapter_overrides=config.adapters if isinstance(config.adapters, dict) else None,
    )
    host_id = config.runtime_host()
    host_override = config.adapter_module_override("runtime", host_id)
    if host_id == "python_local" and not host_override:
        return WatchService(
            config=config,
            embedding_provider=embedding_provider,
            cache=cache,
            vector_store=vector_store,
            parser_registry=parser_registry,
        )
    runtime_host = create_runtime_host(config)
    return runtime_host.build_watch_service(
        config=config,
        embedding_provider=embedding_provider,
        cache=cache,
        vector_store=vector_store,
        parser_registry=parser_registry,
    )


def _initialize_runtime(
    config: GloggurConfig,
    *,
    rebuild_on_profile_change: bool = False,
    write_locked: bool = False,
) -> tuple[GloggurConfig, CacheManager, object]:
    """Create cache/vector runtime for an already-loaded config instance."""
    expected_profile = _canonicalize_embedding_profile(config.embedding_profile()) or ""
    cache = _create_cache_manager(config.cache_dir)
    vector_store = _create_vector_store(config)
    if cache.last_reset_reason:
        vector_store.clear()
    metadata_present = cache.get_index_metadata() is not None
    cached_profile = _canonicalize_embedding_profile(cache.get_index_profile())
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


def _create_runtime(
    config_path: str | None,
    embedding_provider: str | None = None,
    rebuild_on_profile_change: bool = False,
    write_locked: bool = False,
) -> tuple[GloggurConfig, CacheManager, object]:
    """Create config/cache/vector runtime and apply profile rebuild logic."""
    resolved_config_path = _normalize_config_path(config_path)
    config = _load_config(resolved_config_path)
    # Apply CLI provider override in-memory to avoid a second config-file read.
    if embedding_provider:
        config.embedding_provider = embedding_provider
    config = _normalize_watch_paths(config, resolved_config_path)
    return _initialize_runtime(
        config,
        rebuild_on_profile_change=rebuild_on_profile_change,
        write_locked=write_locked,
    )


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
    if error.operation == "execute cache database transaction":
        status_recovery_tokens = (
            "no such table",
            "database schema has changed",
            "file is not a database",
            "database disk image is malformed",
            "database corrupted",
        )
        if any(token in detail for token in status_recovery_tokens):
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


def _build_status_payload(
    config: GloggurConfig,
    cache: CacheManager,
    *,
    allow_tool_version_drift: bool = False,
) -> dict[str, object]:
    """Build status payload from cache metadata/profile state."""
    metadata = cache.get_index_metadata()
    health = _build_search_health_snapshot(
        config,
        cache,
        entrypoint="status_cli",
        contract_version="status_v1",
        allow_tool_version_drift=allow_tool_version_drift,
    )
    resume_contract = health["resume_contract"]
    assert isinstance(resume_contract, dict)
    raw_total_symbols = _cache_total_symbols(cache)
    total_symbols = (
        raw_total_symbols if resume_contract.get("resume_decision") == "resume_ok" else 0
    )
    return {
        "cache_dir": config.cache_dir,
        "metadata": metadata.model_dump(mode="json") if metadata else None,
        "schema_version": cache.get_schema_version(),
        "entrypoint": health["entrypoint"],
        "contract_version": health["contract_version"],
        "expected_index_profile": health["expected_index_profile"],
        "cached_index_profile": health["cached_index_profile"],
        "needs_reindex": health["needs_reindex"],
        "reindex_reason": health["reindex_reason"],
        "warning_codes": health["warning_codes"],
        "semantic_search_allowed": health["semantic_search_allowed"],
        "search_integrity": health["search_integrity"],
        "build_state": health["build_state"],
        "raw_total_symbols": raw_total_symbols,
        "total_symbols": total_symbols,
        **resume_contract,
    }


def _cache_total_symbols(cache: CacheManager) -> int:
    """Return total symbol count with compatibility for older cache test doubles."""
    count_symbols = getattr(cache, "count_symbols", None)
    if callable(count_symbols):
        return int(count_symbols())
    return len(cache.list_symbols())


def _create_status_payload(
    config: GloggurConfig,
    *,
    allow_tool_version_drift: bool = False,
) -> dict[str, object]:
    """Create cache manager and build status payload."""
    cache = _create_cache_manager(config.cache_dir)
    return _build_status_payload(
        config,
        cache,
        allow_tool_version_drift=allow_tool_version_drift,
    )


def _create_watch_status_payload(config: GloggurConfig) -> dict[str, object]:
    """Create watch status payload without emitting CLI output."""
    pid = _read_pid_file(config.watch_pid_file)
    running = is_process_running(pid)
    state = _read_watch_state_for_status(config.watch_state_file)
    payload: dict[str, object] = {
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
    return payload


def _load_support_config(config_path: str | None = None) -> tuple[GloggurConfig, str | None]:
    """Load support-command config and normalize relative watch paths."""
    resolved_config_path = _normalize_config_path(config_path)
    config = _normalize_watch_paths(_load_config(resolved_config_path), resolved_config_path)
    return config, resolved_config_path


def _create_embedding_provider_for_command(
    config: GloggurConfig,
    *,
    require_provider: bool = False,
) -> EmbeddingProvider | None:
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


def _profile_matches_filter(cached_profile: str | None, profile_filter: str) -> bool:
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


def _sha256_file(path: str) -> str:
    """Return SHA256 digest for a file using chunked reads."""
    try:
        return archive_sha256_file(path)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read artifact source file",
            path=path,
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    """Return SHA256 digest for bytes payloads."""
    return archive_sha256_bytes(payload)


def _artifact_rel_path(source_dir: str, file_path: str) -> str:
    """Return deterministic POSIX-style relative path under source_dir."""
    relative = os.path.relpath(file_path, source_dir)
    return relative.replace(os.sep, "/")


def _collect_artifact_file_entries(source_dir: str) -> list[dict[str, object]]:
    """Collect deterministic file metadata for artifact manifest generation."""
    entries: list[dict[str, object]] = []
    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        files.sort()
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = _artifact_rel_path(source_dir, full_path)
            try:
                size_bytes = os.path.getsize(full_path)
            except OSError as exc:
                raise wrap_io_error(
                    exc,
                    operation="read artifact source file metadata",
                    path=full_path,
                ) from exc
            entries.append(
                {
                    "path": rel_path,
                    "bytes": size_bytes,
                    "sha256": _sha256_file(full_path),
                }
            )
    return entries


def _build_artifact_manifest(
    cache: CacheManager,
    *,
    source_dir: str,
    file_entries: list[dict[str, object]],
) -> dict[str, object]:
    """Build artifact manifest with deterministic cache compatibility metadata."""
    metadata = cache.get_index_metadata()
    if metadata is None:
        raise CLIContractError(
            "Index metadata is missing; cannot publish an uninitialized artifact.",
            error_code="artifact_source_uninitialized",
        )
    return {
        "manifest_schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": GLOGGUR_VERSION,
        "cache": {
            "source_dir": source_dir,
            "schema_version": cache.get_schema_version(),
            "index_profile": cache.get_index_profile(),
            "index_metadata": metadata.model_dump(mode="json"),
            "last_success_resume_fingerprint": cache.get_last_success_resume_fingerprint(),
            "last_success_resume_at": cache.get_last_success_resume_at(),
            "last_success_tool_version": cache.get_last_success_tool_version(),
        },
        "files_total": len(file_entries),
        "bytes_total": sum(int(item["bytes"]) for item in file_entries),
        "files": file_entries,
    }


def _resolve_artifact_destination(destination: str, *, default_filename: str) -> str:
    """Resolve destination into a local absolute file path or raise a contract error."""
    resolved = destination
    parsed = urlparse(destination)
    if parsed.scheme:
        if parsed.scheme != "file":
            raise CLIContractError(
                f"Unsupported artifact destination scheme: {parsed.scheme}",
                error_code="artifact_destination_unsupported",
            )
        if parsed.params or parsed.query or parsed.fragment:
            raise CLIContractError(
                "file:// destination must not include params/query/fragment components.",
                error_code="artifact_destination_unsupported",
            )
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            resolved = f"//{parsed.netloc}{parsed.path}"
        else:
            resolved = parsed.path
        resolved = unquote(resolved)
    resolved = os.path.abspath(os.path.expanduser(resolved))
    if resolved.endswith(os.sep) or (os.path.exists(resolved) and os.path.isdir(resolved)):
        resolved = os.path.join(resolved, default_filename)
    return resolved


def _render_artifact_uploader_command(
    uploader_command: str,
    *,
    artifact_path: str,
    destination: str,
    artifact_name: str,
    archive_sha256: str,
    archive_bytes: int,
    manifest_sha256: str,
) -> list[str]:
    """Render uploader command template into argv without invoking a shell."""
    if not uploader_command.strip():
        raise CLIContractError(
            "Uploader command template must not be empty.",
            error_code="artifact_uploader_command_invalid",
        )
    try:
        argv_template = shlex.split(uploader_command)
    except ValueError as exc:
        raise CLIContractError(
            f"Uploader command template could not be parsed: {exc}",
            error_code="artifact_uploader_command_invalid",
        ) from exc
    if not argv_template:
        raise CLIContractError(
            "Uploader command template must produce at least one argv token.",
            error_code="artifact_uploader_command_invalid",
        )

    format_values = {
        "artifact": artifact_path,
        "artifact_path": artifact_path,
        "destination": destination,
        "artifact_name": artifact_name,
        "archive_sha256": archive_sha256,
        "archive_bytes": str(archive_bytes),
        "manifest_sha256": manifest_sha256,
    }
    argv: list[str] = []
    try:
        for token in argv_template:
            argv.append(token.format(**format_values))
    except KeyError as exc:
        missing_key = str(exc).strip("'")
        raise CLIContractError(
            f"Uploader command template references unknown placeholder: {missing_key}",
            error_code="artifact_uploader_command_invalid",
        ) from exc
    return argv


def _run_artifact_uploader_command(
    uploader_command: str,
    *,
    artifact_path: str,
    destination: str,
    artifact_name: str,
    archive_sha256: str,
    archive_bytes: int,
    manifest_sha256: str,
    timeout_seconds: float,
) -> dict[str, object]:
    """Execute external uploader command and return structured success metadata."""
    argv = _render_artifact_uploader_command(
        uploader_command,
        artifact_path=artifact_path,
        destination=destination,
        artifact_name=artifact_name,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        manifest_sha256=manifest_sha256,
    )
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="spawn artifact uploader command",
            path=argv[0],
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CLIContractError(
            (
                "Artifact uploader command timed out after "
                f"{timeout_seconds:.3f}s for destination {destination}: "
                f"{' '.join(argv)}"
            ),
            error_code="artifact_uploader_timeout",
        ) from exc
    if completed.returncode != 0:
        detail_parts = [f"Artifact uploader command exited with code {completed.returncode}"]
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        if stderr:
            detail_parts.append(f"stderr={stderr}")
        if stdout:
            detail_parts.append(f"stdout={stdout}")
        raise CLIContractError(
            "; ".join(detail_parts),
            error_code="artifact_uploader_failed",
        )
    payload: dict[str, object] = {
        "mode": "uploader_command",
        "command": argv,
        "destination": destination,
        "exit_code": completed.returncode,
    }
    if completed.stdout:
        payload["stdout"] = completed.stdout
    if completed.stderr:
        payload["stderr"] = completed.stderr
    return payload


def _upload_artifact_http(
    destination: str,
    *,
    artifact_path: str,
    archive_sha256: str,
    archive_bytes: int,
    manifest_sha256: str,
    timeout_seconds: float,
) -> dict[str, object]:
    """Upload artifact archive via direct HTTP PUT and return structured success metadata."""
    try:
        with open(artifact_path, "rb") as handle:
            payload_bytes = handle.read()
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read artifact upload payload",
            path=artifact_path,
        ) from exc

    request = urllib_request.Request(
        destination,
        data=payload_bytes,
        method="PUT",
        headers={
            "Content-Type": "application/gzip",
            "Content-Length": str(archive_bytes),
            "X-Gloggur-Archive-Sha256": archive_sha256,
            "X-Gloggur-Archive-Bytes": str(archive_bytes),
            "X-Gloggur-Manifest-Sha256": manifest_sha256,
        },
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read()
            response_headers = dict(response.headers.items())
            status_code = int(getattr(response, "status", response.getcode()))
    except urllib_error.HTTPError as exc:
        response_body = exc.read()
        detail = f"HTTP upload failed with status {exc.code} for destination {destination}"
        if response_body:
            body_text = response_body.decode("utf8", errors="replace").strip()
            if body_text:
                detail = f"{detail}; body={body_text}"
        raise CLIContractError(
            detail,
            error_code="artifact_http_upload_failed",
        ) from exc
    except (urllib_error.URLError, TimeoutError) as exc:
        raise CLIContractError(
            (
                "HTTP upload timed out or could not connect after "
                f"{timeout_seconds:.3f}s for destination {destination}: {exc}"
            ),
            error_code="artifact_http_upload_timeout",
        ) from exc

    payload: dict[str, object] = {
        "mode": "http_put",
        "destination": destination,
        "status_code": status_code,
    }
    if response_headers:
        payload["response_headers"] = response_headers
    if response_body:
        payload["response_body"] = response_body.decode("utf8", errors="replace")
    return payload


def _create_artifact_archive(
    source_dir: str,
    *,
    artifact_path: str,
    manifest_bytes: bytes,
    file_entries: list[dict[str, object]],
) -> None:
    """Create a deterministic tar.gz artifact containing cache files and manifest."""
    try:
        file_sources = [
            ArchiveFileSource(
                source_path=Path(source_dir) / str(entry["path"]).replace("/", os.sep),
                archive_path=f"cache/{entry['path']}",
            )
            for entry in file_entries
        ]
        create_deterministic_tar_gz(
            artifact_path,
            file_sources=file_sources,
            extra_files=(("manifest.json", manifest_bytes),),
        )
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="create artifact archive",
            path=artifact_path,
        ) from exc
    except tarfile.TarError as exc:
        raise StorageIOError(
            category="unknown_io_error",
            operation="create artifact archive",
            path=artifact_path,
            probable_cause="Failed to package source cache files into a tar archive.",
            remediation=[
                "Verify source cache files are readable and not concurrently deleted.",
                "Rerun artifact publish after filesystem stability is restored.",
            ],
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


def _validate_artifact_archive(
    artifact_path: str,
    *,
    verify_file_hashes: bool = True,
) -> dict[str, object]:
    """Validate a published artifact archive and return deterministic metadata."""
    normalized_artifact_path = os.path.abspath(os.path.expanduser(artifact_path))
    if not os.path.exists(normalized_artifact_path):
        raise CLIContractError(
            f"Artifact path does not exist: {normalized_artifact_path}",
            error_code="artifact_path_missing",
        )
    if not os.path.isfile(normalized_artifact_path):
        raise CLIContractError(
            f"Artifact path is not a file: {normalized_artifact_path}",
            error_code="artifact_path_not_file",
        )

    try:
        archive_bytes = os.path.getsize(normalized_artifact_path)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read artifact file metadata",
            path=normalized_artifact_path,
        ) from exc
    archive_sha256 = _sha256_file(normalized_artifact_path)

    try:
        with tarfile.open(normalized_artifact_path, "r:gz") as archive:
            try:
                manifest_member = archive.getmember("manifest.json")
            except KeyError as exc:
                raise CLIContractError(
                    "Artifact archive is missing manifest.json",
                    error_code="artifact_manifest_missing",
                ) from exc

            manifest_file = archive.extractfile(manifest_member)
            if manifest_file is None:
                raise CLIContractError(
                    "Artifact manifest.json could not be read from archive",
                    error_code="artifact_manifest_missing",
                )
            manifest_bytes = manifest_file.read()
            try:
                manifest_payload = json.loads(manifest_bytes.decode("utf8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CLIContractError(
                    f"Artifact manifest.json is not valid UTF-8 JSON: {exc}",
                    error_code="artifact_manifest_invalid",
                ) from exc
            if not isinstance(manifest_payload, dict):
                raise CLIContractError(
                    "Artifact manifest.json must be a JSON object",
                    error_code="artifact_manifest_invalid",
                )

            schema_version = manifest_payload.get("manifest_schema_version")
            if schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
                raise CLIContractError(
                    "Artifact manifest schema version is unsupported "
                    f"(found={schema_version}, expected={ARTIFACT_MANIFEST_SCHEMA_VERSION})",
                    error_code="artifact_manifest_schema_unsupported",
                )

            files = manifest_payload.get("files")
            if not isinstance(files, list):
                raise CLIContractError(
                    "Artifact manifest.files must be a list",
                    error_code="artifact_manifest_invalid",
                )

            verified_files = 0
            verified_bytes = 0
            if verify_file_hashes:
                for entry in files:
                    if not isinstance(entry, dict):
                        raise CLIContractError(
                            "Artifact manifest file entry is not an object",
                            error_code="artifact_manifest_invalid",
                        )
                    rel_path = entry.get("path")
                    expected_sha = entry.get("sha256")
                    expected_bytes = entry.get("bytes")
                    if not isinstance(rel_path, str) or not rel_path:
                        raise CLIContractError(
                            "Artifact manifest file entry has invalid path",
                            error_code="artifact_manifest_invalid",
                        )
                    if not isinstance(expected_sha, str) or not expected_sha:
                        raise CLIContractError(
                            f"Artifact manifest entry for {rel_path} has invalid sha256",
                            error_code="artifact_manifest_invalid",
                        )
                    if not isinstance(expected_bytes, int) or expected_bytes < 0:
                        raise CLIContractError(
                            f"Artifact manifest entry for {rel_path} has invalid bytes",
                            error_code="artifact_manifest_invalid",
                        )

                    archive_member_name = f"cache/{rel_path}"
                    try:
                        file_member = archive.getmember(archive_member_name)
                    except KeyError as exc:
                        raise CLIContractError(
                            f"Artifact is missing expected file: {archive_member_name}",
                            error_code="artifact_manifest_file_mismatch",
                        ) from exc
                    file_obj = archive.extractfile(file_member)
                    if file_obj is None:
                        raise CLIContractError(
                            f"Artifact file could not be read: {archive_member_name}",
                            error_code="artifact_manifest_file_mismatch",
                        )
                    file_bytes = file_obj.read()
                    actual_bytes = len(file_bytes)
                    actual_sha = _sha256_bytes(file_bytes)
                    if actual_bytes != expected_bytes or actual_sha != expected_sha:
                        raise CLIContractError(
                            "Artifact file checksum/size mismatch for "
                            f"{archive_member_name} (expected_bytes={expected_bytes}, "
                            f"actual_bytes={actual_bytes})",
                            error_code="artifact_manifest_file_mismatch",
                        )
                    verified_files += 1
                    verified_bytes += actual_bytes

            manifest_files_total = manifest_payload.get("files_total")
            manifest_bytes_total = manifest_payload.get("bytes_total")
            if not isinstance(manifest_files_total, int) or not isinstance(
                manifest_bytes_total, int
            ):
                raise CLIContractError(
                    "Artifact manifest totals (files_total/bytes_total) must be integers",
                    error_code="artifact_manifest_invalid",
                )

            computed_files_total = len(files)
            computed_bytes_total = 0
            for entry in files:
                if not isinstance(entry, dict):
                    continue
                entry_bytes = entry.get("bytes")
                if isinstance(entry_bytes, int) and entry_bytes >= 0:
                    computed_bytes_total += entry_bytes

            if (
                manifest_files_total != computed_files_total
                or manifest_bytes_total != computed_bytes_total
            ):
                raise CLIContractError(
                    "Artifact manifest totals do not match file entries "
                    f"(files_total={manifest_files_total}, expected_files={computed_files_total}, "
                    f"bytes_total={manifest_bytes_total}, expected_bytes={computed_bytes_total})",
                    error_code="artifact_manifest_totals_mismatch",
                )
    except CLIContractError:
        raise
    except tarfile.TarError as exc:
        raise CLIContractError(
            f"Artifact archive is unreadable: {type(exc).__name__}: {exc}",
            error_code="artifact_archive_invalid",
        ) from exc
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read artifact archive",
            path=normalized_artifact_path,
        ) from exc

    return {
        "valid": True,
        "artifact_path": normalized_artifact_path,
        "artifact_uri": Path(normalized_artifact_path).resolve().as_uri(),
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "manifest_path": "manifest.json",
        "manifest": manifest_payload,
        "file_hash_verification": {
            "enabled": verify_file_hashes,
            "checked_files": verified_files if verify_file_hashes else 0,
            "checked_bytes": verified_bytes if verify_file_hashes else 0,
        },
    }


def _resolve_artifact_restore_path(root_dir: str, rel_path: str) -> str:
    """Resolve an artifact member path under root_dir and reject traversal."""
    if not rel_path or rel_path.startswith("/"):
        raise CLIContractError(
            f"Artifact manifest entry has invalid restore path: {rel_path!r}",
            error_code="artifact_manifest_invalid",
        )
    normalized_root = os.path.abspath(root_dir)
    candidate = os.path.abspath(os.path.join(normalized_root, rel_path.replace("/", os.sep)))
    try:
        within_root = os.path.commonpath([normalized_root, candidate]) == normalized_root
    except ValueError as exc:
        raise CLIContractError(
            f"Artifact manifest entry escapes restore root: {rel_path!r}",
            error_code="artifact_manifest_invalid",
        ) from exc
    if not within_root:
        raise CLIContractError(
            f"Artifact manifest entry escapes restore root: {rel_path!r}",
            error_code="artifact_manifest_invalid",
        )
    return candidate


def _restore_artifact_archive(
    artifact_path: str,
    *,
    destination_dir: str,
    overwrite: bool = False,
    verify_file_hashes: bool = True,
) -> dict[str, object]:
    """Restore a validated cache artifact into a destination directory."""
    validation_payload = _validate_artifact_archive(
        artifact_path,
        verify_file_hashes=verify_file_hashes,
    )
    manifest = validation_payload["manifest"]
    if not isinstance(manifest, dict):
        raise CLIContractError(
            "Artifact manifest payload is malformed",
            error_code="artifact_manifest_invalid",
        )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise CLIContractError(
            "Artifact manifest.files must be a list",
            error_code="artifact_manifest_invalid",
        )

    normalized_destination = os.path.abspath(os.path.expanduser(destination_dir))
    if os.path.exists(normalized_destination):
        if not os.path.isdir(normalized_destination):
            raise CLIContractError(
                f"Artifact restore destination is not a directory: {normalized_destination}",
                error_code="artifact_restore_destination_not_directory",
            )
        if not overwrite:
            raise CLIContractError(
                f"Artifact restore destination already exists: {normalized_destination}",
                error_code="artifact_restore_destination_exists",
            )

    destination_parent = os.path.dirname(normalized_destination) or "."
    try:
        os.makedirs(destination_parent, exist_ok=True)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="create artifact restore parent directory",
            path=destination_parent,
        ) from exc

    try:
        staging_dir = tempfile.mkdtemp(
            prefix="gloggur-artifact-restore-",
            dir=destination_parent,
        )
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="create artifact restore staging directory",
            path=destination_parent,
        ) from exc
    restored_files = 0
    restored_bytes = 0

    try:
        try:
            with tarfile.open(str(validation_payload["artifact_path"]), "r:gz") as archive:
                for entry in files:
                    if not isinstance(entry, dict):
                        raise CLIContractError(
                            "Artifact manifest file entry is not an object",
                            error_code="artifact_manifest_invalid",
                        )
                    rel_path = entry.get("path")
                    entry_bytes = entry.get("bytes")
                    if not isinstance(rel_path, str) or not rel_path:
                        raise CLIContractError(
                            "Artifact manifest file entry has invalid path",
                            error_code="artifact_manifest_invalid",
                        )
                    if not isinstance(entry_bytes, int) or entry_bytes < 0:
                        raise CLIContractError(
                            f"Artifact manifest entry for {rel_path} has invalid bytes",
                            error_code="artifact_manifest_invalid",
                        )
                    member_name = f"cache/{rel_path}"
                    try:
                        member = archive.getmember(member_name)
                    except KeyError as exc:
                        raise CLIContractError(
                            f"Artifact is missing expected file during restore: {member_name}",
                            error_code="artifact_manifest_file_mismatch",
                        ) from exc
                    file_obj = archive.extractfile(member)
                    if file_obj is None:
                        raise CLIContractError(
                            f"Artifact file could not be read during restore: {member_name}",
                            error_code="artifact_manifest_file_mismatch",
                        )
                    target_path = _resolve_artifact_restore_path(staging_dir, rel_path)
                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with open(target_path, "wb") as handle:
                            shutil.copyfileobj(file_obj, handle)
                    except OSError as exc:
                        raise wrap_io_error(
                            exc,
                            operation="restore artifact cache file",
                            path=target_path,
                        ) from exc
                    restored_files += 1
                    restored_bytes += entry_bytes
        except tarfile.TarError as exc:
            raise CLIContractError(
                f"Artifact archive became unreadable during restore: {type(exc).__name__}: {exc}",
                error_code="artifact_archive_invalid",
            ) from exc

        if os.path.exists(normalized_destination):
            try:
                shutil.rmtree(normalized_destination)
            except OSError as exc:
                raise wrap_io_error(
                    exc,
                    operation="remove existing artifact restore destination",
                    path=normalized_destination,
                ) from exc
        try:
            shutil.move(staging_dir, normalized_destination)
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="activate restored artifact cache",
                path=normalized_destination,
            ) from exc
        staging_dir = ""
    finally:
        if staging_dir and os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)

    return {
        "restored": True,
        "artifact_path": validation_payload["artifact_path"],
        "artifact_uri": validation_payload["artifact_uri"],
        "archive_sha256": validation_payload["archive_sha256"],
        "archive_bytes": validation_payload["archive_bytes"],
        "manifest_sha256": validation_payload["manifest_sha256"],
        "manifest_path": validation_payload["manifest_path"],
        "manifest": manifest,
        "destination_cache_dir": normalized_destination,
        "restored_files": restored_files,
        "restored_bytes": restored_bytes,
        "overwrite_applied": overwrite,
        "file_hash_verification": validation_payload["file_hash_verification"],
    }


def _parse_extract_byte_value(raw_value: str, *, field_name: str) -> int:
    """Parse one extract byte boundary with stable contract errors."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise CLIContractError(
            f"{field_name} must be an integer",
            error_code="extract_byte_range_invalid",
        ) from exc
    if value < 0:
        raise CLIContractError(
            f"{field_name} must be >= 0",
            error_code="extract_byte_range_invalid",
        )
    return value


def _extract_payload(path: str, start_byte_raw: str, end_byte_raw: str) -> dict[str, object]:
    """Build the extract success payload or raise a stable contract error."""
    repo_root = discover_repo_root()
    start_byte = _parse_extract_byte_value(start_byte_raw, field_name="start_byte")
    end_byte = _parse_extract_byte_value(end_byte_raw, field_name="end_byte")
    if end_byte < start_byte:
        raise CLIContractError(
            "end_byte must be >= start_byte",
            error_code="extract_byte_range_invalid",
        )
    try:
        absolute_path = resolve_repo_relative_path(repo_root, path)
    except RepoPathResolutionError as exc:
        raise CLIContractError(
            f"Invalid extract path: {path!r}",
            error_code="extract_path_invalid",
        ) from exc
    if not absolute_path.exists() or not absolute_path.is_file():
        raise CLIContractError(
            f"Extract file does not exist: {path!r}",
            error_code="extract_file_missing",
        )
    try:
        raw_bytes = absolute_path.read_bytes()
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read extract source file",
            path=str(absolute_path),
        ) from exc
    span_index = LineByteSpanIndex.from_bytes(raw_bytes)
    if end_byte > span_index.total_bytes:
        raise CLIContractError(
            f"Extract byte range exceeds file bounds: {end_byte} > {span_index.total_bytes}",
            error_code="extract_range_out_of_bounds",
        )
    relative_path = os.path.relpath(str(absolute_path), str(repo_root)).replace(os.sep, "/")
    return {
        "path": relative_path,
        "start_byte": start_byte,
        "end_byte": end_byte,
        "text": span_index.extract_text(start_byte, end_byte),
    }


@cli.command()
@click.argument("path")
@click.argument("start_byte")
@click.argument("end_byte")
@click.option("--json", "as_json", is_flag=True, default=False)
def extract(path: str, start_byte: str, end_byte: str, as_json: bool) -> None:
    """Extract an exact raw-byte text slice from a repo-relative file path."""
    try:
        payload = _extract_payload(path, start_byte, end_byte)
    except CLIContractError as exc:
        if as_json:
            _emit_json_error(exc.to_payload())
        else:
            click.echo(f"ERROR: {exc.error_code}")
        raise click.exceptions.Exit(exc.exit_code) from exc
    except StorageIOError as exc:
        if as_json:
            _emit_json_error(exc.to_payload())
        else:
            click.echo(format_io_error_message(exc), err=True)
        raise click.exceptions.Exit(1) from exc

    if as_json:
        _emit(payload, as_json=True)
        return
    click.echo(str(payload["text"]), nl=False)


def _build_failure_contract(
    failed_reasons: dict[str, int],
    *,
    remediation_by_reason: dict[str, list[str]],
    default_remediation: str,
) -> dict[str, object]:
    """Build deterministic machine-readable failure codes and remediation payloads."""
    normalized_reasons = _normalize_reason_counts(failed_reasons)
    if not normalized_reasons:
        return {}
    return {
        "failure_codes": sorted(normalized_reasons),
        "failure_guidance": {
            reason: remediation_by_reason.get(reason, [default_remediation])
            for reason in sorted(normalized_reasons)
        },
    }


def _attach_primary_error_from_failure_contract(
    payload: dict[str, object],
    *,
    error_type: str,
    detail: str,
    probable_cause: str,
    default_remediation: str,
) -> None:
    """Attach a top-level error block from the existing failure contract fields."""
    raw_codes = payload.get("failure_codes")
    if not isinstance(raw_codes, list) or not raw_codes:
        return

    failure_codes = [str(code) for code in raw_codes if str(code)]
    if not failure_codes:
        return
    primary_code = failure_codes[0]

    remediation = [default_remediation]
    raw_guidance = payload.get("failure_guidance")
    if isinstance(raw_guidance, dict):
        primary_guidance = raw_guidance.get(primary_code)
        if isinstance(primary_guidance, list):
            normalized_guidance = [str(item) for item in primary_guidance if str(item)]
            if normalized_guidance:
                remediation = normalized_guidance

    payload["error"] = {
        "type": error_type,
        "code": primary_code,
        "detail": detail,
        "probable_cause": probable_cause,
        "remediation": remediation,
    }


def _build_index_failure_contract(failed_reasons: dict[str, int]) -> dict[str, object]:
    """Build deterministic machine-readable failure codes and remediation for index payloads."""
    return _build_failure_contract(
        failed_reasons,
        remediation_by_reason=FAILURE_REMEDIATION,
        default_remediation=DEFAULT_INDEX_FAILURE_REMEDIATION,
    )


def _build_inspect_failure_contract(failed_reasons: dict[str, int]) -> dict[str, object]:
    """Build deterministic machine-readable failure codes and remediation for inspect payloads."""
    return _build_failure_contract(
        failed_reasons,
        remediation_by_reason=INSPECT_FAILURE_REMEDIATION,
        default_remediation=DEFAULT_INSPECT_FAILURE_REMEDIATION,
    )


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
@click.option(
    "--debug-timings",
    is_flag=True,
    default=False,
    help="Include per-file timing breakdowns in index --json output.",
)
@_with_io_failure_handling
def index(
    path: str,
    config_path: str | None,
    as_json: bool,
    embedding_provider: str | None,
    allow_partial: bool,
    debug_timings: bool,
) -> None:
    """Load config/runtime, index path, and emit summary counts."""
    command_started = time.perf_counter()
    resolved_config_path = _normalize_config_path(config_path)
    config = _normalize_watch_paths(_load_config(resolved_config_path), resolved_config_path)
    if embedding_provider:
        config.embedding_provider = embedding_provider

    stage_recorder = IndexStageRecorder()
    bootstrap_started = time.perf_counter()
    embedding = _create_embedding_provider_for_command(
        config,
        require_provider=True,
    )
    bootstrap_counts = _warm_embedding_provider(embedding)
    stage_recorder.record(
        "bootstrap_model",
        status="completed",
        duration_ms=int((time.perf_counter() - bootstrap_started) * 1000),
        counts=bootstrap_counts,
    )

    parser_registry = ParserRegistry(
        extension_map=config.parser_extension_map,
        adapter_overrides=config.adapters if isinstance(config.adapters, dict) else None,
    )
    build_id = _new_build_id()
    build_started_at = datetime.now(timezone.utc).isoformat()
    current_stage = "bootstrap_model"
    active_cache: CacheManager | None = None
    publish_succeeded = False

    def _merge_failure_reasons(
        target: dict[str, int],
        source: object,
    ) -> dict[str, int]:
        """Merge positive failure counts from one payload into the aggregate mapping."""
        if not isinstance(source, dict):
            return target
        for raw_reason, raw_count in source.items():
            reason = str(raw_reason)
            count = int(raw_count or 0)
            if count <= 0:
                continue
            target[reason] = target.get(reason, 0) + count
        return target

    def _merge_failure_samples(
        target: list[str],
        source: object,
        *,
        limit: int = 5,
    ) -> list[str]:
        """Append failure samples up to the stable output cap."""
        if not isinstance(source, list):
            return target
        for raw_sample in source:
            if len(target) >= limit:
                break
            target.append(str(raw_sample))
        return target

    with cache_write_lock(config.cache_dir):
        click.echo("Indexing...", err=True)
        active_cache = _create_cache_manager(config.cache_dir)
        get_prior_build_state = getattr(active_cache, "get_build_state", None)
        raw_prior_build_state = get_prior_build_state() if callable(get_prior_build_state) else None
        prior_build_state, stale_prior_build_state = _classify_build_state_for_health(
            raw_prior_build_state
        )
        if stale_prior_build_state and prior_build_state is not None:
            clear_prior_build_state = getattr(active_cache, "clear_build_state", None)
            if callable(clear_prior_build_state):
                clear_prior_build_state()
        active_cache.cleanup_staged_builds()
        stage_dir = active_cache.prepare_staged_build(build_id)
        stage_config = replace(config, cache_dir=stage_dir)
        stage_config, cache, vector_store = _initialize_runtime(
            stage_config,
            rebuild_on_profile_change=True,
            write_locked=True,
        )
        indexer = Indexer(
            config=stage_config,
            cache=cache,
            parser_registry=parser_registry,
            embedding_provider=embedding,
            vector_store=vector_store,
        )

        def _update_build_stage(stage_name: str) -> None:
            """Advance the active build-state sidecar to the current lifecycle stage."""
            nonlocal current_stage
            current_stage = stage_name
            assert active_cache is not None
            _write_cache_build_state(
                active_cache,
                state="building",
                build_id=build_id,
                started_at=build_started_at,
                stage=stage_name,
            )

        _update_build_stage("scan_source")
        indexer._stage_callback = _update_build_stage

        if not as_json:

            def _scan(done: int, total: int, status: str) -> None:
                """Render one in-place scan progress update for interactive index runs."""
                _ = status
                click.echo(f"\rScanning: {done}/{total} files    ", nl=False, err=True)

            indexer._scan_callback = _scan

            def _progress(done: int, total: int) -> None:
                """Render one in-place embedding progress update for interactive runs."""
                click.echo(
                    f"\rEmbedding: {done}/{total} symbols    ",
                    nl=False,
                    err=True,
                )

            indexer._progress_callback = _progress

        def _handle_interrupt(_signal_name: str) -> None:
            """Mark the build interrupted and terminate child workers on SIGINT/SIGTERM."""
            assert active_cache is not None
            _write_cache_build_state(
                active_cache,
                state="interrupted",
                build_id=build_id,
                started_at=build_started_at,
                stage=current_stage,
                cleanup_pending=True,
            )
            _terminate_index_children()

        try:
            with _index_signal_guard(_handle_interrupt):
                if os.path.isdir(path):
                    result = indexer.index_repository(path)
                    _record_repository_index_stages(stage_recorder, result)

                    _update_build_stage("update_symbol_index")
                    symbol_payload = _run_symbol_index(
                        index_target=path,
                        config=config,
                        parser_registry=parser_registry,
                        prefetched_files=result.parsed_files,
                        file_paths=result.source_files,
                    )
                    _record_symbol_index_stage(stage_recorder, symbol_payload)

                    overall_failed_reasons = dict(result.failed_reasons)
                    _merge_failure_reasons(
                        overall_failed_reasons,
                        symbol_payload.get("failed_reasons"),
                    )
                    overall_failed_samples = list(result.failed_samples)
                    _merge_failure_samples(
                        overall_failed_samples,
                        symbol_payload.get("failed_samples"),
                    )
                    overall_failed = result.failed + int(symbol_payload.get("failed", 0) or 0)

                    commit_duration_ms = 0
                    total_symbols = 0
                    indexed_files = 0
                    if overall_failed == 0:
                        _update_build_stage("commit_metadata")
                        commit_started = time.perf_counter()
                        active_cache.publish_staged_build(build_id)
                        publish_succeeded = True
                        total_symbols = active_cache.count_symbols()
                        indexed_files = active_cache.count_files()
                        _persist_last_success_resume_state(config, active_cache)
                        active_cache.clear_build_state()
                        commit_duration_ms = int((time.perf_counter() - commit_started) * 1000)
                        _record_commit_metadata_stage(
                            stage_recorder,
                            status="completed",
                            duration_ms=commit_duration_ms,
                            total_symbols=total_symbols,
                            indexed_files=indexed_files,
                        )
                    else:
                        _write_cache_build_state(
                            active_cache,
                            state="interrupted",
                            build_id=build_id,
                            started_at=build_started_at,
                            stage=current_stage,
                            cleanup_pending=True,
                        )
                        _record_commit_metadata_stage(
                            stage_recorder,
                            status="not_run",
                            duration_ms=0,
                        )

                    command_duration_ms = int((time.perf_counter() - command_started) * 1000)
                    if not as_json:
                        click.echo("", err=True)
                    payload = result.as_payload()
                    payload["failed"] = overall_failed
                    payload["failed_reasons"] = overall_failed_reasons
                    payload["failed_samples"] = overall_failed_samples
                    payload["symbol_index"] = symbol_payload
                    payload["timings_ms"] = {
                        "total": command_duration_ms,
                        "legacy_index": result.duration_ms,
                        "symbol_index": int(symbol_payload.get("duration_ms", 0) or 0),
                        "cleanup": int(result.phase_timings_ms.get("cleanup", 0)),
                        "consistency_checks": int(
                            result.phase_timings_ms.get("consistency_checks", 0)
                        ),
                    }
                    payload["duration_ms"] = command_duration_ms
                    payload["stages"] = stage_recorder.as_payload()
                    if as_json and debug_timings:
                        payload["slow_files"] = _build_slow_files_payload(result.file_timings)
                    payload.update(_build_index_failure_contract(overall_failed_reasons))
                    if as_json and overall_failed > 0 and not allow_partial:
                        _attach_primary_error_from_failure_contract(
                            payload,
                            error_type="index_failure",
                            detail="Indexing did not finish cleanly.",
                            probable_cause=(
                                "One or more repository or symbol-index stages failed; "
                                "inspect failed_reasons and failed_samples for the concrete causes."
                            ),
                            default_remediation=DEFAULT_INDEX_FAILURE_REMEDIATION,
                        )
                    _emit(payload, as_json)
                    if overall_failed > 0 and not allow_partial:
                        raise click.exceptions.Exit(1)
                    return

                files_considered = 1
                if any(path.endswith(ext) for ext in config.supported_extensions):
                    segments = set(os.path.normpath(os.path.abspath(path)).split(os.sep))
                    if any(excluded in segments for excluded in config.excluded_dirs):
                        files_considered = 0
                else:
                    files_considered = 0

                stage_recorder.record(
                    "scan_source",
                    status="completed",
                    duration_ms=0,
                    counts={"source_files": files_considered},
                )

                if files_considered:
                    cache.delete_index_metadata()

                _update_build_stage("extract_symbols")
                single_index_started = time.perf_counter()
                execution = indexer.index_file_with_details(path) if files_considered else None
                legacy_index_ms = int((time.perf_counter() - single_index_started) * 1000)

                cleanup_started = time.perf_counter()
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
                cleanup_ms = int((time.perf_counter() - cleanup_started) * 1000)

                _update_build_stage("validate_integrity")
                consistency_started = time.perf_counter()
                consistency = (
                    indexer.validate_vector_metadata_consistency()
                    if files_considered
                    else {"failed": 0, "failed_reasons": {}, "failed_samples": []}
                )
                consistency_ms = int((time.perf_counter() - consistency_started) * 1000)

                failed_reasons: dict[str, int] = {}
                failed_samples: list[str] = []
                indexed = 0
                unchanged = 0
                failed = 0
                indexed_symbols = 0
                execution_timing = (
                    execution.timing if execution and execution.timing is not None else None
                )
                outcome = execution.outcome if execution else None
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
                _merge_failure_reasons(failed_reasons, stale_cleanup.get("failed_reasons"))
                _merge_failure_samples(failed_samples, stale_cleanup.get("failed_samples"))
                failed += int(consistency.get("failed", 0))
                _merge_failure_reasons(failed_reasons, consistency.get("failed_reasons"))
                _merge_failure_samples(failed_samples, consistency.get("failed_samples"))

                vector_integrity = consistency.get("integrity")
                if not isinstance(vector_integrity, dict):
                    vector_integrity = _integrity_marker(
                        name="vector_cache",
                        status="missing",
                        reason_codes=["vector_integrity_missing"],
                        detail="vector/cache integrity marker missing",
                    )
                chunk_integrity = _integrity_marker(
                    name="chunk_span",
                    status=(
                        "failed"
                        if failed_reasons.get("chunk_span_integrity_error", 0) > 0
                        else "passed"
                    ),
                    reason_codes=(
                        ["chunk_span_integrity_error"]
                        if failed_reasons.get("chunk_span_integrity_error", 0) > 0
                        else []
                    ),
                    detail=(
                        failed_samples[0]
                        if (
                            failed_reasons.get("chunk_span_integrity_error", 0) > 0
                            and failed_samples
                        )
                        else "chunk/span integrity checks passed"
                    ),
                )
                cache.set_search_integrity(
                    {
                        "vector_cache": vector_integrity,
                        "chunk_span": chunk_integrity,
                    }
                )

                extract_duration_ms = 0
                embed_duration_ms = 0
                persist_duration_ms = cleanup_ms
                if execution_timing is not None:
                    extract_duration_ms = execution_timing.parse_ms + execution_timing.edge_ms
                    embed_duration_ms = execution_timing.embed_ms
                    persist_duration_ms += execution_timing.persist_ms

                stage_recorder.record(
                    "extract_symbols",
                    status=(
                        _stage_status_from_reasons(failed_reasons, "extract_symbols")
                        if files_considered
                        else "not_run"
                    ),
                    duration_ms=extract_duration_ms,
                    counts={"files_considered": files_considered},
                )
                stage_recorder.record(
                    "embed_chunks",
                    status=(
                        _stage_status_from_reasons(failed_reasons, "embed_chunks")
                        if files_considered
                        else "not_run"
                    ),
                    duration_ms=embed_duration_ms,
                    counts={"indexed_symbols": indexed_symbols},
                )
                stage_recorder.record(
                    "persist_cache",
                    status=(
                        _stage_status_from_reasons(failed_reasons, "persist_cache")
                        if files_considered
                        else "not_run"
                    ),
                    duration_ms=persist_duration_ms if files_considered else 0,
                    counts={
                        "files_changed": indexed,
                        "files_removed": files_removed,
                        "symbols_removed": (outcome.symbols_removed if outcome else 0)
                        + symbols_removed,
                    },
                )
                stage_recorder.record(
                    "validate_integrity",
                    status=(
                        _stage_status_from_reasons(failed_reasons, "validate_integrity")
                        if files_considered
                        else "not_run"
                    ),
                    duration_ms=consistency_ms if files_considered else 0,
                    counts={"failed": int(consistency.get("failed", 0) or 0)},
                )

                if outcome and outcome.status != "failed" and failed == 0:
                    metadata = IndexMetadata(
                        version=config.index_version,
                        total_symbols=cache.count_symbols(),
                        indexed_files=cache.count_files(),
                    )
                    cache.set_index_metadata(metadata)
                    cache.set_index_profile(config.embedding_profile())

                _update_build_stage("update_symbol_index")
                symbol_payload = _run_symbol_index(
                    index_target=path,
                    config=config,
                    parser_registry=parser_registry,
                    prefetched_files=(
                        [execution.prepared.snapshot] if execution and execution.prepared else None
                    ),
                    file_paths=[path] if files_considered else None,
                )
                _record_symbol_index_stage(stage_recorder, symbol_payload)

                overall_failed_reasons = dict(failed_reasons)
                _merge_failure_reasons(
                    overall_failed_reasons,
                    symbol_payload.get("failed_reasons"),
                )
                overall_failed_samples = list(failed_samples)
                _merge_failure_samples(
                    overall_failed_samples,
                    symbol_payload.get("failed_samples"),
                )
                overall_failed = failed + int(symbol_payload.get("failed", 0) or 0)

                if overall_failed == 0:
                    _update_build_stage("commit_metadata")
                    commit_started = time.perf_counter()
                    active_cache.publish_staged_build(build_id)
                    publish_succeeded = True
                    _persist_last_success_resume_state(config, active_cache)
                    active_cache.clear_build_state()
                    commit_duration_ms = int((time.perf_counter() - commit_started) * 1000)
                    _record_commit_metadata_stage(
                        stage_recorder,
                        status="completed",
                        duration_ms=commit_duration_ms,
                        total_symbols=active_cache.count_symbols(),
                        indexed_files=active_cache.count_files(),
                    )
                else:
                    _write_cache_build_state(
                        active_cache,
                        state="interrupted",
                        build_id=build_id,
                        started_at=build_started_at,
                        stage=current_stage,
                        cleanup_pending=True,
                    )
                    _record_commit_metadata_stage(
                        stage_recorder,
                        status="not_run",
                        duration_ms=0,
                    )

                result = {
                    "files_considered": files_considered,
                    "indexed": indexed,
                    "unchanged": unchanged,
                    "failed": overall_failed,
                    "failed_reasons": overall_failed_reasons,
                    "failed_samples": overall_failed_samples,
                    "files_changed": indexed,
                    "files_removed": files_removed,
                    "symbols_added": outcome.symbols_added if outcome else 0,
                    "symbols_updated": outcome.symbols_updated if outcome else 0,
                    "symbols_removed": (outcome.symbols_removed if outcome else 0)
                    + symbols_removed,
                    "indexed_files": indexed,
                    "skipped_files": unchanged,
                    "indexed_symbols": indexed_symbols,
                    "duration_ms": 0,
                    "symbol_index": symbol_payload,
                }
                command_duration_ms = int((time.perf_counter() - command_started) * 1000)
                result["timings_ms"] = {
                    "total": command_duration_ms,
                    "legacy_index": legacy_index_ms,
                    "symbol_index": int(symbol_payload.get("duration_ms", 0) or 0),
                    "cleanup": cleanup_ms,
                    "consistency_checks": consistency_ms,
                }
                result["duration_ms"] = command_duration_ms
                result["stages"] = stage_recorder.as_payload()
                if as_json and debug_timings and execution_timing is not None:
                    result["slow_files"] = [execution_timing.as_payload()]
                result.update(_build_index_failure_contract(overall_failed_reasons))
                if as_json and overall_failed > 0 and not allow_partial:
                    _attach_primary_error_from_failure_contract(
                        result,
                        error_type="index_failure",
                        detail="Indexing did not finish cleanly.",
                        probable_cause=(
                            "One or more file, cleanup, consistency, or symbol-index stages "
                            "failed; inspect failed_reasons and failed_samples "
                            "for the concrete causes."
                        ),
                        default_remediation=DEFAULT_INDEX_FAILURE_REMEDIATION,
                    )
                _emit(result, as_json)
                if overall_failed > 0 and not allow_partial:
                    raise click.exceptions.Exit(1)
        except BaseException:
            if active_cache is not None:
                try:
                    if publish_succeeded:
                        active_cache.clear_build_state()
                    else:
                        _write_cache_build_state(
                            active_cache,
                            state="interrupted",
                            build_id=build_id,
                            started_at=build_started_at,
                            stage=current_stage,
                            cleanup_pending=True,
                        )
                except Exception:
                    pass
            raise


def _validate_search_confidence_options(
    *,
    top_k: int,
    confidence_threshold: float,
    max_requery_attempts: int,
) -> None:
    """Validate search confidence/retry CLI options with fail-closed contracts."""
    if top_k < 1:
        raise CLIContractError(
            "--top-k must be >= 1",
            error_code="search_top_k_invalid",
        )
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise CLIContractError(
            "--confidence-threshold must be between 0.0 and 1.0",
            error_code="search_confidence_threshold_invalid",
        )
    if max_requery_attempts < 0:
        raise CLIContractError(
            "--max-requery-attempts must be >= 0",
            error_code="search_max_requery_attempts_invalid",
        )


def _validate_search_evidence_options(
    *,
    evidence_min_confidence: float,
    evidence_min_items: int,
) -> None:
    """Validate evidence trace/grounding options with fail-closed contracts."""
    if evidence_min_confidence < 0.0 or evidence_min_confidence > 1.0:
        raise CLIContractError(
            "--evidence-min-confidence must be between 0.0 and 1.0",
            error_code="search_evidence_min_confidence_invalid",
        )
    if evidence_min_items < 1:
        raise CLIContractError(
            "--evidence-min-items must be >= 1",
            error_code="search_evidence_min_items_invalid",
        )


def _validate_search_router_options(
    *,
    max_files: int | None,
    max_snippets: int | None,
    time_budget_ms: int | None,
) -> None:
    """Validate ContextPack router constraint options with fail-closed contracts."""
    if max_files is not None and max_files < 1:
        raise CLIContractError(
            "--max-files must be >= 1",
            error_code="search_max_files_invalid",
        )
    if max_snippets is not None and max_snippets < 1:
        raise CLIContractError(
            "--max-snippets must be >= 1",
            error_code="search_max_snippets_invalid",
        )
    if time_budget_ms is not None and time_budget_ms < 1:
        raise CLIContractError(
            "--time-budget-ms must be >= 1",
            error_code="search_time_budget_invalid",
        )


def _resolve_router_repo_root(*, metadata_store: MetadataStore, fallback: Path) -> Path:
    """Resolve router repo root from indexed symbol paths with cwd fallback."""
    try:
        file_paths = metadata_store.sample_symbol_file_paths(limit=64)
    except Exception:
        return fallback
    if not file_paths:
        return fallback

    normalized_paths: list[str] = []
    for raw in file_paths:
        candidate = raw.strip()
        if not candidate:
            continue
        if os.path.isabs(candidate):
            absolute = candidate
        else:
            absolute = os.path.abspath(str(fallback / candidate))
        normalized_paths.append(os.path.normpath(absolute))
    if not normalized_paths:
        return fallback

    try:
        common = Path(os.path.commonpath(normalized_paths))
    except ValueError:
        return fallback

    # If every sampled path is the same file, step up to the parent directory.
    if all(Path(item) == common for item in normalized_paths):
        common = common.parent
    return common if str(common).strip() else fallback


def _resolve_search_path_filter_for_routing(*, repo_root: Path, raw_path: str | None) -> str | None:
    """Resolve a search path filter against the indexed repo for backend matching."""
    if not isinstance(raw_path, str):
        return None
    candidate = raw_path.strip()
    if not candidate:
        return None
    if is_path_absolute(candidate):
        return os.path.abspath(candidate)
    normalized = os.path.normpath(candidate.replace("\\", os.sep))
    if normalized in {"", ".", ".."} or normalized.startswith(f"..{os.sep}"):
        return candidate
    repo_root_str = os.path.abspath(str(repo_root))
    resolved = os.path.abspath(os.path.join(repo_root_str, normalized))
    try:
        within_root = os.path.commonpath([repo_root_str, resolved]) == repo_root_str
    except ValueError:
        within_root = False
    if within_root:
        return resolved
    return candidate


def _resolve_symbol_index_root(index_target: str) -> Path:
    """Resolve repo root used for .gloggur/index/symbols.db storage."""
    target = Path(os.path.abspath(index_target))
    current = target if target.is_dir() else target.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / ".gloggur").exists():
            return candidate
    if target.is_dir():
        return target
    return target.parent


def _run_symbol_index(
    *,
    index_target: str,
    config: GloggurConfig,
    parser_registry: ParserRegistry,
    prefetched_files: Sequence[ParsedFileSnapshot] | None = None,
    file_paths: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build/update local symbol occurrence index and return additive payload fields."""
    repo_root = _resolve_symbol_index_root(index_target)
    indexer = SymbolIndexer(
        repo_root=repo_root,
        config=config,
        parser_registry=parser_registry,
    )
    result = indexer.index_path_with_prefetched(
        index_target,
        prefetched_files=list(prefetched_files or []),
        file_paths=list(file_paths) if file_paths is not None else None,
    )
    return result.as_payload()


def _build_slow_files_payload(
    file_timings: Sequence[FileTimingTrace],
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Return top slow files for debug timing payloads."""
    ranked = sorted(
        (timing for timing in file_timings if timing.total_ms > 0),
        key=lambda timing: (timing.total_ms, timing.path),
        reverse=True,
    )
    return [timing.as_payload() for timing in ranked[:limit]]


def _resolve_search_ranking_metadata(
    *,
    query: str,
    ranking_mode: str,
    file_path: str | None,
    context_radius: int,
) -> dict[str, object]:
    """Resolve stable ranking metadata for search payloads."""
    ranking_filters: dict[str, str] = {"ranking_mode": ranking_mode}
    if file_path:
        ranking_filters["file"] = file_path
    metadata = hybrid_search_module.HybridSearch.build_ranking_metadata(query, ranking_filters)
    metadata["context_radius"] = context_radius
    return metadata


def _resolve_allow_tool_version_drift(
    *,
    cli_flag_enabled: bool,
) -> bool:
    """Resolve tool-version drift override from CLI flag + env with strict validation."""
    env_values = _merged_env_values()
    raw_value = env_values.get("GLOGGUR_ALLOW_TOOL_VERSION_DRIFT")
    if raw_value is None or raw_value.strip() == "":
        return cli_flag_enabled
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        env_enabled = True
    elif normalized in {"0", "false", "no", "off"}:
        env_enabled = False
    else:
        raise CLIContractError(
            "GLOGGUR_ALLOW_TOOL_VERSION_DRIFT must be one of: 1, true, yes, on, 0, false, no, off.",
            error_code="allow_tool_version_drift_env_invalid",
        )
    return cli_flag_enabled or env_enabled


def _merged_env_values() -> dict[str, str]:
    """Return merged dotenv + process environment values with process env precedence."""
    env_values = GloggurConfig._load_dotenv()
    env_values.update(os.environ)
    return env_values


def _validate_legacy_local_fallback_env() -> None:
    """Fail closed when legacy local fallback env is configured."""
    raw_value = _merged_env_values().get("GLOGGUR_LOCAL_FALLBACK")
    if raw_value is None or raw_value.strip() == "":
        return
    raise CLIContractError(
        "GLOGGUR_LOCAL_FALLBACK is no longer supported. "
        "Use GLOGGUR_EMBEDDING_PROVIDER=test for deterministic test-only embeddings.",
        error_code="local_fallback_env_unsupported",
    )


def _extract_similarity_scores(results: object) -> list[float]:
    """Extract bounded similarity scores or fail loudly on malformed payloads."""
    if not isinstance(results, list):
        raise ValueError("search payload 'results' must be a list")
    scores: list[float] = []
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"search result at index {idx} is not an object")
        raw_score = item.get("similarity_score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"search result at index {idx} has non-numeric similarity_score"
            ) from exc
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        scores.append(score)
    return scores


def _compute_retrieval_confidence(results: object) -> float:
    """Compute deterministic confidence from top result quality and top-3 average."""
    scores = _extract_similarity_scores(results)
    if not scores:
        return 0.0
    ranked = sorted(scores, reverse=True)
    top_score = ranked[0]
    top_three = ranked[:3]
    top_three_average = sum(top_three) / len(top_three)
    confidence = (0.7 * top_score) + (0.3 * top_three_average)
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _next_retry_top_k(current_top_k: int, *, max_top_k: int = MAX_REQUERY_TOP_K) -> int:
    """Return deterministic expanded top-k for bounded retry strategy."""
    if current_top_k < 1:
        raise ValueError("current_top_k must be >= 1")
    if max_top_k < 1:
        raise ValueError("max_top_k must be >= 1")
    expanded = max(current_top_k + 1, current_top_k * 2)
    if expanded > max_top_k:
        expanded = max_top_k
    return expanded


def _validate_search_payload(payload: object) -> dict[str, object]:
    """Validate search payload shape and fail loudly on schema drift."""
    if not isinstance(payload, dict):
        raise ValueError("search payload must be an object")
    if "results" not in payload:
        raise ValueError("search payload is missing 'results'")
    if "metadata" not in payload:
        raise ValueError("search payload is missing 'metadata'")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("search payload 'metadata' must be an object")
    return payload


def _search_with_bounded_retry(
    *,
    searcher: HybridSearch,
    query: str,
    filters: dict[str, str],
    initial_top_k: int,
    context_radius: int,
    confidence_threshold: float,
    max_requery_attempts: int,
    disable_bounded_requery: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run search with optional bounded retry and deterministic confidence telemetry."""
    result = _validate_search_payload(
        searcher.search(
            query,
            filters=filters,
            top_k=initial_top_k,
            context_radius=context_radius,
        )
    )
    initial_confidence = _compute_retrieval_confidence(result.get("results"))
    final_confidence = initial_confidence
    final_top_k = initial_top_k
    retry_performed = False
    retry_attempts = 0
    retry_strategy: str | None = None

    retry_enabled = (not disable_bounded_requery) and max_requery_attempts > 0
    while (
        retry_enabled
        and retry_attempts < max_requery_attempts
        and final_confidence < confidence_threshold
    ):
        next_top_k = _next_retry_top_k(final_top_k)
        if next_top_k == final_top_k:
            break
        retry_performed = True
        retry_attempts += 1
        retry_strategy = REQUERY_STRATEGY_TOP_K_EXPANSION
        final_top_k = next_top_k
        result = _validate_search_payload(
            searcher.search(
                query,
                filters=filters,
                top_k=final_top_k,
                context_radius=context_radius,
            )
        )
        final_confidence = _compute_retrieval_confidence(result.get("results"))

    confidence_payload: dict[str, object] = {
        "confidence_threshold": confidence_threshold,
        "initial_confidence": initial_confidence,
        "final_confidence": final_confidence,
        "retry_performed": retry_performed,
        "retry_attempts": retry_attempts,
        "max_requery_attempts": max_requery_attempts,
        "retry_strategy": retry_strategy,
        "retry_enabled": retry_enabled,
        "initial_top_k": initial_top_k,
        "final_top_k": final_top_k,
        "low_confidence": final_confidence < confidence_threshold,
    }
    return result, confidence_payload


@cli.command()
@click.argument("query", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option(
    "--mode",
    type=click.Choice(["auto", "exact", "semantic", "hybrid"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Retrieval mode for ContextPack routing.",
)
@click.option("--language", type=str, default=None)
@click.option("--path-prefix", type=str, default=None)
@click.option("--max-files", type=int, default=None)
@click.option("--max-snippets", type=int, default=None)
@click.option("--time-budget-ms", type=int, default=None)
@click.option(
    "--debug-router",
    is_flag=True,
    default=False,
    help="Include debug payload with backend timings/scores/thresholds.",
)
@click.option("--kind", type=str, default=None)
@click.option("--file", "file_path", type=str, default=None)
@click.option(
    "--search-mode",
    type=click.Choice(
        ["semantic", "by_fqname", "by_fqname_regex", "by_path"],
        case_sensitive=False,
    ),
    default="semantic",
    show_default=True,
    help="Search mode: semantic chunk retrieval or exact/regex/path metadata lookup.",
)
@click.option("--top-k", type=int, default=10)
@click.option(
    "--context-radius",
    type=click.IntRange(1, 200),
    default=DEFAULT_SEARCH_CONTEXT_RADIUS,
    show_default=True,
    help="Number of context lines to include on each side of a symbol match.",
)
@click.option(
    "--ranking-mode",
    type=click.Choice(["balanced", "source-first"], case_sensitive=False),
    default="balanced",
    show_default=True,
    help="Ranking profile for source-vs-test preference in search results.",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD,
    show_default=True,
    help="Low-confidence threshold (0.0-1.0) that triggers bounded retry.",
)
@click.option(
    "--max-requery-attempts",
    type=int,
    default=DEFAULT_MAX_REQUERY_ATTEMPTS,
    show_default=True,
    help="Maximum bounded retry attempts when confidence is below threshold.",
)
@click.option(
    "--disable-bounded-requery",
    is_flag=True,
    default=False,
    help="Disable bounded retry logic and emit confidence from initial retrieval only.",
)
@click.option(
    "--with-evidence-trace",
    is_flag=True,
    default=False,
    help="Include evidence trace payload tied to returned symbols in JSON output.",
)
@click.option(
    "--validate-grounding",
    is_flag=True,
    default=False,
    help="Run default grounding validator over evidence trace and emit pass/fail metadata.",
)
@click.option(
    "--evidence-min-confidence",
    type=float,
    default=DEFAULT_EVIDENCE_MIN_CONFIDENCE,
    show_default=True,
    help="Minimum per-evidence confidence contribution for grounding pass criteria.",
)
@click.option(
    "--evidence-min-items",
    type=int,
    default=DEFAULT_EVIDENCE_MIN_ITEMS,
    show_default=True,
    help="Minimum number of evidence items required at/above confidence threshold.",
)
@click.option(
    "--fail-on-ungrounded",
    is_flag=True,
    default=False,
    help="Exit non-zero when grounding validation fails.",
)
@click.option("--stream", is_flag=True, default=False)
@click.option(
    "--allow-tool-version-drift",
    is_flag=True,
    default=False,
    help="Allow search when only tool-version drift is detected in resume metadata.",
)
@_with_io_failure_handling
def search(
    query: str,
    config_path: str | None,
    as_json: bool,
    mode: str,
    language: str | None,
    path_prefix: str | None,
    max_files: int | None,
    max_snippets: int | None,
    time_budget_ms: int | None,
    debug_router: bool,
    kind: str | None,
    file_path: str | None,
    search_mode: str,
    top_k: int,
    context_radius: int,
    ranking_mode: str,
    confidence_threshold: float,
    max_requery_attempts: int,
    disable_bounded_requery: bool,
    with_evidence_trace: bool,
    validate_grounding: bool,
    evidence_min_confidence: float,
    evidence_min_items: int,
    fail_on_ungrounded: bool,
    stream: bool,
    allow_tool_version_drift: bool,
) -> None:
    """Search repository context and return ContextPack v2 payload."""
    allow_tool_version_drift = _resolve_allow_tool_version_drift(
        cli_flag_enabled=allow_tool_version_drift
    )
    _validate_search_confidence_options(
        top_k=top_k,
        confidence_threshold=confidence_threshold,
        max_requery_attempts=max_requery_attempts,
    )
    _validate_search_evidence_options(
        evidence_min_confidence=evidence_min_confidence,
        evidence_min_items=evidence_min_items,
    )
    _validate_search_router_options(
        max_files=max_files,
        max_snippets=max_snippets,
        time_budget_ms=time_budget_ms,
    )
    if with_evidence_trace or validate_grounding or fail_on_ungrounded:
        raise CLIContractError(
            "Legacy evidence-trace/grounding options were removed from search JSON v2.",
            error_code="search_contract_v1_removed",
        )

    config, cache, vector_store = _create_runtime(config_path=config_path)
    health = _build_search_health_snapshot(
        config,
        cache,
        entrypoint="search_cli_v2",
        contract_version="contextpack_v2",
        allow_tool_version_drift=allow_tool_version_drift,
    )
    needs_reindex = bool(health["needs_reindex"])
    reindex_reason = health["reindex_reason"]
    resume_contract = health["resume_contract"]
    assert isinstance(resume_contract, dict)

    resolved_path_prefix = path_prefix if path_prefix else file_path
    resolved_max_snippets = max_snippets if max_snippets is not None else top_k
    query_mode = mode.strip().lower()
    normalized_search_mode = search_mode.strip().lower() if search_mode else "semantic"

    if needs_reindex:
        payload = _build_search_not_ready_payload(query=query, health=health)
        if debug_router:
            payload["debug"] = {
                "search_health": health,
                "query_mode": query_mode,
                "search_mode": normalized_search_mode,
                "language": language,
                "path_prefix": resolved_path_prefix,
                "max_files": max_files,
                "max_snippets": resolved_max_snippets,
                "time_budget_ms": time_budget_ms,
                "ranking_mode": ranking_mode,
                "kind": kind,
                "context_radius": context_radius,
                "top_k": top_k,
                "confidence_threshold": confidence_threshold,
                "max_requery_attempts": max_requery_attempts,
                "bounded_retry_enabled": not disable_bounded_requery,
                "tool_version": GLOGGUR_VERSION,
            }
        _emit(payload, as_json)
        raise click.exceptions.Exit(1)

    metadata_store = _create_metadata_store(config)
    searcher: HybridSearch | None = None
    semantic_init_error: str | None = None
    try:
        embedding = _create_embedding_provider_for_command(
            config,
            require_provider=False,
        )
    except EmbeddingProviderError as exc:
        embedding = None
        semantic_init_error = format_embedding_error_message(exc)
    if embedding is not None:
        searcher = HybridSearch(
            embedding,
            vector_store,
            metadata_store,
            health_evaluator=lambda: _build_search_health_snapshot(
                config,
                cache,
                entrypoint="hybrid_search_legacy",
                contract_version="legacy",
                allow_tool_version_drift=allow_tool_version_drift,
            ),
        )

    router_repo_root = _resolve_router_repo_root(
        metadata_store=metadata_store,
        fallback=Path.cwd(),
    )
    symbol_store = SymbolIndexStore(
        SymbolIndexStoreConfig(repo_root=router_repo_root),
        create_if_missing=False,
    )
    router_config = load_search_router_config(router_repo_root)
    router = SearchRouter(
        repo_root=router_repo_root,
        searcher=searcher,
        metadata_store=metadata_store,
        symbol_store=symbol_store,
        config=router_config,
    )
    routing_path_prefix = _resolve_search_path_filter_for_routing(
        repo_root=router_repo_root,
        raw_path=resolved_path_prefix,
    )
    constraints = SearchConstraints(
        search_mode=normalized_search_mode,
        language=language,
        path_prefix=routing_path_prefix,
        max_files=max_files,
        max_snippets=resolved_max_snippets,
        time_budget_ms=time_budget_ms,
    )
    pack = router.search(
        query=query,
        constraints=constraints,
        mode=mode,
        include_debug=debug_router,
    )
    payload = pack.to_dict(include_debug=debug_router)
    summary_payload = payload.get("summary")
    if isinstance(summary_payload, dict):
        existing_warning_codes = summary_payload.get("warning_codes")
        merged_warning_codes: list[str] = []
        if isinstance(existing_warning_codes, list):
            for code in existing_warning_codes:
                if isinstance(code, str) and code and code not in merged_warning_codes:
                    merged_warning_codes.append(code)
        for code in health["warning_codes"]:
            if isinstance(code, str) and code and code not in merged_warning_codes:
                merged_warning_codes.append(code)
        summary_payload.setdefault("query_mode", query_mode)
        summary_payload.setdefault("search_mode", normalized_search_mode)
        summary_payload.setdefault("needs_reindex", health["needs_reindex"])
        summary_payload.setdefault("reindex_reason", reindex_reason)
        summary_payload.setdefault("expected_index_profile", health["expected_index_profile"])
        summary_payload.setdefault("cached_index_profile", health["cached_index_profile"])
        summary_payload.setdefault("entrypoint", health["entrypoint"])
        summary_payload.setdefault("contract_version", health["contract_version"])
        summary_payload["warning_codes"] = merged_warning_codes
        summary_payload.setdefault("semantic_search_allowed", health["semantic_search_allowed"])
        summary_payload.setdefault("search_integrity", health["search_integrity"])
        # Preserve legacy option observability as non-routing summary fields.
        summary_payload.setdefault("legacy_ranking_mode", ranking_mode)
        summary_payload.setdefault("legacy_kind_filter", kind)
        summary_payload.setdefault("legacy_context_radius", context_radius)
        summary_payload.setdefault("legacy_top_k", top_k)
        summary_payload.setdefault("legacy_confidence_threshold", confidence_threshold)
        summary_payload.setdefault("legacy_max_requery_attempts", max_requery_attempts)
        summary_payload.setdefault("legacy_retry_enabled", not disable_bounded_requery)
        summary_payload.setdefault("file_filter", resolved_path_prefix)
        summary_payload.setdefault("tool_version", GLOGGUR_VERSION)
        for key, value in resume_contract.items():
            summary_payload.setdefault(str(key), value)

    if debug_router:
        debug_payload = payload.get("debug")
        if not isinstance(debug_payload, dict):
            debug_payload = {}
            payload["debug"] = debug_payload
        debug_payload["resume_contract"] = resume_contract
        debug_payload["search_health"] = health
        debug_payload["tool_version"] = GLOGGUR_VERSION
        if semantic_init_error is not None:
            backend_errors = debug_payload.get("backend_errors")
            if not isinstance(backend_errors, dict):
                backend_errors = {}
                debug_payload["backend_errors"] = backend_errors
            backend_errors.setdefault("semantic_init", semantic_init_error)

    debug_info = pack.debug or {}
    backend_scores = debug_info.get("backend_scores")
    backend_errors = debug_info.get("backend_errors")
    if (
        isinstance(backend_scores, dict)
        and isinstance(backend_errors, dict)
        and backend_scores
        and len(backend_errors) >= len(backend_scores)
        and not pack.hits
    ):
        raise CLIContractError(
            "Search router backends failed to return usable evidence.",
            error_code="search_router_backends_failed",
        )

    if stream and as_json:
        hits_payload = payload.get("hits")
        if isinstance(hits_payload, list):
            for item in hits_payload:
                click.echo(json.dumps(item))
            return
        raise CLIContractError(
            "search response contract invalid: hits must be a list in stream mode",
            error_code="search_result_payload_invalid",
        )
    _emit(payload, as_json)


@cli.group()
def graph() -> None:
    """Traverse and search extracted reference-graph edges."""


@graph.command("neighbors")
@click.argument("symbol_id", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--edge-type", "edge_type", type=str, default=None)
@click.option(
    "--direction",
    type=click.Choice(["both", "incoming", "outgoing"], case_sensitive=False),
    default="both",
    show_default=True,
)
@click.option("--k", type=click.IntRange(1, 1000), default=20, show_default=True)
@_with_io_failure_handling
def graph_neighbors(
    symbol_id: str,
    config_path: str | None,
    as_json: bool,
    edge_type: str | None,
    direction: str,
    k: int,
) -> None:
    """Return ranked neighboring edges for a symbol endpoint."""
    config = _load_config(config_path)
    metadata_store = _create_metadata_store(config)
    service = GraphService(metadata_store=metadata_store)
    payload = service.neighbors(
        symbol_id,
        edge_type=edge_type,
        direction=direction.lower(),
        k=k,
    )
    _emit(payload, as_json)


@graph.command("incoming")
@click.argument("symbol_id", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--edge-type", "edge_type", type=str, default=None)
@click.option("--k", type=click.IntRange(1, 1000), default=20, show_default=True)
@_with_io_failure_handling
def graph_incoming(
    symbol_id: str,
    config_path: str | None,
    as_json: bool,
    edge_type: str | None,
    k: int,
) -> None:
    """Return incoming edges for a symbol."""
    config = _load_config(config_path)
    metadata_store = _create_metadata_store(config)
    service = GraphService(metadata_store=metadata_store)
    payload = service.incoming(symbol_id, edge_type=edge_type, k=k)
    _emit(payload, as_json)


@graph.command("outgoing")
@click.argument("symbol_id", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--edge-type", "edge_type", type=str, default=None)
@click.option("--k", type=click.IntRange(1, 1000), default=20, show_default=True)
@_with_io_failure_handling
def graph_outgoing(
    symbol_id: str,
    config_path: str | None,
    as_json: bool,
    edge_type: str | None,
    k: int,
) -> None:
    """Return outgoing edges for a symbol."""
    config = _load_config(config_path)
    metadata_store = _create_metadata_store(config)
    service = GraphService(metadata_store=metadata_store)
    payload = service.outgoing(symbol_id, edge_type=edge_type, k=k)
    _emit(payload, as_json)


@graph.command("search")
@click.argument("query", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--edge-type", "edge_type", type=str, default=None)
@click.option("--k", type=click.IntRange(1, 1000), default=20, show_default=True)
@click.option("--embedding-provider", type=str, default=None)
@_with_io_failure_handling
def graph_search(
    query: str,
    config_path: str | None,
    as_json: bool,
    edge_type: str | None,
    k: int,
    embedding_provider: str | None,
) -> None:
    """Semantic search over edge fact records."""
    config = _load_config(config_path)
    if embedding_provider:
        config.embedding_provider = embedding_provider
    embedding = _create_embedding_provider_for_command(config, require_provider=True)
    metadata_store = _create_metadata_store(config)
    service = GraphService(metadata_store=metadata_store, embedding_provider=embedding)
    payload = service.search(query, edge_type=edge_type, k=k)
    _emit(payload, as_json)


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


def _build_inspect_warning_summary(
    warning_reports: list[dict[str, object]],
    *,
    symbol_file_paths: dict[str, str],
) -> dict[str, object]:
    """Build deterministic warning summaries for inspect JSON payloads."""
    warning_counts_by_type: dict[str, int] = {}
    warning_counts_by_path_class: dict[str, int] = {
        "src": 0,
        "tests": 0,
        "scripts": 0,
        "other": 0,
    }
    report_counts_by_path_class: dict[str, int] = {
        "src": 0,
        "tests": 0,
        "scripts": 0,
        "other": 0,
    }
    warning_counts_by_file: dict[str, int] = {}
    total_warnings = 0

    for report in warning_reports:
        symbol_id = report.get("symbol_id")
        if not isinstance(symbol_id, str):
            continue
        report_warnings = report.get("warnings")
        if not isinstance(report_warnings, list):
            continue
        file_path = symbol_file_paths.get(symbol_id)
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


def _load_cached_inspect_reports(
    cache: CacheManager,
    file_path: str,
    *,
    symbol_ids: tuple[str, ...],
) -> list[DocstringAuditReport]:
    """Rehydrate cached audit reports for unchanged files."""
    cached_rows = cache.list_audit_reports_for_file(file_path)
    selected_rows = (
        [
            (symbol_id, warnings, semantic_score, score_metadata)
            for symbol_id, warnings, semantic_score, score_metadata in cached_rows
            if symbol_id in symbol_ids
        ]
        if symbol_ids
        else cached_rows
    )
    reports: list[DocstringAuditReport] = []
    for symbol_id, warnings, semantic_score, score_metadata in selected_rows:
        cached_score_metadata = dict(score_metadata) if isinstance(score_metadata, dict) else {}
        cached_score_metadata["cached_report_reuse"] = True
        if warnings:
            cached_score_metadata["cached_warning_reuse"] = True
        reports.append(
            DocstringAuditReport(
                symbol_id=symbol_id,
                warnings=warnings,
                semantic_score=semantic_score,
                score_metadata=cached_score_metadata or None,
            )
        )
    return reports


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
    config_path: str | None,
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
    parser_registry = ParserRegistry(
        extension_map=config.parser_extension_map,
        adapter_overrides=config.adapters if isinstance(config.adapters, dict) else None,
    )
    embedding = _create_embedding_provider_for_command(config)
    symbols = []
    code_texts: dict[str, str] = {}
    processed_files: list[tuple[str, str]] = []
    files_considered = 0
    inspected_files = 0
    failed_files = 0
    failed_reasons: dict[str, int] = {}
    failed_samples: list[str] = []
    skipped_files = 0
    cached_files_reused = 0
    cached_reports_reused = 0
    cached_warning_reports_reused = 0
    symbol_file_paths: dict[str, str] = {}
    reports: list[DocstringAuditReport] = []
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
            with open(file_path, encoding="utf8") as handle:
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
                cached_reports = _load_cached_inspect_reports(
                    cache,
                    file_path,
                    symbol_ids=symbol_ids,
                )
                reports.extend(cached_reports)
                cached_files_reused += 1
                cached_reports_reused += len(cached_reports)
                cached_warning_reports_reused += sum(
                    1 for report in cached_reports if report.warnings
                )
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
    fresh_reports = audit_docstrings(
        symbols,
        code_texts=code_texts,
        embedding_provider=embedding,
        semantic_threshold=config.docstring_semantic_threshold,
        semantic_min_chars=config.docstring_semantic_min_chars,
        semantic_max_chars=config.docstring_semantic_max_chars,
        semantic_min_code_chars=config.docstring_semantic_min_code_chars,
        kind_thresholds=config.docstring_semantic_kind_thresholds,
    )
    reports.extend(fresh_reports)
    for file_path, _content_hash in processed_files:
        cache.delete_audit_reports_for_file(file_path)
    for report in fresh_reports:
        cache.set_audit_report(
            report.symbol_id,
            warnings=report.warnings,
            file_path=symbol_file_paths.get(report.symbol_id),
            semantic_score=report.semantic_score,
            score_metadata=report.score_metadata,
        )
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
        "inspect_payload_schema_policy": INSPECT_PAYLOAD_SCHEMA_BUMP_POLICY,
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
        "cached_files_reused": cached_files_reused,
        "cached_reports_reused": cached_reports_reused,
        "cached_warning_reports_reused": cached_warning_reports_reused,
        "failure_codes": [],
        "failure_guidance": {},
        "allow_partial": allow_partial,
        "allow_partial_applied": allow_partial and failed_files > 0,
        "inspected_files": inspected_files,
        "skipped_files": skipped_files,
    }
    payload.update(_build_inspect_failure_contract(failed_reasons))
    if as_json and failed_files > 0 and not allow_partial:
        _attach_primary_error_from_failure_contract(
            payload,
            error_type="inspect_failure",
            detail="Inspect completed with file-level failures.",
            probable_cause=(
                "One or more files could not be inspected; inspect failed_reasons and "
                "failed_samples for the per-file cause."
            ),
            default_remediation=DEFAULT_INSPECT_FAILURE_REMEDIATION,
        )
    _emit(payload, as_json)
    if failed_files > 0 and not allow_partial:
        raise click.exceptions.Exit(1)


@cli.group()
def artifact() -> None:
    """Package, validate, and restore cache artifacts for CI/CD reuse."""


@artifact.command("publish")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option(
    "--source",
    "source_path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=None,
    help="Cache directory to package (defaults to config cache_dir).",
)
@click.option(
    "--destination",
    type=str,
    required=True,
    help="Artifact output path or file:// URI.",
)
@click.option(
    "--name",
    "artifact_name",
    type=str,
    default=None,
    help="Default artifact filename used when destination is a directory.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite destination file when it already exists.",
)
@click.option(
    "--uploader-command",
    type=str,
    default=None,
    help=(
        "External uploader argv-template. Supported placeholders: "
        "{artifact_path}, {artifact}, {destination}, {artifact_name}, "
        "{archive_sha256}, {archive_bytes}, {manifest_sha256}."
    ),
)
@click.option(
    "--uploader-timeout-seconds",
    type=click.FloatRange(min=0.0, min_open=True),
    default=60.0,
    show_default=True,
    help="Timeout for the external uploader command.",
)
@_with_io_failure_handling
def artifact_publish(
    config_path: str | None,
    as_json: bool,
    source_path: str | None,
    destination: str,
    artifact_name: str | None,
    overwrite: bool,
    uploader_command: str | None,
    uploader_timeout_seconds: float,
) -> None:
    """Publish a deterministic index artifact to local/file, HTTP, or uploader transports."""
    config = _load_config(config_path)
    source_dir = os.path.abspath(os.path.expanduser(source_path or config.cache_dir))
    if not os.path.exists(source_dir):
        raise CLIContractError(
            f"Artifact source path does not exist: {source_dir}",
            error_code="artifact_source_missing",
        )
    if not os.path.isdir(source_dir):
        raise CLIContractError(
            f"Artifact source path is not a directory: {source_dir}",
            error_code="artifact_source_not_directory",
        )
    source_db_path = os.path.join(source_dir, "index.db")
    if not os.path.exists(source_db_path):
        raise CLIContractError(
            f"Artifact source cache database is missing: {source_db_path}",
            error_code="artifact_source_uninitialized",
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_filename = artifact_name or f"gloggur-index-{timestamp}.tar.gz"
    destination_scheme = urlparse(destination).scheme.lower()
    use_http_upload = uploader_command is None and destination_scheme in {"http", "https"}
    destination_path: str | None = None
    artifact_uri = destination
    if uploader_command is not None:
        publish_transport = "uploader_command"
    elif use_http_upload:
        publish_transport = "http_put"
    else:
        publish_transport = "local_path"
    if uploader_command is None and not use_http_upload:
        destination_path = _resolve_artifact_destination(
            destination,
            default_filename=default_filename,
        )
        artifact_uri = Path(destination_path).resolve().as_uri()
        try:
            if os.path.commonpath([source_dir, os.path.abspath(destination_path)]) == source_dir:
                raise CLIContractError(
                    "Artifact destination must be outside the source cache directory.",
                    error_code="artifact_destination_inside_source",
                )
        except ValueError:
            pass
        if os.path.exists(destination_path) and not overwrite:
            raise CLIContractError(
                f"Artifact destination already exists: {destination_path}",
                error_code="artifact_destination_exists",
            )
        destination_parent = os.path.dirname(destination_path) or "."
        try:
            os.makedirs(destination_parent, exist_ok=True)
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="create artifact destination directory",
                path=destination_parent,
            ) from exc

    cache = _create_cache_manager(source_dir)
    file_entries = _collect_artifact_file_entries(source_dir)
    manifest = _build_artifact_manifest(
        cache,
        source_dir=source_dir,
        file_entries=file_entries,
    )
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf8") + b"\n"
    manifest_sha256 = _sha256_bytes(manifest_bytes)

    temp_fd, temp_archive = tempfile.mkstemp(prefix="gloggur-artifact-", suffix=".tar.gz")
    os.close(temp_fd)
    try:
        _create_artifact_archive(
            source_dir,
            artifact_path=temp_archive,
            manifest_bytes=manifest_bytes,
            file_entries=file_entries,
        )
        archive_sha256 = _sha256_file(temp_archive)
        try:
            archive_bytes = os.path.getsize(temp_archive)
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="read published artifact metadata",
                path=temp_archive,
            ) from exc
        uploader_payload: dict[str, object] | None = None
        http_upload_payload: dict[str, object] | None = None
        if destination_path is not None:
            try:
                shutil.copyfile(temp_archive, destination_path)
            except OSError as exc:
                raise wrap_io_error(
                    exc,
                    operation="publish artifact to destination",
                    path=destination_path,
                ) from exc
            archive_sha256 = _sha256_file(destination_path)
            try:
                archive_bytes = os.path.getsize(destination_path)
            except OSError as exc:
                raise wrap_io_error(
                    exc,
                    operation="read published artifact metadata",
                    path=destination_path,
                ) from exc
            archive_path_for_payload: str | None = destination_path
            artifact_uri = Path(destination_path).resolve().as_uri()
        elif use_http_upload:
            http_upload_payload = _upload_artifact_http(
                destination,
                artifact_path=temp_archive,
                archive_sha256=archive_sha256,
                archive_bytes=archive_bytes,
                manifest_sha256=manifest_sha256,
                timeout_seconds=uploader_timeout_seconds,
            )
            archive_path_for_payload = None
        else:
            uploader_payload = _run_artifact_uploader_command(
                str(uploader_command),
                artifact_path=temp_archive,
                destination=destination,
                artifact_name=default_filename,
                archive_sha256=archive_sha256,
                archive_bytes=archive_bytes,
                manifest_sha256=manifest_sha256,
                timeout_seconds=uploader_timeout_seconds,
            )
            archive_path_for_payload = None
    finally:
        try:
            os.remove(temp_archive)
        except OSError:
            pass

    payload = {
        "published": True,
        "transport": publish_transport,
        "source_cache_dir": source_dir,
        "artifact_destination": destination,
        "artifact_path": archive_path_for_payload,
        "artifact_uri": artifact_uri,
        "archive_sha256": archive_sha256,
        "manifest_sha256": manifest_sha256,
        "manifest_path": "manifest.json",
        "manifest": manifest,
        "archive_bytes": archive_bytes,
    }
    if uploader_payload is not None:
        payload["uploader"] = uploader_payload
    if http_upload_payload is not None:
        payload["http_upload"] = http_upload_payload
    _emit(payload, as_json)


@artifact.command("validate")
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option(
    "--artifact",
    "artifact_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
    help="Path to artifact .tar.gz file to validate.",
)
@click.option(
    "--skip-file-hash-check",
    is_flag=True,
    default=False,
    help="Validate manifest shape/totals without hashing every archived cache file.",
)
@_with_io_failure_handling
def artifact_validate(as_json: bool, artifact_path: str, skip_file_hash_check: bool) -> None:
    """Validate a published cache artifact and emit deterministic metadata."""
    payload = _validate_artifact_archive(
        artifact_path,
        verify_file_hashes=not skip_file_hash_check,
    )
    _emit(payload, as_json)


@artifact.command("restore")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option(
    "--artifact",
    "artifact_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
    help="Path to artifact .tar.gz file to restore.",
)
@click.option(
    "--destination",
    "destination_dir",
    type=click.Path(exists=False, file_okay=True, dir_okay=True),
    default=None,
    help="Cache directory to restore into (defaults to config cache_dir).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Replace destination directory if it already exists.",
)
@click.option(
    "--skip-file-hash-check",
    is_flag=True,
    default=False,
    help="Validate manifest shape/totals before restore without hashing every archived cache file.",
)
@_with_io_failure_handling
def artifact_restore(
    config_path: str | None,
    as_json: bool,
    artifact_path: str,
    destination_dir: str | None,
    overwrite: bool,
    skip_file_hash_check: bool,
) -> None:
    """Restore a validated cache artifact into a destination cache directory."""
    config = _load_config(config_path)
    payload = _restore_artifact_archive(
        artifact_path,
        destination_dir=destination_dir or config.cache_dir,
        overwrite=overwrite,
        verify_file_hashes=not skip_file_hash_check,
    )
    _emit(payload, as_json)


def _support_callbacks() -> SupportCallbacks:
    """Build support-command callbacks from existing CLI helpers."""
    return SupportCallbacks(
        load_config=_load_support_config,
        build_status_payload=lambda config: _create_status_payload(config),
        build_watch_status_payload=_create_watch_status_payload,
    )


@cli.group()
def support() -> None:
    """Capture traced support sessions and package field diagnostics."""


@support.command("run", context_settings={"ignore_unknown_options": True})
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--note", type=str, default=None, help="Optional tester note saved into notes.txt.")
@click.option(
    "--bundle-on-failure/--no-bundle-on-failure",
    default=True,
    show_default=True,
    help="Automatically create a support bundle when the child command exits non-zero.",
)
@click.option(
    "--destination",
    type=click.Path(exists=False, file_okay=True, dir_okay=True),
    default=None,
    help="Optional bundle output path used only when a failure bundle is created.",
)
@click.argument("child_args", nargs=-1, type=click.UNPROCESSED)
@_with_io_failure_handling
def support_run(
    as_json: bool,
    note: str | None,
    bundle_on_failure: bool,
    destination: str | None,
    child_args: tuple[str, ...],
) -> None:
    """Run a Gloggur subcommand inside a traced support session."""
    try:
        payload, exit_code = run_support_command_impl(
            as_json=as_json,
            child_args=child_args,
            note=note,
            bundle_on_failure=bundle_on_failure,
            destination=destination,
            callbacks=_support_callbacks(),
        )
    except SupportContractError as exc:
        raise CLIContractError(str(exc), error_code=exc.code) from exc
    _emit(payload, as_json)
    if exit_code != 0:
        raise click.exceptions.Exit(exit_code)


@support.command("collect")
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option(
    "--session",
    "session_id",
    type=str,
    default=None,
    help="Existing support session id.",
)
@click.option(
    "--destination",
    type=click.Path(exists=False, file_okay=True, dir_okay=True),
    default=None,
    help="Output path for the support bundle archive.",
)
@click.option(
    "--include-cache",
    is_flag=True,
    default=False,
    help="Include raw cache/index artifacts.",
)
@click.option("--note", type=str, default=None, help="Optional tester note saved into notes.txt.")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite an existing bundle path.")
@_with_io_failure_handling
def support_collect(
    as_json: bool,
    session_id: str | None,
    destination: str | None,
    include_cache: bool,
    note: str | None,
    overwrite: bool,
) -> None:
    """Collect diagnostics and create a deterministic support bundle."""
    try:
        payload = collect_support_bundle_impl(
            session_id=session_id,
            note=note,
            destination=destination,
            include_cache=include_cache,
            overwrite=overwrite,
            callbacks=_support_callbacks(),
        )
    except SupportContractError as exc:
        raise CLIContractError(str(exc), error_code=exc.code) from exc
    _emit(payload, as_json)


@cli.group()
def watch() -> None:
    """Manage save-triggered incremental indexing."""


@watch.command("init")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@_with_io_failure_handling
def watch_init(path: str, config_path: str | None, as_json: bool) -> None:
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
    config_path: str | None,
    as_json: bool,
    force_foreground: bool,
    force_daemon: bool,
    allow_partial: bool,
) -> None:
    """Start watcher in foreground or daemon mode."""

    if force_foreground and force_daemon:
        raise CLIContractError(
            "Use only one of --foreground or --daemon.",
            error_code="watch_mode_conflict",
        )

    resolved_config_path = _normalize_config_path(config_path)
    config, cache, vector_store = _create_runtime(
        config_path=resolved_config_path,
        rebuild_on_profile_change=True,
    )
    watch_path = os.path.abspath(config.watch_path)
    if not os.path.exists(watch_path):
        raise CLIContractError(
            f"Watch path does not exist: {watch_path}",
            error_code="watch_path_missing",
        )

    mode = config.watch_mode
    if force_foreground:
        mode = "foreground"
    if force_daemon:
        mode = "daemon"
    if mode not in {"foreground", "daemon"}:
        raise CLIContractError(
            f"Unsupported watch mode: {mode}",
            error_code="watch_mode_invalid",
        )

    pid_path = config.watch_pid_file
    pid = _read_pid_file(pid_path)
    daemon_child = os.getenv("GLOGGUR_WATCH_DAEMON_CHILD") == "1"
    if daemon_child and pid == os.getpid():
        pid = None
    if is_process_running(pid):
        _emit({"started": False, "reason": "already_running", "pid": pid}, as_json)
        return

    embedding = _create_embedding_provider_for_command(config)
    service = _create_watch_service(
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
                        "last_error": (f"watch daemon exited early with code {daemon_exit_code}"),
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
                    "Fix configuration/dependency issues and "
                    "rerun `gloggur watch start --daemon "
                    "--json`.",
                ],
                detail=f"RuntimeError: watch daemon exited early with code {daemon_exit_code}",
            )
        try:
            _write_pid_file(pid_path, process.pid)
            _write_watch_state(
                config.watch_state_file,
                _watch_starting_state_payload(watch_path=watch_path, pid=process.pid),
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
                        "last_error": (f"watch daemon exited early with code {daemon_exit_code}"),
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
                    "Fix configuration/dependency issues and "
                    "rerun `gloggur watch start --daemon "
                    "--json`.",
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
    payload = {
        "started": True,
        "mode": "foreground",
        "pid": os.getpid(),
        **result,
    }
    failed_count = int(result.get("failed", result.get("error_count", 0)))
    if as_json and failed_count > 0 and not allow_partial:
        _attach_primary_error_from_failure_contract(
            payload,
            error_type="watch_failure",
            detail="Watch foreground run completed with file-level failures.",
            probable_cause=(
                "One or more incremental watch batches failed; inspect failed_reasons and "
                "failed_samples for the underlying indexing or cleanup cause."
            ),
            default_remediation=DEFAULT_WATCH_FAILURE_REMEDIATION,
        )
    _emit(payload, as_json)
    if failed_count > 0 and not allow_partial:
        raise click.exceptions.Exit(1)


@watch.command("stop")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@_with_io_failure_handling
def watch_stop(config_path: str | None, as_json: bool) -> None:
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
def watch_status(config_path: str | None, as_json: bool) -> None:
    """Show watcher process and heartbeat status."""

    resolved_config_path = _normalize_config_path(config_path)
    config = _normalize_watch_paths(_load_config(resolved_config_path), resolved_config_path)
    payload = _create_watch_status_payload(config)
    _emit(payload, as_json)


@cli.command()
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option(
    "--allow-tool-version-drift",
    is_flag=True,
    default=False,
    help="Allow status resume_ok when only tool-version drift is detected.",
)
@_with_io_failure_handling
def status(config_path: str | None, as_json: bool, allow_tool_version_drift: bool) -> None:
    """Show index statistics and metadata."""
    allow_tool_version_drift = _resolve_allow_tool_version_drift(
        cli_flag_enabled=allow_tool_version_drift
    )
    config = _load_config(config_path)
    try:
        payload = _create_status_payload(
            config,
            allow_tool_version_drift=allow_tool_version_drift,
        )
    except StorageIOError as error:
        if not _is_transient_status_race_error(error):
            raise
        try:
            payload = _create_status_payload(
                config,
                allow_tool_version_drift=allow_tool_version_drift,
            )
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
def clear_cache(config_path: str | None, as_json: bool, profile_filter: str | None) -> None:
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
        vector_store = _create_vector_store(config, load_existing=False)
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


@cli.command("guidance")
@click.argument("symbol_id", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--embedding-provider", type=str, default=None)
@_with_io_failure_handling
def guidance(
    symbol_id: str,
    config_path: str | None,
    as_json: bool,
    embedding_provider: str | None,
) -> None:
    """Generate agent-consumable context for a given symbol."""
    from gloggur.search.guidance import AgentGuidance

    resolved_config_path = _normalize_config_path(config_path)
    config = _load_config(resolved_config_path)
    config, cache_manager, vector_store = _create_runtime(
        config_path=resolved_config_path,
        embedding_provider=embedding_provider,
        rebuild_on_profile_change=False,
        write_locked=False,
    )
    embedding = _create_embedding_provider_for_command(
        config,
        require_provider=True,
    )
    if embedding is None:
        raise click.exceptions.Exit(1)
    metadata_store = _create_metadata_store(config)
    searcher = HybridSearch(embedding, vector_store, metadata_store)
    guidance_layer = AgentGuidance(searcher)

    payload = guidance_layer.generate_agent_context(symbol_id)
    if "error" in payload:
        click.echo(payload["error"], err=True)
        raise click.exceptions.Exit(1)

    _emit(payload, as_json)


@coverage.command("ingest")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to gloggur configuration file.",
)
@click.option("--json", "as_json", is_flag=True, help="Output results in JSON format.")
@_with_io_failure_handling
def coverage_ingest(file: str, config_path: str | None, as_json: bool) -> None:
    """Ingest a generic gloggur-coverage.json mapping and apply to index."""
    start_time = time.time()
    config = _load_config(config_path)

    metadata = _create_metadata_store(config)
    ingester = CoverageIngester(metadata)

    try:
        report = ingester.ingest_json(file)
    except ValueError as exc:
        raise CLIContractError(
            f"Coverage schema error: {exc}",
            error_code="coverage_file_invalid",
        ) from exc

    # Apply to cache
    symbols_to_update = report.get("symbols_to_update", [])
    if symbols_to_update:
        cache = CacheManager(CacheConfig(cache_dir=config.cache_dir))
        with cache_write_lock(config.cache_dir):
            cache.upsert_symbols(symbols_to_update)

    duration_ms = int((time.time() - start_time) * 1000)
    payload = {
        "tests_processed": report.get("tests_processed", 0),
        "files_affected": report.get("files_affected", 0),
        "symbols_updated": len(symbols_to_update),
        "duration_ms": duration_ms,
    }

    if as_json:
        _emit(payload, as_json=True)
    else:
        click.echo("Coverage ingestion complete:")
        click.echo(f"  Tests mapped:    {payload['tests_processed']}")
        click.echo(f"  Files affected:  {payload['files_affected']}")
        click.echo(f"  Symbols updated: {payload['symbols_updated']}")
        click.echo(f"  Duration:        {duration_ms}ms")


@coverage.command("import-python")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to save the resulting gloggur-coverage.json file.",
)
@click.option("--json", "as_json", is_flag=True, help="Output results in JSON format.")
@_with_io_failure_handling
def coverage_import_python(file: str, output: str, as_json: bool) -> None:
    """Convert a Python coverage.py SQLite database into gloggur-coverage.json."""
    _run_coverage_import(
        file=file,
        output=output,
        importer_id="python",
        as_json=as_json,
    )


def _run_coverage_import(
    *,
    file: str,
    output: str,
    importer_id: str,
    as_json: bool,
) -> None:
    """Run coverage import through adapter registry and emit deterministic payloads."""
    start_time = time.time()
    config = _load_config(None)
    try:
        importer = create_coverage_importer(config, importer_id)
        contexts = importer.import_contexts(file)
    except CoverageImportError as exc:
        raise CLIContractError(
            str(exc),
            error_code=exc.error_code,
        ) from exc
    except AdapterResolutionError as exc:
        raise CLIContractError(
            f"Unknown coverage importer: {importer_id}",
            error_code="coverage_file_invalid",
        ) from exc

    try:
        with open(output, "w", encoding="utf8") as handle:
            json.dump(contexts, handle, indent=2)
    except OSError as exc:
        raise wrap_io_error(exc, operation="write JSON coverage file", path=output) from exc

    duration_ms = int((time.time() - start_time) * 1000)
    payload = {
        "tests_extracted": len(contexts),
        "output_file": output,
        "duration_ms": duration_ms,
    }

    if as_json:
        _emit(payload, as_json=True)
    else:
        click.echo("Coverage import complete:")
        click.echo(f"  Tests extracted: {len(contexts)}")
        click.echo(f"  Written to:      {output}")
        click.echo(f"  Duration:        {duration_ms}ms")


@coverage.command("import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to save the resulting gloggur-coverage.json file.",
)
@click.option(
    "--importer",
    "importer_id",
    type=str,
    default="python",
    show_default=True,
    help="Coverage importer adapter id.",
)
@click.option("--json", "as_json", is_flag=True, help="Output results in JSON format.")
@_with_io_failure_handling
def coverage_import(file: str, output: str, importer_id: str, as_json: bool) -> None:
    """Convert external coverage formats into gloggur-coverage.json via adapter."""
    _run_coverage_import(
        file=file,
        output=output,
        importer_id=importer_id,
        as_json=as_json,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI with deterministic JSON error envelopes in --json mode."""
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    try:
        cli_exit_code = cli.main(args=args, prog_name="gloggur", standalone_mode=False)
        if isinstance(cli_exit_code, int):
            return cli_exit_code
        return 0
    except click.exceptions.Exit as exc:
        return int(exc.exit_code or 0)
    except click.ClickException as exc:
        if as_json:
            error_code = "cli_usage_error"
            guidance = CLI_FAILURE_REMEDIATION.get(
                error_code,
                [DEFAULT_CLI_FAILURE_REMEDIATION],
            )
            _emit_json_error(
                _json_error_envelope(
                    error_code=error_code,
                    error=exc.message,
                    stage="dispatch",
                    compatibility={
                        "error": {
                            "type": "cli_usage_error",
                            "code": error_code,
                            "detail": exc.message,
                            "probable_cause": (
                                "Command arguments/options were invalid for this CLI path."
                            ),
                            "remediation": guidance,
                        },
                        "failure_codes": [error_code],
                        "failure_guidance": {error_code: guidance},
                    },
                )
            )
            return int(exc.exit_code or 1)
        exc.show(file=sys.stderr)
        return int(exc.exit_code or 1)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        if as_json:
            _emit_json_error(
                _json_error_envelope(
                    error_code="broken_environment",
                    error="Unexpected CLI runtime failure.",
                    stage="dispatch",
                    compatibility={"exception": repr(exc)},
                )
            )
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
