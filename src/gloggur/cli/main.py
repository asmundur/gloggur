from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import shlex
import signal
import socket
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import unquote, urlparse

import click
import yaml

from gloggur import __version__ as GLOGGUR_VERSION
from gloggur.audit.docstring_audit import DocstringAuditReport, audit_docstrings
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
from gloggur.search.evidence import (
    build_evidence_trace,
    validate_evidence_trace,
)
from gloggur.search.hybrid_search import HybridSearch
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.watch.service import (
    DEFAULT_WATCH_FAILURE_REMEDIATION,
    WatchService,
    is_process_running,
    load_watch_state,
    utc_now_iso,
)


@click.group()
def cli() -> None:
    """Gloggur CLI for indexing, search, and docstring inspection."""


INSPECT_PAYLOAD_SCHEMA_VERSION = "1"
INSPECT_PAYLOAD_SCHEMA_POLICY_VERSION = "1"
INSPECT_PAYLOAD_SCHEMA_BUMP_POLICY: Dict[str, object] = {
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
    "Inspect failed_samples, then rerun `gloggur inspect <path> --json --force` after resolving the file-level failure."
)
DEFAULT_CLI_FAILURE_REMEDIATION = (
    "Resolve the CLI precondition failure and rerun the command with --json for machine-readable diagnostics."
)
ARTIFACT_MANIFEST_SCHEMA_VERSION = "1"
DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_MAX_REQUERY_ATTEMPTS = 1
DEFAULT_EVIDENCE_MIN_CONFIDENCE = 0.6
DEFAULT_EVIDENCE_MIN_ITEMS = 1
MAX_REQUERY_TOP_K = 64
REQUERY_STRATEGY_TOP_K_EXPANSION = "top_k_expansion"
CLI_FAILURE_REMEDIATION: Dict[str, List[str]] = {
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
    "artifact_source_missing": [
        "Set --source to an existing cache directory or run `gloggur index . --json` to create one.",
    ],
    "artifact_source_not_directory": [
        "Set --source to a directory path containing cache artifacts.",
    ],
    "artifact_source_uninitialized": [
        "Run `gloggur index . --json` before publishing so metadata and vectors are present.",
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
        "Set --artifact to an existing .tar.gz artifact path created by `gloggur artifact publish`.",
    ],
    "artifact_path_not_file": [
        "Set --artifact to a regular file path (not a directory).",
    ],
    "artifact_archive_invalid": [
        "Artifact is not a readable tar.gz archive; rebuild and republish the artifact.",
    ],
    "artifact_manifest_missing": [
        "Artifact is missing manifest.json; republish with `gloggur artifact publish --json`.",
    ],
    "artifact_manifest_invalid": [
        "manifest.json is malformed or missing required fields; republish artifact from a healthy cache.",
    ],
    "artifact_manifest_schema_unsupported": [
        "Artifact manifest schema is unsupported by this CLI version; rebuild with a compatible gloggur version.",
    ],
    "artifact_manifest_file_mismatch": [
        "Artifact file checksums/sizes do not match manifest entries; treat artifact as corrupted and republish.",
    ],
    "artifact_manifest_totals_mismatch": [
        "Manifest aggregate totals do not match file entries; republish artifact from source cache.",
    ],
    "artifact_restore_destination_exists": [
        "Choose a new restore destination or pass --overwrite to replace the existing cache directory.",
    ],
    "artifact_restore_destination_not_directory": [
        "Set --destination to a directory path (not an existing file).",
    ],
    "artifact_uploader_command_invalid": [
        "Set --uploader-command to a valid argv-style template using supported placeholders such as {artifact_path} and {destination}.",
    ],
    "artifact_uploader_failed": [
        "Inspect uploader stderr/stdout and exit code, then rerun after fixing the external uploader command or destination permissions.",
    ],
    "artifact_uploader_timeout": [
        "Increase --uploader-timeout-seconds or fix the remote uploader path so the command completes within the expected time.",
    ],
    "artifact_http_upload_failed": [
        "Inspect the HTTP status/body and destination URL, then rerun after fixing remote auth, permissions, or presigned URL configuration.",
    ],
    "artifact_http_upload_timeout": [
        "Increase --uploader-timeout-seconds or fix the remote upload endpoint so the HTTP upload completes within the expected time.",
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
        "Grounding validation failed; retry with broader query/top-k or adjust evidence thresholds explicitly.",
    ],
    "search_stream_contract_conflict": [
        "Disable --stream when requesting evidence trace/grounding validation payloads.",
    ],
}
INSPECT_FAILURE_REMEDIATION: Dict[str, List[str]] = {
    "decode_error": [
        "File contents could not be decoded as UTF-8; convert the file encoding or exclude it from inspect scope.",
        "Rerun `gloggur inspect <path> --json --force` after normalizing file encoding.",
    ],
    "read_error": [
        "File could not be read from disk; verify file permissions/path availability and rerun inspect.",
    ],
    "parser_unavailable": [
        "No parser is registered for this file extension; inspect currently supports configured language extensions only.",
    ],
    "parse_error": [
        "Parser failed on file contents; inspect syntax validity and parser compatibility, then rerun inspect.",
    ],
}
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
    "tool_version_changed_override": [
        "Override is active: verify retrieval correctness in this runtime and schedule a full reindex when possible.",
        "Disable override and rerun `gloggur index . --json` before normal operations.",
    ],
    "cache_corruption_recovered": [
        "Rebuild the index after corruption recovery to restore full symbol coverage.",
    ],
    "cache_schema_rebuilt": [
        "Run a full `gloggur index . --json` after schema rebuild before search operations.",
    ],
}


class CLIContractError(click.ClickException):
    """Click exception with stable machine-readable failure code/guidance payload."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        remediation: Optional[List[str]] = None,
    ) -> None:
        """Capture a stable machine-readable code and optional remediation override."""
        super().__init__(message)
        self.error_code = error_code
        self.remediation = remediation

    def to_payload(self) -> Dict[str, object]:
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
        """Normalize CLI/runtime failures into deterministic non-zero command exits."""
        as_json = _resolve_as_json(kwargs)
        try:
            return callback(*args, **kwargs)
        except CLIContractError as exc:
            if as_json:
                click.echo(
                    f"CLI contract error [{exc.error_code}]: {exc.message}",
                    err=True,
                )
                _emit(exc.to_payload(), as_json=True)
                raise click.exceptions.Exit(exc.exit_code) from exc
            raise
        except click.ClickException as exc:
            if as_json:
                error_code = "cli_usage_error"
                guidance = CLI_FAILURE_REMEDIATION.get(
                    error_code,
                    [DEFAULT_CLI_FAILURE_REMEDIATION],
                )
                click.echo(f"CLI usage error [{error_code}]: {exc.message}", err=True)
                _emit(
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
                    },
                    as_json=True,
                )
                raise click.exceptions.Exit(exc.exit_code) from exc
            raise
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
    allow_tool_version_drift: bool = False,
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
    tool_version_drift_detected = tool_version_reason is not None
    tool_version_override_applied = allow_tool_version_drift and tool_version_drift_detected
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
        "tool_version_drift_detected": tool_version_drift_detected,
        "allow_tool_version_drift": allow_tool_version_drift,
        "tool_version_drift_override_applied": tool_version_override_applied,
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


def _watch_starting_state_payload(*, watch_path: str, pid: int) -> Dict[str, object]:
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


def _build_status_payload(
    config: GloggurConfig,
    cache: CacheManager,
    *,
    allow_tool_version_drift: bool = False,
) -> Dict[str, object]:
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
    effective_tool_version_reason = tool_version_reason
    if allow_tool_version_drift and tool_version_reason is not None:
        effective_tool_version_reason = None
    reindex_reason = metadata_reason or profile_reason or effective_tool_version_reason
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
        allow_tool_version_drift=allow_tool_version_drift,
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


def _create_status_payload(
    config: GloggurConfig,
    *,
    allow_tool_version_drift: bool = False,
) -> Dict[str, object]:
    """Create cache manager and build status payload."""
    cache = _create_cache_manager(config.cache_dir)
    return _build_status_payload(
        config,
        cache,
        allow_tool_version_drift=allow_tool_version_drift,
    )


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


def _sha256_file(path: str) -> str:
    """Return SHA256 digest for a file using chunked reads."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read artifact source file",
            path=path,
        ) from exc
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    """Return SHA256 digest for bytes payloads."""
    return hashlib.sha256(payload).hexdigest()


def _artifact_rel_path(source_dir: str, file_path: str) -> str:
    """Return deterministic POSIX-style relative path under source_dir."""
    relative = os.path.relpath(file_path, source_dir)
    return relative.replace(os.sep, "/")


def _collect_artifact_file_entries(source_dir: str) -> List[Dict[str, object]]:
    """Collect deterministic file metadata for artifact manifest generation."""
    entries: List[Dict[str, object]] = []
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
    file_entries: List[Dict[str, object]],
) -> Dict[str, object]:
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
) -> List[str]:
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
    argv: List[str] = []
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
) -> Dict[str, object]:
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
    payload: Dict[str, object] = {
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
) -> Dict[str, object]:
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
        detail = (
            f"HTTP upload failed with status {exc.code} for destination {destination}"
        )
        if response_body:
            body_text = response_body.decode("utf8", errors="replace").strip()
            if body_text:
                detail = f"{detail}; body={body_text}"
        raise CLIContractError(
            detail,
            error_code="artifact_http_upload_failed",
        ) from exc
    except (urllib_error.URLError, socket.timeout, TimeoutError) as exc:
        raise CLIContractError(
            (
                "HTTP upload timed out or could not connect after "
                f"{timeout_seconds:.3f}s for destination {destination}: {exc}"
            ),
            error_code="artifact_http_upload_timeout",
        ) from exc

    payload: Dict[str, object] = {
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
    file_entries: List[Dict[str, object]],
) -> None:
    """Create a deterministic tar.gz artifact containing cache files and manifest."""
    try:
        with open(artifact_path, "wb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
                with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for entry in file_entries:
                        rel_path = str(entry["path"])
                        source_path = os.path.join(source_dir, rel_path.replace("/", os.sep))
                        tar_info = tar.gettarinfo(
                            source_path,
                            arcname=f"cache/{rel_path}",
                        )
                        tar_info.uid = 0
                        tar_info.gid = 0
                        tar_info.uname = ""
                        tar_info.gname = ""
                        tar_info.mtime = 0
                        with open(source_path, "rb") as source_handle:
                            tar.addfile(tar_info, source_handle)

                    manifest_info = tarfile.TarInfo(name="manifest.json")
                    manifest_info.size = len(manifest_bytes)
                    manifest_info.mode = 0o644
                    manifest_info.uid = 0
                    manifest_info.gid = 0
                    manifest_info.uname = ""
                    manifest_info.gname = ""
                    manifest_info.mtime = 0
                    tar.addfile(manifest_info, io.BytesIO(manifest_bytes))
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
) -> Dict[str, object]:
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
            if not isinstance(manifest_files_total, int) or not isinstance(manifest_bytes_total, int):
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
) -> Dict[str, object]:
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


def _build_failure_contract(
    failed_reasons: Dict[str, int],
    *,
    remediation_by_reason: Dict[str, List[str]],
    default_remediation: str,
) -> Dict[str, object]:
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
    payload: Dict[str, object],
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


def _build_index_failure_contract(failed_reasons: Dict[str, int]) -> Dict[str, object]:
    """Build deterministic machine-readable failure codes and remediation for index payloads."""
    return _build_failure_contract(
        failed_reasons,
        remediation_by_reason=FAILURE_REMEDIATION,
        default_remediation=DEFAULT_INDEX_FAILURE_REMEDIATION,
    )


def _build_inspect_failure_contract(failed_reasons: Dict[str, int]) -> Dict[str, object]:
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
                """Render one in-place scan progress update for interactive index runs."""
                _ = status
                click.echo(f"\rScanning: {done}/{total} files    ", nl=False, err=True)

            indexer._scan_callback = _scan

            if embedding is not None:
                def _progress(done: int, total: int) -> None:
                    """Render one in-place embedding progress update for interactive runs."""
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
            if as_json and result.failed > 0 and not allow_partial:
                _attach_primary_error_from_failure_contract(
                    payload,
                    error_type="index_failure",
                    detail="Indexing completed with file-level failures.",
                    probable_cause=(
                        "One or more files failed during indexing; inspect failed_reasons "
                        "and failed_samples for the per-file causes."
                    ),
                    default_remediation=DEFAULT_INDEX_FAILURE_REMEDIATION,
                )
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
        if as_json and failed > 0 and not allow_partial:
            _attach_primary_error_from_failure_contract(
                result,
                error_type="index_failure",
                detail="Indexing completed with file-level failures.",
                probable_cause=(
                    "One or more file, cleanup, or vector-consistency steps failed; inspect "
                    "failed_reasons and failed_samples for the concrete cause."
                ),
                default_remediation=DEFAULT_INDEX_FAILURE_REMEDIATION,
            )
        _emit(result, as_json)
        if failed > 0 and not allow_partial:
            raise click.exceptions.Exit(1)


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


def _resolve_allow_tool_version_drift(
    *,
    cli_flag_enabled: bool,
) -> bool:
    """Resolve tool-version drift override from CLI flag + env with strict validation."""
    env_values = GloggurConfig._load_dotenv()
    env_values.update(os.environ)
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
            "GLOGGUR_ALLOW_TOOL_VERSION_DRIFT must be one of: "
            "1, true, yes, on, 0, false, no, off.",
            error_code="allow_tool_version_drift_env_invalid",
        )
    return cli_flag_enabled or env_enabled


def _extract_similarity_scores(results: object) -> List[float]:
    """Extract bounded similarity scores or fail loudly on malformed payloads."""
    if not isinstance(results, list):
        raise ValueError("search payload 'results' must be a list")
    scores: List[float] = []
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"search result at index {idx} is not an object")
        raw_score = item.get("similarity_score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"search result at index {idx} has non-numeric similarity_score") from exc
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


def _validate_search_payload(payload: object) -> Dict[str, object]:
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
    filters: Dict[str, str],
    initial_top_k: int,
    confidence_threshold: float,
    max_requery_attempts: int,
    disable_bounded_requery: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run search with optional bounded retry and deterministic confidence telemetry."""
    result = _validate_search_payload(searcher.search(query, filters=filters, top_k=initial_top_k))
    initial_confidence = _compute_retrieval_confidence(result.get("results"))
    final_confidence = initial_confidence
    final_top_k = initial_top_k
    retry_performed = False
    retry_attempts = 0
    retry_strategy: Optional[str] = None

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
        result = _validate_search_payload(searcher.search(query, filters=filters, top_k=final_top_k))
        final_confidence = _compute_retrieval_confidence(result.get("results"))

    confidence_payload: Dict[str, object] = {
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
@click.option("--kind", type=str, default=None)
@click.option("--file", "file_path", type=str, default=None)
@click.option("--top-k", type=int, default=10)
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
    config_path: Optional[str],
    as_json: bool,
    kind: Optional[str],
    file_path: Optional[str],
    top_k: int,
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
    """Search indexed symbols with optional filters."""
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
    if stream and (with_evidence_trace or validate_grounding or fail_on_ungrounded):
        raise CLIContractError(
            "--stream cannot be combined with evidence trace or grounding validation options",
            error_code="search_stream_contract_conflict",
        )
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
    effective_tool_version_reason = tool_version_reason
    if allow_tool_version_drift and tool_version_reason is not None:
        effective_tool_version_reason = None
    reindex_reason = metadata_reason or profile_reason or effective_tool_version_reason
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
        allow_tool_version_drift=allow_tool_version_drift,
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
                "confidence_threshold": confidence_threshold,
                "initial_confidence": None,
                "final_confidence": None,
                "retry_performed": False,
                "retry_attempts": 0,
                "max_requery_attempts": max_requery_attempts,
                "retry_strategy": None,
                "retry_enabled": not disable_bounded_requery and max_requery_attempts > 0,
                "initial_top_k": top_k,
                "final_top_k": top_k,
                "low_confidence": None,
                "grounding_validation_enabled": validate_grounding,
                "grounding_validation_passed": None,
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
    try:
        result, confidence_payload = _search_with_bounded_retry(
            searcher=searcher,
            query=query,
            filters=filters,
            initial_top_k=top_k,
            confidence_threshold=confidence_threshold,
            max_requery_attempts=max_requery_attempts,
            disable_bounded_requery=disable_bounded_requery,
        )
    except ValueError as exc:
        raise CLIContractError(
            f"search response contract invalid: {exc}",
            error_code="search_result_payload_invalid",
        ) from exc
    metadata_payload = result.get("metadata")
    if isinstance(metadata_payload, dict):
        metadata_payload.setdefault("needs_reindex", False)
        metadata_payload.setdefault("reindex_reason", None)
        metadata_payload.update(confidence_payload)
        metadata_payload.update(resume_contract)
    else:
        raise CLIContractError(
            "search response contract invalid: metadata must be an object",
            error_code="search_result_payload_invalid",
        )
    evidence_trace_payload: Optional[List[Dict[str, object]]] = None
    validation_payload: Optional[Dict[str, object]] = None
    if with_evidence_trace or validate_grounding:
        try:
            evidence_trace_payload = build_evidence_trace(result.get("results"))
        except ValueError as exc:
            raise CLIContractError(
                f"search evidence trace invalid: {exc}",
                error_code="search_evidence_trace_invalid",
            ) from exc
    if validate_grounding:
        if evidence_trace_payload is None:
            raise CLIContractError(
                "search evidence trace unavailable for grounding validation",
                error_code="search_evidence_trace_invalid",
            )
        try:
            validation_payload = validate_evidence_trace(
                evidence_trace_payload,
                min_confidence=evidence_min_confidence,
                min_items=evidence_min_items,
            )
        except ValueError as exc:
            raise CLIContractError(
                f"search grounding validation invalid: {exc}",
                error_code="search_evidence_trace_invalid",
            ) from exc
    metadata_payload["grounding_validation_enabled"] = validate_grounding
    metadata_payload["grounding_validation_passed"] = (
        validation_payload.get("passed") if isinstance(validation_payload, dict) else None
    )
    if with_evidence_trace and evidence_trace_payload is not None:
        result["evidence_trace"] = evidence_trace_payload
    if validation_payload is not None:
        result["validation"] = validation_payload
    if validation_payload is not None and not bool(validation_payload.get("passed")) and fail_on_ungrounded:
        error_code = "search_grounding_validation_failed"
        guidance = CLI_FAILURE_REMEDIATION.get(error_code, [DEFAULT_CLI_FAILURE_REMEDIATION])
        result["error"] = {
            "type": "cli_contract_error",
            "code": error_code,
            "detail": "Grounding validation failed for search output.",
            "probable_cause": str(validation_payload.get("reason")),
            "remediation": guidance,
        }
        result["failure_codes"] = [error_code]
        result["failure_guidance"] = {error_code: guidance}
        if as_json:
            _emit(result, as_json=True)
            raise click.exceptions.Exit(1)
        raise CLIContractError(
            "Grounding validation failed for search output.",
            error_code=error_code,
            remediation=guidance,
        )
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


def _load_cached_inspect_reports(
    cache: CacheManager,
    file_path: str,
    *,
    symbol_ids: tuple[str, ...],
) -> List[DocstringAuditReport]:
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
    reports: List[DocstringAuditReport] = []
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
    cached_files_reused = 0
    cached_reports_reused = 0
    cached_warning_reports_reused = 0
    symbol_file_paths: Dict[str, str] = {}
    reports: List[DocstringAuditReport] = []
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
                cached_reports = _load_cached_inspect_reports(
                    cache,
                    file_path,
                    symbol_ids=symbol_ids,
                )
                reports.extend(cached_reports)
                cached_files_reused += 1
                cached_reports_reused += len(cached_reports)
                cached_warning_reports_reused += sum(1 for report in cached_reports if report.warnings)
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
    config_path: Optional[str],
    as_json: bool,
    source_path: Optional[str],
    destination: str,
    artifact_name: Optional[str],
    overwrite: bool,
    uploader_command: Optional[str],
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
    destination_path: Optional[str] = None
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
        uploader_payload: Optional[Dict[str, object]] = None
        http_upload_payload: Optional[Dict[str, object]] = None
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
            archive_path_for_payload: Optional[str] = destination_path
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
    config_path: Optional[str],
    as_json: bool,
    artifact_path: str,
    destination_dir: Optional[str],
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
@click.option(
    "--allow-tool-version-drift",
    is_flag=True,
    default=False,
    help="Allow status resume_ok when only tool-version drift is detected.",
)
@_with_io_failure_handling
def status(config_path: Optional[str], as_json: bool, allow_tool_version_drift: bool) -> None:
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
