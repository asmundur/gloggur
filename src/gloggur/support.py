from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gloggur import __version__ as GLOGGUR_VERSION
from gloggur.archive_utils import (
    ArchiveFileSource,
    create_deterministic_tar_gz,
    sha256_bytes,
    sha256_file,
)
from gloggur.bootstrap_launcher import resolve_bootstrap_status
from gloggur.byte_spans import discover_repo_root
from gloggur.config import GloggurConfig
from gloggur.io_failures import StorageIOError, wrap_io_error
from gloggur.search.router.config import load_search_router_config

SUPPORT_MANIFEST_SCHEMA_VERSION = "1"
SUPPORT_BUNDLE_TYPE = "support"
DEFAULT_TAIL_BYTES = 5 * 1024 * 1024
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SECRET_KEY_RE = re.compile(r"(?:^|_)(?:key|token|secret|password)$", re.IGNORECASE)
_REPO_ROOT_PLACEHOLDER = "<REPO_ROOT>"
_HOME_PLACEHOLDER = "<HOME>"
_REDACTED_PLACEHOLDER = "<REDACTED>"
_CAPTURE_META_PATH = "diagnostics/capture_meta.json"
_ALLOWLISTED_ENV_KEYS = (
    "GLOGGUR_CACHE_DIR",
    "GLOGGUR_EMBEDDING_PROVIDER",
    "GLOGGUR_WATCH_ENABLED",
    "GLOGGUR_WATCH_PATH",
    "GLOGGUR_WATCH_MODE",
    "GLOGGUR_WATCH_STATE_FILE",
    "GLOGGUR_WATCH_LOG_FILE",
    "GLOGGUR_STORAGE_BACKEND",
    "GLOGGUR_RUNTIME_HOST",
    "GLOGGUR_RUN_FROM_CALLER_CWD",
    "BOOTSTRAP_GLOGGUR_LOG_FILE",
    "BOOTSTRAP_GLOGGUR_STATE_FILE",
)


class SupportContractError(RuntimeError):
    """Stable contract failure for support session commands."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SupportRoots:
    """Filesystem roots used by support sessions and bundles."""

    repo_root: Path
    support_root: Path
    sessions_root: Path
    bundles_root: Path


@dataclass(frozen=True)
class SupportCallbacks:
    """Late-bound callbacks supplied by the CLI entrypoints."""

    load_config: Callable[[str | None], tuple[GloggurConfig, str | None]]
    build_status_payload: Callable[[GloggurConfig], dict[str, object]]
    build_watch_status_payload: Callable[[GloggurConfig], dict[str, object]]


def run_support_command(
    *,
    as_json: bool,
    child_args: Sequence[str],
    note: str | None,
    bundle_on_failure: bool,
    destination: str | None,
    callbacks: SupportCallbacks,
) -> tuple[dict[str, object], int]:
    """Run a traced Glöggur child command inside a support session."""
    normalized_child_args = _validate_child_args(child_args)
    config_path = _extract_config_path(normalized_child_args)
    roots = _support_roots(discover_repo_root())
    session_dir = _create_session_dir(roots)
    session_id = session_dir.name
    cwd = Path.cwd().resolve()
    session_payload = {
        "session_id": session_id,
        "mode": "run",
        "repo_root": _sanitize_path(str(roots.repo_root), roots.repo_root),
        "cwd": _sanitize_path(str(cwd), roots.repo_root),
        "child_argv": list(normalized_child_args),
        "child_command": normalized_child_args[0],
        "config_path": _sanitize_optional_string(config_path, roots.repo_root),
        "created_at": _utc_now_iso(),
        "status": "running",
        "note_present": bool(note),
    }
    _write_json(session_dir / "session.json", session_payload)
    if note:
        _append_note(session_dir / "notes.txt", note, roots.repo_root)

    bootstrap_log = session_dir / "logs" / "bootstrap.log"
    bootstrap_state = session_dir / "logs" / "bootstrap.state.json"
    command = _resolve_child_command(roots.repo_root, normalized_child_args)
    env = os.environ.copy()
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(bootstrap_log)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(bootstrap_state)
    env["GLOGGUR_RUN_FROM_CALLER_CWD"] = "1"

    started = time.perf_counter()
    if as_json:
        completed = _run_child_captured(command, cwd=cwd, env=env)
        raw_stdout = completed.stdout
        raw_stderr = completed.stderr
        exit_code = int(completed.returncode)
        _write_sanitized_text_log(
            session_dir / "logs" / "child.stdout.log",
            raw_stdout,
            repo_root=roots.repo_root,
            session_dir=session_dir,
        )
        _write_sanitized_text_log(
            session_dir / "logs" / "child.stderr.log",
            raw_stderr,
            repo_root=roots.repo_root,
            session_dir=session_dir,
        )
    else:
        raw_stdout, raw_stderr, exit_code = _run_child_streaming(
            command,
            cwd=cwd,
            env=env,
            stdout_log_path=session_dir / "logs" / "child.stdout.log",
            stderr_log_path=session_dir / "logs" / "child.stderr.log",
            repo_root=roots.repo_root,
        )
    duration_ms = int((time.perf_counter() - started) * 1000)

    child_payload = _parse_first_json_object(raw_stdout)
    if child_payload is not None:
        _write_json(
            session_dir / "diagnostics" / "child_payload.json",
            _sanitize_object(child_payload, roots.repo_root),
        )
    child_error_code, child_failure_codes = _extract_failure_contract(child_payload)

    _capture_support_diagnostics(
        session_dir=session_dir,
        repo_root=roots.repo_root,
        callbacks=callbacks,
        config_path=config_path,
        support_env=env,
        note=None,
    )
    _snapshot_optional_runtime_files(
        session_dir=session_dir,
        repo_root=roots.repo_root,
        config_path=config_path,
        callbacks=callbacks,
    )

    bundle_payload: dict[str, object] | None = None
    bundle_error: dict[str, object] | None = None
    if exit_code != 0 and bundle_on_failure:
        try:
            bundle_payload = collect_support_bundle(
                session_id=session_id,
                note=None,
                destination=destination,
                include_cache=False,
                overwrite=False,
                callbacks=callbacks,
                config_path=config_path,
            )
        except (SupportContractError, StorageIOError) as exc:
            bundle_error = _bundle_error_payload(exc)

    session_payload.update(
        {
            "status": "completed" if exit_code == 0 else "failed",
            "completed_at": _utc_now_iso(),
            "duration_ms": duration_ms,
            "exit_code": exit_code,
            "child_error_code": child_error_code,
            "child_failure_codes": child_failure_codes,
            "bundle_created": bundle_payload is not None,
            "bundle_path": bundle_payload.get("bundle_path") if bundle_payload else None,
            "bundle_error": bundle_error,
        }
    )
    _write_json(session_dir / "session.json", session_payload)

    result = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "child_argv": list(normalized_child_args),
        "child_exit_code": exit_code,
        "child_error_code": child_error_code,
        "failure_codes": child_failure_codes,
        "bundle_created": bundle_payload is not None,
        "bundle_path": bundle_payload.get("bundle_path") if bundle_payload else None,
        "bundle_manifest_sha256": bundle_payload.get("manifest_sha256") if bundle_payload else None,
        "bundle_error": bundle_error,
    }
    return result, exit_code


def collect_support_bundle(
    *,
    session_id: str | None,
    note: str | None,
    destination: str | None,
    include_cache: bool,
    allow_sensitive_data: bool = False,
    overwrite: bool,
    callbacks: SupportCallbacks,
    config_path: str | None = None,
) -> dict[str, object]:
    """Collect diagnostics for a session and package a deterministic support bundle."""
    roots = _support_roots(discover_repo_root())
    if session_id is None:
        session_dir = _create_session_dir(roots)
        session_payload = {
            "session_id": session_dir.name,
            "mode": "manual_snapshot",
            "repo_root": _sanitize_path(str(roots.repo_root), roots.repo_root),
            "cwd": _sanitize_path(str(Path.cwd().resolve()), roots.repo_root),
            "created_at": _utc_now_iso(),
            "status": "collecting",
            "child_argv": [],
            "child_command": None,
            "config_path": _sanitize_optional_string(config_path, roots.repo_root),
            "note_present": bool(note),
        }
        _write_json(session_dir / "session.json", session_payload)
    else:
        session_dir = _resolve_existing_session_dir(roots, session_id)
        session_payload = _read_session_payload(session_dir)
    if note:
        _append_note(session_dir / "notes.txt", note, roots.repo_root)
        session_payload["note_present"] = True

    _capture_support_diagnostics(
        session_dir=session_dir,
        repo_root=roots.repo_root,
        callbacks=callbacks,
        config_path=config_path,
        support_env=None,
        note=None,
    )
    _snapshot_optional_runtime_files(
        session_dir=session_dir,
        repo_root=roots.repo_root,
        config_path=config_path,
        callbacks=callbacks,
    )

    bundle_path = _resolve_bundle_destination(
        roots=roots,
        session_id=session_dir.name,
        destination=destination,
        overwrite=overwrite,
    )
    config: GloggurConfig | None = None
    try:
        config, _ = callbacks.load_config(config_path)
    except (StorageIOError, OSError, ValueError, TypeError):
        config = None
    cache_dir = config.cache_dir if config is not None else str(roots.repo_root / ".gloggur-cache")
    runtime_artifacts = _inventory_runtime_artifacts(roots.repo_root, cache_dir)
    included_artifacts = runtime_artifacts if include_cache else []
    excluded_artifacts = [] if include_cache else runtime_artifacts
    _write_bundle(
        repo_root=roots.repo_root,
        session_dir=session_dir,
        bundle_path=bundle_path,
        include_cache=include_cache,
        allow_sensitive_data=allow_sensitive_data,
        included_artifacts=included_artifacts,
        excluded_artifacts=excluded_artifacts,
    )
    manifest_payload = _read_bundle_manifest(bundle_path)
    public_included_artifacts = _public_artifact_inventory(included_artifacts)
    public_excluded_artifacts = _public_artifact_inventory(excluded_artifacts)

    session_payload.update(
        {
            "status": "collected",
            "collected_at": _utc_now_iso(),
            "bundle_path": str(bundle_path),
            "redaction_mode": "sanitized",
            "include_cache": include_cache,
            "bundle_sensitivity": "sensitive" if include_cache else "sanitized",
            "includes_cache_artifacts": include_cache,
            "sensitive_data_acknowledged": allow_sensitive_data,
            "excluded_artifacts": public_excluded_artifacts,
            "included_artifacts": public_included_artifacts,
        }
    )
    _write_json(session_dir / "session.json", session_payload)

    return {
        "collected": True,
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "bundle_path": str(bundle_path),
        "bundle_uri": bundle_path.resolve().as_uri(),
        "include_cache": include_cache,
        "bundle_sensitivity": "sensitive" if include_cache else "sanitized",
        "includes_cache_artifacts": include_cache,
        "sensitive_data_acknowledged": allow_sensitive_data,
        "manifest_sha256": manifest_payload["manifest_sha256"],
        "archive_sha256": manifest_payload["archive_sha256"],
        "archive_bytes": manifest_payload["archive_bytes"],
        "excluded_artifacts": public_excluded_artifacts,
        "included_artifacts": public_included_artifacts,
    }


def _support_roots(repo_root: Path) -> SupportRoots:
    support_root = repo_root / ".gloggur" / "support"
    return SupportRoots(
        repo_root=repo_root,
        support_root=support_root,
        sessions_root=support_root / "sessions",
        bundles_root=support_root / "bundles",
    )


def _validate_child_args(child_args: Sequence[str]) -> list[str]:
    normalized = [str(arg) for arg in child_args if str(arg)]
    if not normalized:
        raise SupportContractError(
            "Support run requires a Glöggur child command after `--`.",
            code="support_command_invalid",
        )
    first = normalized[0]
    if first in {"gloggur", "support"}:
        raise SupportContractError(
            "Support run only accepts Glöggur subcommand argv.",
            code="support_command_invalid",
        )
    if first.startswith("-") or "/" in first or "\\" in first:
        raise SupportContractError(
            "Support run only accepts Glöggur subcommand argv, not executable paths.",
            code="support_command_invalid",
        )
    return normalized


def _extract_config_path(child_args: Sequence[str]) -> str | None:
    for index, arg in enumerate(child_args):
        if arg == "--config" and index + 1 < len(child_args):
            return child_args[index + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _resolve_child_command(repo_root: Path, child_args: Sequence[str]) -> list[str]:
    wrapper = repo_root / "scripts" / "gloggur"
    if wrapper.exists():
        return [str(wrapper), *child_args]
    return [sys.executable, "-m", "gloggur.cli.main", *child_args]


def _create_session_dir(roots: SupportRoots) -> Path:
    base_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    session_id = base_id
    candidate = roots.sessions_root / session_id
    suffix = 1
    while candidate.exists():
        session_id = f"{base_id}-{suffix}"
        candidate = roots.sessions_root / session_id
        suffix += 1
    for path in (candidate, candidate / "logs", candidate / "diagnostics"):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise wrap_io_error(
                exc, operation="create support session directory", path=str(path)
            ) from exc
    return candidate


def _resolve_existing_session_dir(roots: SupportRoots, session_id: str) -> Path:
    normalized = session_id.strip()
    if not normalized or not _SESSION_ID_RE.fullmatch(normalized):
        raise SupportContractError(
            f"Support session id is invalid: {session_id!r}",
            code="support_session_invalid",
        )
    session_dir = (roots.sessions_root / normalized).resolve()
    try:
        within_root = os.path.commonpath(
            [str(roots.sessions_root.resolve()), str(session_dir)]
        ) == str(roots.sessions_root.resolve())
    except ValueError as exc:
        raise SupportContractError(
            f"Support session id escapes the support root: {session_id!r}",
            code="support_session_invalid",
        ) from exc
    if not within_root:
        raise SupportContractError(
            f"Support session id escapes the support root: {session_id!r}",
            code="support_session_invalid",
        )
    if not session_dir.exists():
        raise SupportContractError(
            f"Support session does not exist: {normalized}",
            code="support_session_missing",
        )
    if not (session_dir / "session.json").exists():
        raise SupportContractError(
            f"Support session is missing session.json: {normalized}",
            code="support_session_invalid",
        )
    return session_dir


def _capture_support_diagnostics(
    *,
    session_dir: Path,
    repo_root: Path,
    callbacks: SupportCallbacks,
    config_path: str | None,
    support_env: dict[str, str] | None,
    note: str | None,
) -> None:
    if note:
        _append_note(session_dir / "notes.txt", note, repo_root)

    config_payload: dict[str, object]
    config: GloggurConfig | None = None
    resolved_config_path: str | None = None
    try:
        config, resolved_config_path = callbacks.load_config(config_path)
        config_payload = {
            "captured": True,
            "config_path": _sanitize_optional_string(resolved_config_path, repo_root),
            "config": _sanitize_object(asdict(config), repo_root),
            "search_router": _sanitize_object(
                asdict(load_search_router_config(repo_root)), repo_root
            ),
            "env_summary": _sanitize_object(_collect_env_summary(support_env), repo_root),
        }
    except (StorageIOError, OSError, ValueError, TypeError) as exc:
        config_payload = _diagnostic_error_payload(exc, repo_root)
    _write_json(session_dir / "diagnostics" / "config.redacted.json", config_payload)

    bootstrap_payload = {
        "captured": True,
        "bootstrap_status": _sanitize_object(
            asdict(resolve_bootstrap_status(env=support_env or os.environ)),
            repo_root,
        ),
    }
    _write_json(session_dir / "diagnostics" / "bootstrap_status.json", bootstrap_payload)

    status_payload = _capture_callback_payload(
        callback=((lambda: callbacks.build_status_payload(config)) if config is not None else None),
        repo_root=repo_root,
    )
    _write_json(session_dir / "diagnostics" / "status.json", status_payload)

    watch_payload = _capture_callback_payload(
        callback=(
            (lambda: callbacks.build_watch_status_payload(config)) if config is not None else None
        ),
        repo_root=repo_root,
    )
    _write_json(session_dir / "diagnostics" / "watch_status.json", watch_payload)


def _capture_callback_payload(
    *,
    callback: Callable[[], dict[str, object]] | None,
    repo_root: Path,
) -> dict[str, object]:
    if callback is None:
        return {"captured": False, "reason": "config_unavailable"}
    try:
        return {"captured": True, "payload": _sanitize_object(callback(), repo_root)}
    except (StorageIOError, OSError, ValueError, TypeError) as exc:
        return _diagnostic_error_payload(exc, repo_root)


def _snapshot_optional_runtime_files(
    *,
    session_dir: Path,
    repo_root: Path,
    config_path: str | None,
    callbacks: SupportCallbacks,
) -> None:
    config: GloggurConfig | None = None
    try:
        config, _ = callbacks.load_config(config_path)
    except (StorageIOError, OSError, ValueError, TypeError):
        config = None
    router_config = load_search_router_config(repo_root)
    log_sources: list[tuple[Path, Path, bool]] = []
    if config is not None:
        log_sources.extend(
            [
                (Path(config.watch_log_file), session_dir / "logs" / "watch.log", True),
                (Path(config.watch_state_file), session_dir / "logs" / "watch_state.json", True),
                (Path(config.watch_pid_file), session_dir / "logs" / "watch.pid", True),
            ]
        )
    router_log_path = repo_root / router_config.log_path
    log_sources.append((router_log_path, session_dir / "logs" / "search_router.jsonl", True))
    bootstrap_log_path = os.environ.get("BOOTSTRAP_GLOGGUR_LOG_FILE")
    if bootstrap_log_path:
        log_sources.append((Path(bootstrap_log_path), session_dir / "logs" / "bootstrap.log", True))
    bootstrap_state_path = os.environ.get("BOOTSTRAP_GLOGGUR_STATE_FILE")
    if bootstrap_state_path:
        log_sources.append(
            (Path(bootstrap_state_path), session_dir / "logs" / "bootstrap.state.json", True)
        )

    for source_path, destination_path, sanitize in log_sources:
        if destination_path.exists():
            continue
        if sanitize:
            _snapshot_optional_text_file(
                source_path=source_path,
                destination_path=destination_path,
                repo_root=repo_root,
                session_dir=session_dir,
            )
        else:
            _snapshot_optional_file(source_path=source_path, destination_path=destination_path)


def _snapshot_optional_text_file(
    *,
    source_path: Path,
    destination_path: Path,
    repo_root: Path,
    session_dir: Path,
) -> None:
    if not source_path.exists() or not source_path.is_file():
        return
    try:
        raw_bytes = source_path.read_bytes()
    except OSError as exc:
        raise wrap_io_error(
            exc, operation="read support log source", path=str(source_path)
        ) from exc
    truncated = len(raw_bytes) > DEFAULT_TAIL_BYTES
    payload = raw_bytes[-DEFAULT_TAIL_BYTES:] if truncated else raw_bytes
    text = payload.decode("utf8", errors="replace")
    sanitized = _sanitize_text(text, repo_root)
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(sanitized, encoding="utf8")
    except OSError as exc:
        raise wrap_io_error(
            exc, operation="write support log snapshot", path=str(destination_path)
        ) from exc
    _record_capture_metadata(
        session_dir=session_dir,
        session_rel_path=destination_path.relative_to(session_dir).as_posix(),
        source_path=source_path,
        source_bytes=len(raw_bytes),
        archived_bytes=len(sanitized.encode("utf8")),
        truncated=truncated,
    )


def _snapshot_optional_file(*, source_path: Path, destination_path: Path) -> None:
    if not source_path.exists() or not source_path.is_file():
        return
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)
    except OSError as exc:
        raise wrap_io_error(
            exc, operation="copy support runtime file", path=str(destination_path)
        ) from exc


def _collect_env_summary(env: dict[str, str] | None) -> dict[str, str]:
    source = dict(os.environ if env is None else env)
    summary: dict[str, str] = {}
    for key in _ALLOWLISTED_ENV_KEYS:
        value = source.get(key)
        if value is None or value == "":
            continue
        summary[key] = value
    return summary


def _write_sanitized_text_log(
    log_path: Path,
    raw_text: str,
    *,
    repo_root: Path,
    session_dir: Path,
) -> None:
    payload = raw_text.encode("utf8", errors="replace")
    truncated = len(payload) > DEFAULT_TAIL_BYTES
    truncated_payload = payload[-DEFAULT_TAIL_BYTES:] if truncated else payload
    sanitized = _sanitize_text(truncated_payload.decode("utf8", errors="replace"), repo_root)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(sanitized, encoding="utf8")
    except OSError as exc:
        raise wrap_io_error(exc, operation="write support child log", path=str(log_path)) from exc
    _record_capture_metadata(
        session_dir=session_dir,
        session_rel_path=log_path.relative_to(session_dir).as_posix(),
        source_path=None,
        source_bytes=len(payload),
        archived_bytes=len(sanitized.encode("utf8")),
        truncated=truncated,
    )


def _run_child_captured(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise wrap_io_error(
            exc, operation="spawn support child command", path=str(command[0])
        ) from exc


def _run_child_streaming(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_log_path: Path,
    stderr_log_path: Path,
    repo_root: Path,
) -> tuple[str, str, int]:
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise wrap_io_error(
            exc, operation="spawn support child command", path=str(command[0])
        ) from exc

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_log_path.open("w", encoding="utf8") as stdout_log,
        stderr_log_path.open(
            "w",
            encoding="utf8",
        ) as stderr_log,
    ):
        threads = [
            threading.Thread(
                target=_relay_stream,
                args=(process.stdout, sys.stdout, stdout_parts, stdout_log, repo_root),
                daemon=True,
            ),
            threading.Thread(
                target=_relay_stream,
                args=(process.stderr, sys.stderr, stderr_parts, stderr_log, repo_root),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        exit_code = int(process.wait())
        for thread in threads:
            thread.join()
    raw_stdout = "".join(stdout_parts)
    raw_stderr = "".join(stderr_parts)
    _record_capture_metadata(
        session_dir=stdout_log_path.parent.parent,
        session_rel_path=stdout_log_path.relative_to(stdout_log_path.parent.parent).as_posix(),
        source_path=None,
        source_bytes=len(raw_stdout.encode("utf8", errors="replace")),
        archived_bytes=stdout_log_path.stat().st_size,
        truncated=False,
    )
    _record_capture_metadata(
        session_dir=stderr_log_path.parent.parent,
        session_rel_path=stderr_log_path.relative_to(stderr_log_path.parent.parent).as_posix(),
        source_path=None,
        source_bytes=len(raw_stderr.encode("utf8", errors="replace")),
        archived_bytes=stderr_log_path.stat().st_size,
        truncated=False,
    )
    return raw_stdout, raw_stderr, exit_code


def _relay_stream(
    stream: Any,
    terminal: Any,
    buffer: list[str],
    log_handle: Any,
    repo_root: Path,
) -> None:
    if stream is None:
        return
    try:
        for chunk in iter(stream.readline, ""):
            if not chunk:
                break
            buffer.append(chunk)
            terminal.write(chunk)
            terminal.flush()
            log_handle.write(_sanitize_text(chunk, repo_root))
            log_handle.flush()
        remainder = stream.read()
        if remainder:
            buffer.append(remainder)
            terminal.write(remainder)
            terminal.flush()
            log_handle.write(_sanitize_text(remainder, repo_root))
            log_handle.flush()
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _write_bundle(
    *,
    repo_root: Path,
    session_dir: Path,
    bundle_path: Path,
    include_cache: bool,
    allow_sensitive_data: bool,
    included_artifacts: list[dict[str, object]],
    excluded_artifacts: list[dict[str, object]],
) -> None:
    support_prefix = f"support/{session_dir.name}"
    try:
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f"support-bundle-{session_dir.name}-",
                dir=str(bundle_path.parent),
            )
        )
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="create support bundle staging directory",
            path=str(bundle_path.parent),
        ) from exc
    try:
        shutil.copytree(session_dir, staging_dir / support_prefix)
        extra_sources: list[ArchiveFileSource] = []
        if include_cache:
            for item in included_artifacts:
                rel_path = str(item["path"])
                source_path = Path(str(item["source_path"]))
                extra_sources.append(
                    ArchiveFileSource(
                        source_path=source_path,
                        archive_path=f"{support_prefix}/artifacts/{rel_path}",
                    )
                )
        staged_sources = [
            ArchiveFileSource(
                source_path=path,
                archive_path=(
                    f"{support_prefix}/"
                    f"{path.relative_to(staging_dir / support_prefix).as_posix()}"
                ),
            )
            for path in sorted((staging_dir / support_prefix).rglob("*"))
            if path.is_file()
        ]
        all_sources = [*staged_sources, *extra_sources]
        manifest = _build_bundle_manifest(
            session_dir=session_dir,
            archive_sources=all_sources,
            include_cache=include_cache,
            allow_sensitive_data=allow_sensitive_data,
            included_artifacts=included_artifacts,
            excluded_artifacts=excluded_artifacts,
            repo_root=repo_root,
        )
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf8") + b"\n"
        create_deterministic_tar_gz(
            bundle_path,
            file_sources=all_sources,
            extra_files=(("manifest.json", manifest_bytes),),
        )
    except OSError as exc:
        raise wrap_io_error(
            exc, operation="create support bundle archive", path=str(bundle_path)
        ) from exc
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _build_bundle_manifest(
    *,
    session_dir: Path,
    archive_sources: Sequence[ArchiveFileSource],
    include_cache: bool,
    allow_sensitive_data: bool,
    included_artifacts: list[dict[str, object]],
    excluded_artifacts: list[dict[str, object]],
    repo_root: Path,
) -> dict[str, object]:
    capture_meta = _load_capture_metadata(session_dir)
    file_entries: list[dict[str, object]] = []
    archive_prefix = f"support/{session_dir.name}/"
    for source in sorted(archive_sources, key=lambda item: item.archive_path):
        try:
            size_bytes = source.source_path.stat().st_size
        except OSError as exc:
            raise wrap_io_error(
                exc,
                operation="read support bundle file metadata",
                path=str(source.source_path),
            ) from exc
        session_rel_path = None
        if source.archive_path.startswith(archive_prefix):
            session_rel_path = source.archive_path[len(archive_prefix) :]
        meta = capture_meta.get(session_rel_path or "", {})
        file_entries.append(
            {
                "path": source.archive_path,
                "bytes": size_bytes,
                "sha256": _sha256_file(source.source_path),
                "truncated": bool(meta.get("truncated", False)),
                "source_bytes": meta.get("source_bytes", size_bytes),
                "archived_bytes": meta.get("archived_bytes", size_bytes),
            }
        )
    session_payload = _read_session_payload(session_dir)
    manifest = {
        "manifest_schema_version": SUPPORT_MANIFEST_SCHEMA_VERSION,
        "bundle_type": SUPPORT_BUNDLE_TYPE,
        "created_at": _utc_now_iso(),
        "tool_version": GLOGGUR_VERSION,
        "session": _sanitize_object(session_payload, repo_root),
        "redaction_mode": "sanitized",
        "include_cache": include_cache,
        "bundle_sensitivity": "sensitive" if include_cache else "sanitized",
        "includes_cache_artifacts": include_cache,
        "sensitive_data_acknowledged": allow_sensitive_data,
        "included_artifacts": _public_artifact_inventory(included_artifacts),
        "excluded_artifacts": _public_artifact_inventory(excluded_artifacts),
        "files_total": len(file_entries),
        "bytes_total": sum(int(entry["bytes"]) for entry in file_entries),
        "files": file_entries,
    }
    return manifest


def _resolve_bundle_destination(
    *,
    roots: SupportRoots,
    session_id: str,
    destination: str | None,
    overwrite: bool,
) -> Path:
    default_path = roots.bundles_root / f"gloggur-support-{session_id}.tar.gz"
    bundle_path = Path(destination).expanduser() if destination else default_path
    if destination:
        bundle_path = bundle_path.resolve() if bundle_path.exists() else bundle_path.absolute()
        if str(destination).endswith(os.sep) or bundle_path.is_dir():
            bundle_path = bundle_path / default_path.name
    try:
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="create support bundle directory",
            path=str(bundle_path.parent),
        ) from exc
    if bundle_path.exists() and not overwrite:
        raise SupportContractError(
            f"Support bundle destination already exists: {bundle_path}",
            code="support_destination_exists",
        )
    return bundle_path


def _inventory_runtime_artifacts(repo_root: Path, cache_dir: str) -> list[dict[str, object]]:
    artifact_roots = (
        (Path(cache_dir), ".gloggur-cache"),
        (repo_root / ".gloggur" / "index", ".gloggur/index"),
    )
    artifacts: list[dict[str, object]] = []
    for root, archive_root in artifact_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                size_bytes = path.stat().st_size
            except OSError as exc:
                raise wrap_io_error(
                    exc,
                    operation="read support artifact inventory",
                    path=str(path),
                ) from exc
            artifacts.append(
                {
                    "path": f"{archive_root}/{path.relative_to(root).as_posix()}",
                    "bytes": size_bytes,
                    "source_path": str(path),
                }
            )
    return artifacts


def _public_artifact_inventory(items: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "path": item["path"],
            "bytes": item["bytes"],
        }
        for item in items
    ]


def _read_bundle_manifest(bundle_path: Path) -> dict[str, object]:
    archive_bytes = bundle_path.stat().st_size
    archive_sha256 = _sha256_file(bundle_path)
    with tarfile.open(bundle_path, "r:gz") as archive:
        manifest_member = archive.extractfile("manifest.json")
        if manifest_member is None:
            raise SupportContractError(
                "Support bundle archive is missing manifest.json.",
                code="support_session_invalid",
            )
        manifest_bytes = manifest_member.read()
    return {
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "manifest_sha256": sha256_bytes(manifest_bytes),
    }


def _load_capture_metadata(session_dir: Path) -> dict[str, dict[str, object]]:
    meta_path = session_dir / _CAPTURE_META_PATH
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = dict(value)
    return normalized


def _record_capture_metadata(
    *,
    session_dir: Path,
    session_rel_path: str,
    source_path: Path | None,
    source_bytes: int,
    archived_bytes: int,
    truncated: bool,
) -> None:
    payload = _load_capture_metadata(session_dir)
    record: dict[str, object] = {
        "source_bytes": source_bytes,
        "archived_bytes": archived_bytes,
        "truncated": truncated,
    }
    if source_path is not None:
        record["source_path"] = str(source_path)
    payload[session_rel_path] = record
    _write_json(session_dir / _CAPTURE_META_PATH, payload)


def _read_session_payload(session_dir: Path) -> dict[str, object]:
    try:
        payload = json.loads((session_dir / "session.json").read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportContractError(
            f"Support session metadata is invalid: {session_dir.name}",
            code="support_session_invalid",
        ) from exc
    if not isinstance(payload, dict):
        raise SupportContractError(
            f"Support session metadata is invalid: {session_dir.name}",
            code="support_session_invalid",
        )
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise wrap_io_error(
            exc, operation="write support diagnostics file", path=str(path)
        ) from exc


def _append_note(path: Path, note: str, repo_root: Path) -> None:
    sanitized = _sanitize_text(note, repo_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf8") as handle:
            if path.exists() and path.stat().st_size > 0:
                handle.write("\n")
            handle.write(sanitized)
            if not sanitized.endswith("\n"):
                handle.write("\n")
    except OSError as exc:
        raise wrap_io_error(exc, operation="write support note", path=str(path)) from exc


def _sanitize_optional_string(value: str | None, repo_root: Path) -> str | None:
    if value is None:
        return None
    return _sanitize_text(value, repo_root)


def _sanitize_path(value: str, repo_root: Path) -> str:
    return _sanitize_text(value, repo_root)


def _sanitize_text(value: str, repo_root: Path) -> str:
    result = value
    repo_text = str(repo_root.resolve())
    home_text = str(Path.home().resolve())
    if repo_text:
        result = result.replace(repo_text, _REPO_ROOT_PLACEHOLDER)
    if home_text:
        result = result.replace(home_text, _HOME_PLACEHOLDER)
    return result


def _sanitize_object(value: object, repo_root: Path, *, key_name: str | None = None) -> object:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_object(item, repo_root, key_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_object(item, repo_root, key_name=key_name) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_object(item, repo_root, key_name=key_name) for item in value]
    if isinstance(value, str):
        if key_name and _SECRET_KEY_RE.search(key_name):
            return _REDACTED_PLACEHOLDER
        return _sanitize_text(value, repo_root)
    return value


def _parse_first_json_object(raw: str) -> dict[str, object] | None:
    start = raw.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_failure_contract(payload: dict[str, object] | None) -> tuple[str | None, list[str]]:
    if not isinstance(payload, dict):
        return None, []
    child_error_code: str | None = None
    failure_codes: list[str] = []
    compatibility = payload.get("compatibility")
    if isinstance(compatibility, dict):
        nested_error = compatibility.get("error")
        if isinstance(nested_error, dict) and isinstance(nested_error.get("code"), str):
            child_error_code = str(nested_error["code"])
        nested_failure_codes = compatibility.get("failure_codes")
        if isinstance(nested_failure_codes, list):
            failure_codes = [str(code) for code in nested_failure_codes if str(code)]
    if child_error_code is None:
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            child_error_code = str(error["code"])
        elif isinstance(payload.get("error_code"), str):
            child_error_code = str(payload["error_code"])
    if not failure_codes:
        raw_failure_codes = payload.get("failure_codes")
        if isinstance(raw_failure_codes, list):
            failure_codes = [str(code) for code in raw_failure_codes if str(code)]
    return child_error_code, failure_codes


def _diagnostic_error_payload(exc: Exception, repo_root: Path) -> dict[str, object]:
    return {
        "captured": False,
        "error": {
            "type": type(exc).__name__,
            "detail": _sanitize_text(str(exc), repo_root),
        },
    }


def _bundle_error_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, SupportContractError):
        return {"code": exc.code, "detail": str(exc)}
    if isinstance(exc, StorageIOError):
        return {
            "type": "io_failure",
            "category": exc.category,
            "operation": exc.operation,
            "path": exc.path,
            "detail": exc.detail,
        }
    return {"type": type(exc).__name__, "detail": str(exc)}


def _sha256_file(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except OSError as exc:
        raise wrap_io_error(exc, operation="read support bundle file", path=str(path)) from exc


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
