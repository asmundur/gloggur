from __future__ import annotations

import copy
import faulthandler
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from gloggur.byte_spans import discover_repo_root
from gloggur.io_failures import StorageIOError, wrap_io_error

SUPPORT_RUNTIME_CONFIG_PATH = ".gloggur/config.toml"
DEFAULT_LOG_BYTES = 5 * 1024 * 1024
DEFAULT_RECENT_COMMAND_LIMIT = 20
HEARTBEAT_INTERVAL_SECONDS = 2.0
_REPO_ROOT_PLACEHOLDER = "<REPO_ROOT>"
_HOME_PLACEHOLDER = "<HOME>"
_REDACTED_PLACEHOLDER = "<REDACTED>"
_SECRET_KEY_RE = re.compile(r"(?:^|_)(?:key|token|secret|password)$", re.IGNORECASE)
SUPPORT_RUNTIME_DEGRADED_WARNING_CODE = "support_runtime_degraded"

_ACTIVE_TRACE_SESSION: CommandTraceSession | None = None


@dataclass(frozen=True)
class SupportRuntimeConfig:
    enabled: bool = False
    recent_limit: int = DEFAULT_RECENT_COMMAND_LIMIT
    max_log_bytes: int = DEFAULT_LOG_BYTES


@dataclass(frozen=True)
class SupportRuntimePaths:
    repo_root: Path
    support_root: Path
    runtime_root: Path
    active_root: Path
    recent_root: Path


class _TeeTextStream:
    def __init__(self, base: TextIO, session: CommandTraceSession, stream_name: str) -> None:
        self._base = base
        self._session = session
        self._stream_name = stream_name

    def write(self, value: str) -> int:
        written = self._base.write(value)
        self._session.append_output(self._stream_name, value)
        return written

    def flush(self) -> None:
        self._base.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class CommandTraceSession:
    def __init__(
        self,
        *,
        repo_root: Path,
        command_name: str,
        argv: list[str],
        config: SupportRuntimeConfig,
    ) -> None:
        self.repo_root = repo_root
        self.command_name = command_name
        self.argv = list(argv)
        self.config = config
        self.paths = support_runtime_paths(repo_root)
        started_at = _utc_now_iso()
        self.command_id = f"{started_at.replace(':', '').replace('-', '')}-{os.getpid()}"
        self.command_dir = self.paths.active_root / self.command_id
        self.stdout_path = self.command_dir / "stdout.log"
        self.stderr_path = self.command_dir / "stderr.log"
        self.stack_dump_path = self.command_dir / "stackdump.log"
        self.meta_path = self.command_dir / "meta.json"
        self._lock = threading.Lock()
        self._metadata_io_lock = threading.Lock()
        self._capture_enabled = threading.Event()
        self._capture_enabled.set()
        self._degrade_warning_emitted = False
        self._warning_codes: list[str] = []
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._stdout_base: TextIO | None = None
        self._stderr_base: TextIO | None = None
        self._stack_handle: TextIO | None = None
        self._stack_signal = getattr(signal, "SIGUSR1", None)
        self._stack_supported = False
        self._installed_streams = False
        self._metadata: dict[str, object] = {
            "command_id": self.command_id,
            "command_name": self.command_name,
            "argv": _sanitize_object(self.argv, self.repo_root),
            "cwd": _sanitize_text(str(Path.cwd().resolve()), self.repo_root),
            "repo_root": _sanitize_text(str(self.repo_root.resolve()), self.repo_root),
            "pid": os.getpid(),
            "status": "running",
            "started_at": started_at,
            "updated_at": started_at,
            "heartbeat_at": started_at,
            "completed_at": None,
            "exit_code": None,
            "error_code": None,
            "failure_codes": [],
            "stage": None,
            "build_id": None,
            "build_state": None,
            "stdout_log": "stdout.log",
            "stderr_log": "stderr.log",
            "stack_dump": {
                "path": "stackdump.log",
                "supported": False,
                "signal": None,
                "last_requested_at": None,
                "last_captured_at": None,
            },
            "capture_warnings": [],
        }

    def __enter__(self) -> CommandTraceSession:
        global _ACTIVE_TRACE_SESSION
        self._run_capture_operation(
            action="create support runtime directory",
            callback=lambda: self.command_dir.mkdir(parents=True, exist_ok=True),
        )
        self._run_capture_operation(
            action="install support runtime streams",
            callback=self._install_streams,
        )
        self._run_capture_operation(
            action="install support runtime stack dump handler",
            callback=self._install_stack_dump_handler,
        )
        self._write_metadata()
        if self._capture_enabled.is_set():
            self._run_capture_operation(
                action="start support runtime heartbeat thread",
                callback=self._start_heartbeat_thread,
            )
        _ACTIVE_TRACE_SESSION = self
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        global _ACTIVE_TRACE_SESSION
        try:
            self._stop_event.set()
            self._run_capture_operation(
                action="join support runtime heartbeat thread",
                callback=self._join_heartbeat_thread,
                require_enabled=False,
            )
            self._run_capture_operation(
                action="uninstall support runtime streams",
                callback=self._uninstall_streams,
                require_enabled=False,
            )
            self._run_capture_operation(
                action="uninstall support runtime stack dump handler",
                callback=self._uninstall_stack_dump_handler,
                require_enabled=False,
            )
            with self._lock:
                if self._metadata.get("status") == "running":
                    exit_code = int(self._metadata.get("exit_code") or 0)
                    self._metadata["status"] = "completed" if exit_code == 0 else "failed"
                self._metadata["completed_at"] = _utc_now_iso()
                self._metadata["updated_at"] = _utc_now_iso()
            self._run_capture_operation(
                action="enrich support runtime failure payload",
                callback=self._enrich_failure_from_stdout,
            )
            self._write_metadata()
            self._run_capture_operation(
                action="rotate support runtime trace to recent history",
                callback=self._move_to_recent,
                require_enabled=False,
            )
        finally:
            _ACTIVE_TRACE_SESSION = None

    def set_exit_code(self, exit_code: int) -> None:
        with self._lock:
            self._metadata["exit_code"] = int(exit_code)
            self._metadata["status"] = "completed" if int(exit_code) == 0 else "failed"
            self._metadata["updated_at"] = _utc_now_iso()
        self._write_metadata()

    def warning_codes(self) -> list[str]:
        with self._lock:
            return list(self._warning_codes)

    def record_failure(
        self,
        *,
        error_code: str | None,
        failure_codes: list[str] | None = None,
        exit_code: int | None = None,
    ) -> None:
        with self._lock:
            if error_code:
                self._metadata["error_code"] = error_code
            normalized = [str(code) for code in (failure_codes or []) if str(code)]
            if normalized:
                self._metadata["failure_codes"] = normalized
            if exit_code is not None:
                self._metadata["exit_code"] = int(exit_code)
            self._metadata["status"] = "failed"
            self._metadata["updated_at"] = _utc_now_iso()
        self._write_metadata()

    def update_stage(self, stage: str, *, build_id: str | None = None) -> None:
        with self._lock:
            self._metadata["stage"] = str(stage)
            if build_id:
                self._metadata["build_id"] = str(build_id)
            self._metadata["updated_at"] = _utc_now_iso()
            self._metadata["heartbeat_at"] = _utc_now_iso()
        self._write_metadata()

    def update_build_state(self, payload: dict[str, object]) -> None:
        with self._lock:
            self._metadata["build_state"] = _sanitize_object(payload, self.repo_root)
            self._metadata["updated_at"] = _utc_now_iso()
            self._metadata["heartbeat_at"] = _utc_now_iso()
        self._write_metadata()

    def append_output(self, stream_name: str, value: str) -> None:
        if not value or not self._capture_enabled.is_set():
            return
        sanitized = _sanitize_text(value, self.repo_root)
        path = self.stdout_path if stream_name == "stdout" else self.stderr_path
        try:
            _append_bounded_text(path, sanitized, max_bytes=self.config.max_log_bytes)
        except Exception as exc:
            self._degrade_capture(action="write support runtime output log", exc=exc)
            return
        timestamp = _utc_now_iso()
        with self._lock:
            self._metadata["updated_at"] = timestamp
            self._metadata["heartbeat_at"] = timestamp
        self._write_metadata()

    def note_capture_warning(self, warning: str) -> None:
        if not self._capture_enabled.is_set():
            return
        with self._lock:
            warnings = list(self._metadata.get("capture_warnings", []))
            if warning not in warnings:
                warnings.append(warning)
                self._metadata["capture_warnings"] = warnings
                self._metadata["updated_at"] = _utc_now_iso()
        self._write_metadata()

    def mark_stack_dump_requested(self) -> None:
        if not self._capture_enabled.is_set():
            return
        with self._lock:
            stack_dump = dict(self._metadata.get("stack_dump", {}))
            stack_dump["last_requested_at"] = _utc_now_iso()
            self._metadata["stack_dump"] = stack_dump
            self._metadata["updated_at"] = _utc_now_iso()
        self._write_metadata()

    def mark_stack_dump_captured(self) -> None:
        if not self._capture_enabled.is_set():
            return
        with self._lock:
            stack_dump = dict(self._metadata.get("stack_dump", {}))
            stack_dump["last_captured_at"] = _utc_now_iso()
            self._metadata["stack_dump"] = stack_dump
            self._metadata["updated_at"] = _utc_now_iso()
        self._write_metadata()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
            if not self._capture_enabled.is_set():
                return
            with self._lock:
                if self._metadata.get("status") != "running":
                    return
                self._metadata["heartbeat_at"] = _utc_now_iso()
                self._metadata["updated_at"] = _utc_now_iso()
            self._write_metadata()

    def _start_heartbeat_thread(self) -> None:
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _join_heartbeat_thread(self) -> None:
        if self._heartbeat_thread is None:
            return
        self._heartbeat_thread.join(timeout=0.2)

    def _install_streams(self) -> None:
        if self._installed_streams:
            return
        self._stdout_base = sys.stdout
        self._stderr_base = sys.stderr
        sys.stdout = _TeeTextStream(sys.stdout, self, "stdout")  # type: ignore[assignment]
        sys.stderr = _TeeTextStream(sys.stderr, self, "stderr")  # type: ignore[assignment]
        self._installed_streams = True

    def _uninstall_streams(self) -> None:
        if not self._installed_streams:
            return
        assert self._stdout_base is not None
        assert self._stderr_base is not None
        sys.stdout = self._stdout_base
        sys.stderr = self._stderr_base
        self._installed_streams = False

    def _install_stack_dump_handler(self) -> None:
        if os.name != "posix" or self._stack_signal is None:
            return
        try:
            self._stack_handle = self.stack_dump_path.open("a", encoding="utf8")
            faulthandler.register(self._stack_signal, file=self._stack_handle, all_threads=True)
        except (OSError, RuntimeError, ValueError):
            self._stack_supported = False
            if self._stack_handle is not None:
                self._stack_handle.close()
                self._stack_handle = None
            return
        self._stack_supported = True
        stack_dump = dict(self._metadata.get("stack_dump", {}))
        stack_dump["supported"] = True
        stack_dump["signal"] = signal.Signals(self._stack_signal).name
        self._metadata["stack_dump"] = stack_dump

    def _uninstall_stack_dump_handler(self) -> None:
        if self._stack_supported and self._stack_signal is not None:
            try:
                faulthandler.unregister(self._stack_signal)
            except RuntimeError:
                pass
        if self._stack_handle is not None:
            self._stack_handle.close()
            self._stack_handle = None

    def _enrich_failure_from_stdout(self) -> None:
        try:
            raw = self.stdout_path.read_text(encoding="utf8")
        except OSError:
            return
        payload = _parse_first_json_object(raw)
        if payload is None:
            return
        error_code, failure_codes = _extract_failure_contract(payload)
        with self._lock:
            if error_code and not self._metadata.get("error_code"):
                self._metadata["error_code"] = error_code
            existing_failure_codes = list(self._metadata.get("failure_codes", []))
            if failure_codes and not existing_failure_codes:
                self._metadata["failure_codes"] = failure_codes

    def _move_to_recent(self) -> None:
        destination = self.paths.recent_root / self.command_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(self.command_dir), str(destination))
        _prune_recent_commands(self.paths.recent_root, limit=self.config.recent_limit)

    def _write_metadata(self) -> None:
        if not self._capture_enabled.is_set():
            return
        try:
            with self._lock:
                snapshot = copy.deepcopy(self._metadata)
        except Exception as exc:
            self._degrade_capture(action="snapshot support runtime metadata", exc=exc)
            return
        try:
            with self._metadata_io_lock:
                if not self._capture_enabled.is_set():
                    return
                _atomic_write_json(self.meta_path, snapshot)
        except Exception as exc:
            self._degrade_capture(action="write support runtime metadata", exc=exc)

    def _run_capture_operation(
        self,
        *,
        action: str,
        callback: Callable[[], None],
        require_enabled: bool = True,
    ) -> None:
        if require_enabled and not self._capture_enabled.is_set():
            return
        try:
            callback()
        except Exception as exc:
            self._degrade_capture(action=action, exc=exc)

    def _degrade_capture(self, *, action: str, exc: Exception) -> None:
        normalized = _unwrap_storage_io_error(exc)
        detail = f"{type(normalized).__name__}: {normalized}"
        should_emit_warning = False
        with self._lock:
            self._capture_enabled.clear()
            if SUPPORT_RUNTIME_DEGRADED_WARNING_CODE not in self._warning_codes:
                self._warning_codes.append(SUPPORT_RUNTIME_DEGRADED_WARNING_CODE)
                self._warning_codes.sort()
            warnings = list(self._metadata.get("capture_warnings", []))
            if SUPPORT_RUNTIME_DEGRADED_WARNING_CODE not in warnings:
                warnings.append(SUPPORT_RUNTIME_DEGRADED_WARNING_CODE)
                self._metadata["capture_warnings"] = warnings
                self._metadata["updated_at"] = _utc_now_iso()
            if not self._degrade_warning_emitted:
                self._degrade_warning_emitted = True
                should_emit_warning = True
        if should_emit_warning:
            self._emit_runtime_warning(
                "warning: support runtime tracing degraded while attempting to "
                f"{action}; continuing command execution without support trace writes "
                f"({detail})"
            )

    def _emit_runtime_warning(self, message: str) -> None:
        stream: TextIO | None = self._stderr_base
        if stream is None:
            stream = sys.__stderr__ if getattr(sys, "__stderr__", None) is not None else sys.stderr
        try:
            stream.write(message + "\n")
            stream.flush()
        except Exception:
            return


def support_runtime_paths(repo_root: Path) -> SupportRuntimePaths:
    support_root = repo_root / ".gloggur" / "support"
    runtime_root = support_root / "runtime"
    return SupportRuntimePaths(
        repo_root=repo_root,
        support_root=support_root,
        runtime_root=runtime_root,
        active_root=runtime_root / "active",
        recent_root=runtime_root / "recent",
    )


def load_support_runtime_config(repo_root: Path) -> SupportRuntimeConfig:
    config_path = repo_root / SUPPORT_RUNTIME_CONFIG_PATH
    if not config_path.exists():
        return SupportRuntimeConfig()
    try:
        parsed = _parse_toml(config_path)
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="read support runtime config",
            path=str(config_path),
        ) from exc
    section = parsed.get("support")
    if not isinstance(section, dict):
        return SupportRuntimeConfig()
    enabled = bool(section.get("enabled", False))
    recent_limit = _coerce_positive_int(section.get("recent_limit"), DEFAULT_RECENT_COMMAND_LIMIT)
    max_log_bytes = _coerce_positive_int(section.get("max_log_bytes"), DEFAULT_LOG_BYTES)
    return SupportRuntimeConfig(
        enabled=enabled,
        recent_limit=recent_limit,
        max_log_bytes=max_log_bytes,
    )


def write_support_runtime_config(repo_root: Path, *, enabled: bool) -> Path:
    config_path = repo_root / SUPPORT_RUNTIME_CONFIG_PATH
    section_lines = [
        f"enabled = {'true' if enabled else 'false'}",
        f"recent_limit = {DEFAULT_RECENT_COMMAND_LIMIT}",
        f"max_log_bytes = {DEFAULT_LOG_BYTES}",
    ]
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = config_path.read_text(encoding="utf8") if config_path.exists() else ""
        updated = _replace_toml_section(existing, "support", section_lines)
        config_path.write_text(updated, encoding="utf8")
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="write support runtime config",
            path=str(config_path),
        ) from exc
    return config_path


def maybe_start_command_trace(
    command_name: str,
    *,
    argv: list[str] | None = None,
) -> CommandTraceSession | None:
    repo_root = discover_repo_root()
    config = load_support_runtime_config(repo_root)
    if not config.enabled:
        return None
    if command_name.startswith("support "):
        return None
    session = CommandTraceSession(
        repo_root=repo_root,
        command_name=command_name,
        argv=list(argv or sys.argv[1:]),
        config=config,
    )
    return session


def current_trace_session() -> CommandTraceSession | None:
    return _ACTIVE_TRACE_SESSION


def load_runtime_records(
    repo_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    paths = support_runtime_paths(repo_root)
    active = _load_record_root(paths.active_root)
    recent = _load_record_root(paths.recent_root)
    return active, recent


def request_active_stack_dumps(repo_root: Path) -> list[str]:
    warnings: list[str] = []
    active, _recent = load_runtime_records(repo_root)
    for record in active:
        pid = record.get("pid")
        stack_dump = record.get("stack_dump")
        if not isinstance(pid, int) or pid <= 0:
            continue
        if pid == os.getpid():
            continue
        if not isinstance(stack_dump, dict) or not bool(stack_dump.get("supported")):
            continue
        signal_name = str(stack_dump.get("signal") or "")
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            warnings.append(f"stack_dump_signal_unsupported:{record.get('command_id')}")
            continue
        meta_path = _record_meta_path(repo_root, str(record.get("command_id")))
        stack_path = _record_stack_path(repo_root, str(record.get("command_id")))
        before_mtime = stack_path.stat().st_mtime if stack_path.exists() else 0.0
        try:
            os.kill(pid, signal_value)
        except OSError:
            warnings.append(f"stack_dump_signal_failed:{record.get('command_id')}")
            continue
        _touch_stack_request(meta_path)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if stack_path.exists() and stack_path.stat().st_mtime > before_mtime:
                _touch_stack_capture(meta_path)
                break
            time.sleep(0.05)
        else:
            warnings.append(f"stack_dump_timeout:{record.get('command_id')}")
    return warnings


def should_auto_include_cache(
    *,
    active_records: list[dict[str, object]],
    recent_records: list[dict[str, object]],
    status_payload: dict[str, object] | None,
) -> bool:
    if _status_indicates_cache_trouble(status_payload):
        return True
    for record in active_records:
        if _record_indicates_cache_trouble(record, active=True):
            return True
    for record in recent_records:
        if _record_indicates_cache_trouble(record, active=False):
            return True
    return False


def capture_enabled(repo_root: Path) -> bool:
    return load_support_runtime_config(repo_root).enabled


def _status_indicates_cache_trouble(status_payload: dict[str, object] | None) -> bool:
    if not isinstance(status_payload, dict):
        return False
    build_state = status_payload.get("build_state")
    if isinstance(build_state, dict):
        if str(build_state.get("state")) in {"building", "interrupted"}:
            return True
    warning_codes = status_payload.get("warning_codes")
    warnings = (
        set(str(code) for code in warning_codes) if isinstance(warning_codes, list) else set()
    )
    return bool(
        warnings.intersection(
            {
                "build_in_progress",
                "index_interrupted",
                "stale_build_state",
                "vector_integrity_failed",
                "chunk_span_integrity_failed",
            }
        )
    )


def _record_indicates_cache_trouble(record: dict[str, object], *, active: bool) -> bool:
    if str(record.get("command_name")) == "index":
        if active:
            return True
        exit_code = int(record.get("exit_code") or 0)
        if exit_code != 0:
            return True
    failure_codes = record.get("failure_codes")
    if isinstance(failure_codes, list):
        codes = set(str(code) for code in failure_codes)
        if codes.intersection(
            {
                "cache_lock_held",
                "index_failure",
                "stale_build_state",
                "vector_integrity_failed",
                "chunk_span_integrity_failed",
            }
        ):
            return True
    build_state = record.get("build_state")
    if isinstance(build_state, dict) and str(build_state.get("state")) in {
        "building",
        "interrupted",
    }:
        return True
    return False


def _record_meta_path(repo_root: Path, command_id: str) -> Path:
    paths = support_runtime_paths(repo_root)
    for root in (paths.active_root, paths.recent_root):
        candidate = root / command_id / "meta.json"
        if candidate.exists():
            return candidate
    return paths.active_root / command_id / "meta.json"


def _record_stack_path(repo_root: Path, command_id: str) -> Path:
    paths = support_runtime_paths(repo_root)
    for root in (paths.active_root, paths.recent_root):
        candidate = root / command_id / "stackdump.log"
        if candidate.exists():
            return candidate
    return paths.active_root / command_id / "stackdump.log"


def _touch_stack_request(meta_path: Path) -> None:
    if not meta_path.exists():
        return
    try:
        payload = json.loads(meta_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    stack_dump = dict(payload.get("stack_dump", {}))
    stack_dump["last_requested_at"] = _utc_now_iso()
    payload["stack_dump"] = stack_dump
    _atomic_write_json(meta_path, payload)


def _touch_stack_capture(meta_path: Path) -> None:
    if not meta_path.exists():
        return
    try:
        payload = json.loads(meta_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    stack_dump = dict(payload.get("stack_dump", {}))
    stack_dump["last_captured_at"] = _utc_now_iso()
    payload["stack_dump"] = stack_dump
    _atomic_write_json(meta_path, payload)


def _load_record_root(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    payloads: list[dict[str, object]] = []
    for command_dir in sorted(root.iterdir()):
        meta_path = command_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    payloads.sort(
        key=lambda item: str(item.get("updated_at") or item.get("started_at") or ""),
        reverse=True,
    )
    return payloads


def _prune_recent_commands(root: Path, *, limit: int) -> None:
    if not root.exists():
        return
    command_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in command_dirs[limit:]:
        shutil.rmtree(path, ignore_errors=True)


def _append_bounded_text(path: Path, value: str, *, max_bytes: int) -> None:
    payload = value.encode("utf8", errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
    try:
        if path.stat().st_size <= max_bytes:
            return
        raw = path.read_bytes()
        path.write_bytes(raw[-max_bytes:])
    except OSError as exc:
        raise wrap_io_error(exc, operation="write support runtime log", path=str(path)) from exc


def _unwrap_storage_io_error(exc: Exception) -> Exception:
    if isinstance(exc, StorageIOError) and isinstance(exc.__cause__, Exception):
        return exc.__cause__
    return exc


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}"
    )
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")
        temp_path.replace(path)
    except OSError as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise wrap_io_error(
            exc,
            operation="write support runtime metadata",
            path=str(path),
        ) from exc


def _parse_toml(path: Path) -> dict[str, object]:
    try:
        import tomllib  # type: ignore[attr-defined]

        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except ModuleNotFoundError:
        import tomli  # type: ignore[import-not-found]

        with path.open("rb") as handle:
            parsed = tomli.load(handle)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _replace_toml_section(existing: str, section_name: str, body_lines: list[str]) -> str:
    header = f"[{section_name}]"
    lines = existing.splitlines()
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(lines):
        line = lines[index]
        if line.strip() == header:
            replaced = True
            output.append(header)
            output.extend(body_lines)
            index += 1
            while index < len(lines) and not lines[index].startswith("["):
                index += 1
            continue
        output.append(line)
        index += 1
    if not replaced:
        if output and output[-1] != "":
            output.append("")
        output.append(header)
        output.extend(body_lines)
    return "\n".join(output).rstrip() + "\n"


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate if candidate > 0 else default


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
    error_code: str | None = None
    failure_codes: list[str] = []
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        error_code = str(error["code"])
    elif isinstance(payload.get("error_code"), str):
        error_code = str(payload["error_code"])
    raw_failure_codes = payload.get("failure_codes")
    if isinstance(raw_failure_codes, list):
        failure_codes = [str(code) for code in raw_failure_codes if str(code)]
    compatibility = payload.get("compatibility")
    if isinstance(compatibility, dict) and not failure_codes:
        nested = compatibility.get("failure_codes")
        if isinstance(nested, list):
            failure_codes = [str(code) for code in nested if str(code)]
        nested_error = compatibility.get("error")
        if isinstance(nested_error, dict) and isinstance(nested_error.get("code"), str):
            error_code = str(nested_error["code"])
    return error_code, failure_codes


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
