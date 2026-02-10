from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

_TRACE_ID: ContextVar[str] = ContextVar("gloggur_trace_id", default="-")
_CONFIGURED = False


class TraceIdFilter(logging.Filter):
    """Inject a trace id into log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        """Attach trace id to the log record."""
        record.trace_id = _TRACE_ID.get()
        return True


def set_trace_id(trace_id: str) -> None:
    """Set the current trace id."""
    _TRACE_ID.set(trace_id)


def get_trace_id() -> str:
    """Return the current trace id."""
    return _TRACE_ID.get()


def configure_logging(
    *,
    debug: bool = False,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    trace_id: Optional[str] = None,
    stream: Optional[str] = None,
    force: bool = False,
) -> str:
    """Configure global logging for verification scripts."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return get_trace_id()

    resolved_trace_id = trace_id or os.getenv("GLOGGUR_TRACE_ID") or uuid.uuid4().hex[:12]
    set_trace_id(resolved_trace_id)
    os.environ["GLOGGUR_TRACE_ID"] = resolved_trace_id

    resolved_log_file = log_file or os.getenv("GLOGGUR_LOG_FILE")
    resolved_level = (log_level or os.getenv("GLOGGUR_LOG_LEVEL") or "INFO").upper()
    if debug or os.getenv("GLOGGUR_DEBUG_LOGS"):
        resolved_level = "DEBUG"
        os.environ["GLOGGUR_DEBUG_LOGS"] = "1"
    else:
        os.environ.pop("GLOGGUR_DEBUG_LOGS", None)
    os.environ["GLOGGUR_LOG_LEVEL"] = resolved_level
    if resolved_log_file:
        os.environ["GLOGGUR_LOG_FILE"] = resolved_log_file

    level = getattr(logging, resolved_level, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if force:
        root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    resolved_stream = (stream or os.getenv("GLOGGUR_LOG_STREAM") or "stdout").lower()
    stream_target = sys.stderr if resolved_stream == "stderr" else sys.stdout
    stream_handler = logging.StreamHandler(stream_target)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(TraceIdFilter())
    root.addHandler(stream_handler)

    if resolved_log_file:
        file_handler = logging.FileHandler(resolved_log_file, encoding="utf8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(TraceIdFilter())
        root.addHandler(file_handler)

    root.addFilter(TraceIdFilter())
    _CONFIGURED = True
    return resolved_trace_id


def log_event(logger: logging.Logger, level: int, event: str, **fields: object) -> None:
    """Log an event with structured JSON fields."""
    if fields:
        payload = json.dumps(fields, default=str, ensure_ascii=True, sort_keys=True)
        logger.log(level, "%s | %s", event, payload)
        return
    logger.log(level, "%s", event)
