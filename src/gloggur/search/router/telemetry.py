from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gloggur.search.router.config import SearchRouterConfig


def log_router_event(
    *,
    config: SearchRouterConfig,
    repo_root: Path,
    query: str,
    payload: dict[str, object],
) -> None:
    """Write privacy-safe local router telemetry."""
    if not config.telemetry_enabled:
        return
    log_path = repo_root / config.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    envelope: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_hash": hashlib.sha256(query.encode("utf8")).hexdigest(),
        **payload,
    }
    if config.telemetry_log_query:
        envelope["query"] = query

    with log_path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(envelope, separators=(",", ":"), ensure_ascii=True))
        handle.write("\n")
