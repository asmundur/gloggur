from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SearchRouterConfig:
    """Router-level deterministic retrieval and pack configuration."""

    enabled_backends: tuple[str, ...] = ("exact", "semantic", "symbol")
    default_time_budget_ms: int = 900
    threshold_exact: float = 0.70
    threshold_semantic: float = 0.66
    threshold_symbol: float = 0.68
    max_files: int = 8
    max_snippets: int = 12
    max_chars: int = 12000
    max_snippet_chars: int = 320
    exact_top_k: int = 24
    semantic_top_k: int = 16
    symbol_top_k: int = 24
    ignore_globs: tuple[str, ...] = ("dist/**", "vendor/**", "node_modules/**")
    log_path: str = ".gloggur/logs/search_router.jsonl"
    telemetry_enabled: bool = True
    telemetry_log_query: bool = False
    telemetry_log_snippets: bool = False


def _parse_toml(path: Path) -> dict[str, object]:
    """Parse TOML payload with py3.10 compatibility."""
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


def _coerce_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


def _coerce_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if coerced < 0.0:
        return 0.0
    if coerced > 1.0:
        return 1.0
    return coerced


def _coerce_str_tuple(data: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = data.get(key)
    if not isinstance(raw, list):
        return default
    values = [str(item).strip() for item in raw if str(item).strip()]
    return tuple(values) if values else default


def load_search_router_config(repo_root: Path) -> SearchRouterConfig:
    """Load router config from .gloggur/config.toml with deterministic defaults."""
    config_path = repo_root / ".gloggur" / "config.toml"
    if not config_path.exists():
        return SearchRouterConfig()

    parsed = _parse_toml(config_path)
    section = parsed.get("search_router")
    if not isinstance(section, dict):
        # Backward-friendly alias.
        section = parsed.get("router") if isinstance(parsed.get("router"), dict) else {}

    if not isinstance(section, dict):
        section = {}

    return SearchRouterConfig(
        enabled_backends=_coerce_str_tuple(
            section,
            "enabled_backends",
            SearchRouterConfig.enabled_backends,
        ),
        default_time_budget_ms=_coerce_int(
            section,
            "default_time_budget_ms",
            SearchRouterConfig.default_time_budget_ms,
        ),
        threshold_exact=_coerce_float(
            section,
            "threshold_exact",
            SearchRouterConfig.threshold_exact,
        ),
        threshold_semantic=_coerce_float(
            section,
            "threshold_semantic",
            SearchRouterConfig.threshold_semantic,
        ),
        threshold_symbol=_coerce_float(
            section,
            "threshold_symbol",
            SearchRouterConfig.threshold_symbol,
        ),
        max_files=_coerce_int(section, "max_files", SearchRouterConfig.max_files),
        max_snippets=_coerce_int(section, "max_snippets", SearchRouterConfig.max_snippets),
        max_chars=_coerce_int(section, "max_chars", SearchRouterConfig.max_chars),
        max_snippet_chars=_coerce_int(
            section,
            "max_snippet_chars",
            SearchRouterConfig.max_snippet_chars,
        ),
        exact_top_k=_coerce_int(section, "exact_top_k", SearchRouterConfig.exact_top_k),
        semantic_top_k=_coerce_int(section, "semantic_top_k", SearchRouterConfig.semantic_top_k),
        symbol_top_k=_coerce_int(section, "symbol_top_k", SearchRouterConfig.symbol_top_k),
        ignore_globs=_coerce_str_tuple(section, "ignore_globs", SearchRouterConfig.ignore_globs),
        log_path=str(section.get("log_path", SearchRouterConfig.log_path)),
        telemetry_enabled=bool(
            section.get("telemetry_enabled", SearchRouterConfig.telemetry_enabled)
        ),
        telemetry_log_query=bool(
            section.get("telemetry_log_query", SearchRouterConfig.telemetry_log_query)
        ),
        telemetry_log_snippets=bool(
            section.get("telemetry_log_snippets", SearchRouterConfig.telemetry_log_snippets)
        ),
    )
