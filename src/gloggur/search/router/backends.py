from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
import time
from pathlib import Path

from gloggur.search.hybrid_search import HybridSearch
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.types import BackendHit, BackendResult, QueryHints, SearchConstraints
from gloggur.storage.metadata_store import MetadataStore


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _path_variants(path: str, repo_root: Path) -> tuple[str, ...]:
    """Return stable relative/absolute variants for prefix matching."""
    normalized = _normalize_path(path)
    variants: set[str] = {normalized}
    repo_abs = _normalize_path(str(repo_root.resolve()))
    if os.path.isabs(path):
        try:
            relative = _normalize_path(os.path.relpath(path, str(repo_root)))
        except ValueError:
            relative = ""
        if relative and relative != "." and not relative.startswith(".."):
            variants.add(relative)
    else:
        absolute = _normalize_path(str((repo_root / path).resolve()))
        variants.add(absolute)
        if absolute.startswith(repo_abs + "/"):
            variants.add(_normalize_path(absolute[len(repo_abs) + 1 :]))
    return tuple(variants)


def _prefix_variants(prefix: str, repo_root: Path) -> tuple[str, ...]:
    """Return stable relative/absolute prefix variants."""
    normalized = _normalize_path(prefix)
    variants: set[str] = {normalized}
    repo_abs = _normalize_path(str(repo_root.resolve()))
    if os.path.isabs(prefix):
        try:
            relative = _normalize_path(os.path.relpath(prefix, str(repo_root)))
        except ValueError:
            relative = ""
        if relative and relative != "." and not relative.startswith(".."):
            variants.add(relative)
    else:
        absolute = _normalize_path(str((repo_root / prefix).resolve()))
        variants.add(absolute)
        if absolute.startswith(repo_abs + "/"):
            variants.add(_normalize_path(absolute[len(repo_abs) + 1 :]))
    return tuple(variants)


def _path_matches_prefix(path: str, prefix: str | None, *, repo_root: Path) -> bool:
    """Return True when path matches optional prefix constraint."""
    if not prefix:
        return True
    raw_prefix = _normalize_path(prefix.strip())
    if not raw_prefix:
        return True
    path_candidates = _path_variants(path, repo_root)
    prefix_candidates = _prefix_variants(raw_prefix, repo_root)
    for path_candidate in path_candidates:
        for prefix_candidate in prefix_candidates:
            if not prefix_candidate:
                continue
            if path_candidate == prefix_candidate:
                return True
            if path_candidate.startswith(prefix_candidate + "/"):
                return True
    return False


def _is_ignored(path: str, ignore_globs: tuple[str, ...]) -> bool:
    """Return True if a path matches any ignore glob."""
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in ignore_globs)


def _clip_text(value: str, max_chars: int) -> str:
    """Clip text with deterministic suffix."""
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + " ..."


def _load_snippet(
    repo_root: Path,
    path: str,
    *,
    start_line: int,
    end_line: int,
    radius: int,
    max_chars: int,
) -> str:
    """Load bounded snippet around a source span."""
    absolute_path = Path(path)
    if not absolute_path.is_absolute():
        absolute_path = repo_root / path
    try:
        with absolute_path.open(encoding="utf8") as handle:
            lines = handle.readlines()
    except OSError:
        return ""

    safe_start = max(1, start_line)
    safe_end = max(safe_start, end_line)
    view_start = max(1, safe_start - radius)
    view_end = min(len(lines), safe_end + radius)
    snippet = "".join(lines[view_start - 1 : view_end]).strip()
    return _clip_text(snippet, max_chars)


def run_exact_backend(
    *,
    query: str,
    hints: QueryHints,
    repo_root: Path,
    constraints: SearchConstraints,
    config: SearchRouterConfig,
) -> BackendResult:
    """Run deterministic literal/symbol exact backend using ripgrep."""
    started = time.perf_counter()
    patterns: list[tuple[str, tuple[str, ...], float]] = []

    for literal in hints.literals:
        patterns.append((literal, ("literal_match",), 1.0))
    for symbol in hints.symbols:
        patterns.append((symbol, ("symbol_match", "literal_match"), 0.9))
    if not patterns:
        patterns.append((query.strip(), ("literal_match",), 0.7))

    seen_patterns: set[str] = set()
    deduped_patterns: list[tuple[str, tuple[str, ...], float]] = []
    for pattern, tags, base_score in patterns:
        normalized = pattern.strip()
        if not normalized or normalized in seen_patterns:
            continue
        seen_patterns.add(normalized)
        deduped_patterns.append((normalized, tags, base_score))

    top_k = max(1, constraints.max_snippets or config.exact_top_k)
    cmd_fragments: list[str] = []
    hit_map: dict[tuple[str, int, int], BackendHit] = {}

    time_budget_ms = constraints.time_budget_ms or config.default_time_budget_ms
    time_budget_seconds = max(time_budget_ms / 1000.0, 0.1)
    per_pattern_timeout = max(0.05, min(time_budget_seconds, 0.35))

    search_target = "."
    if constraints.path_prefix:
        candidate = (repo_root / constraints.path_prefix).resolve()
        if candidate.exists():
            if candidate.is_file():
                search_target = str(Path(constraints.path_prefix))
            else:
                search_target = str(Path(constraints.path_prefix))

    for pattern, tags, base_score in deduped_patterns:
        if len(hit_map) >= top_k * 3:
            break
        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--max-count",
            str(top_k * 3),
            "-e",
            pattern,
            search_target,
        ]
        for ignore_glob in config.ignore_globs:
            cmd.extend(["--glob", f"!{ignore_glob}"])
        cmd_fragments.append(" ".join(shlex.quote(part) for part in cmd))

        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=str(repo_root),
                timeout=per_pattern_timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue

        if completed.returncode not in (0, 1):
            continue

        for line in completed.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            relative_path = parts[0].strip()
            if os.path.isabs(relative_path):
                try:
                    relative_path = os.path.relpath(relative_path, str(repo_root))
                except ValueError:
                    relative_path = relative_path
            try:
                line_number = int(parts[1])
            except ValueError:
                continue
            if line_number < 1:
                continue
            if _is_ignored(relative_path, config.ignore_globs):
                continue
            if not _path_matches_prefix(
                relative_path,
                constraints.path_prefix,
                repo_root=repo_root,
            ):
                continue
            absolute_path = relative_path
            if os.path.isabs(absolute_path):
                absolute_path = os.path.abspath(absolute_path)
            else:
                absolute_path = os.path.abspath(str(repo_root / relative_path))

            key = (absolute_path, line_number, line_number)
            snippet = _load_snippet(
                repo_root,
                absolute_path,
                start_line=line_number,
                end_line=line_number,
                radius=2,
                max_chars=config.max_snippet_chars,
            )
            bonus = 0.05 if pattern in hints.literals else 0.0
            score = max(0.0, min(1.0, base_score + bonus))
            existing = hit_map.get(key)
            candidate = BackendHit(
                backend="exact",
                path=absolute_path,
                start_line=line_number,
                end_line=line_number,
                snippet=snippet,
                raw_score=score,
                tags=tags,
            )
            if existing is None or candidate.raw_score > existing.raw_score:
                hit_map[key] = candidate

    hits = tuple(
        sorted(
            hit_map.values(),
            key=lambda item: (-item.raw_score, item.path, item.start_line),
        )[:top_k]
    )
    return BackendResult(
        name="exact",
        hits=hits,
        timing_ms=int((time.perf_counter() - started) * 1000),
        commands=tuple(cmd_fragments),
    )


def run_semantic_backend(
    *,
    query: str,
    searcher: HybridSearch,
    repo_root: Path,
    constraints: SearchConstraints,
    config: SearchRouterConfig,
) -> BackendResult:
    """Run semantic backend over existing HybridSearch."""
    started = time.perf_counter()

    filters: dict[str, str] = {
        "ranking_mode": "balanced",
        "mode": constraints.search_mode or "semantic",
    }
    if constraints.path_prefix:
        filters["file"] = constraints.path_prefix
    if constraints.language:
        filters["language"] = constraints.language

    top_k = max(1, constraints.max_snippets or config.semantic_top_k)
    payload = searcher.search(
        query,
        filters=filters,
        top_k=top_k,
        context_radius=8,
    )

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raw_results = []

    hits: list[BackendHit] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        path = item.get("file")
        if not isinstance(path, str) or not path.strip():
            continue
        normalized_path = path.strip()
        if os.path.isabs(normalized_path):
            normalized_path = os.path.abspath(normalized_path)
        else:
            normalized_path = os.path.abspath(str(repo_root / normalized_path))
        if _is_ignored(normalized_path, config.ignore_globs):
            continue
        if not _path_matches_prefix(normalized_path, constraints.path_prefix, repo_root=repo_root):
            continue
        try:
            start_line = int(item.get("line", 1))
        except (TypeError, ValueError):
            start_line = 1
        try:
            end_line = int(item.get("line_end", start_line))
        except (TypeError, ValueError):
            end_line = start_line
        if end_line < start_line:
            end_line = start_line

        raw_score = item.get("ranking_score")
        if not isinstance(raw_score, (int, float)):
            raw_score = item.get("similarity_score", 0.0)
        score = max(0.0, min(1.0, float(raw_score)))

        raw_snippet = item.get("context")
        snippet = raw_snippet if isinstance(raw_snippet, str) else ""
        snippet = _clip_text(snippet, config.max_snippet_chars)
        tags = ("semantic_match",)
        if score >= 0.8:
            tags = ("semantic_match", "semantic_high_conf")
        hits.append(
            BackendHit(
                backend="semantic",
                path=normalized_path,
                start_line=start_line,
                end_line=end_line,
                snippet=snippet,
                raw_score=score,
                tags=tags,
            )
        )

    ordered = tuple(
        sorted(
            hits,
            key=lambda item: (-item.raw_score, item.path, item.start_line),
        )[:top_k]
    )
    return BackendResult(
        name="semantic",
        hits=ordered,
        timing_ms=int((time.perf_counter() - started) * 1000),
        commands=(f"semantic search mode={filters['mode']} top_k={top_k}",),
    )


def _symbol_match_score(
    symbol_name: str,
    symbol_candidates: tuple[str, ...],
    query_tokens: tuple[str, ...],
) -> float:
    """Score one symbol name against extracted hints."""
    lowered = symbol_name.lower()
    best = 0.0
    for candidate in symbol_candidates:
        candidate_lower = candidate.lower()
        if lowered == candidate_lower:
            best = max(best, 1.0)
            continue
        tail = candidate_lower.split(".")[-1].split(":")[-1].split("#")[-1]
        if lowered == tail:
            best = max(best, 0.92)
            continue
        if tail in lowered or lowered in tail:
            best = max(best, 0.75)
    for token in query_tokens:
        if len(token) < 3:
            continue
        if lowered == token:
            best = max(best, 0.84)
        elif token in lowered:
            best = max(best, 0.65)
    return best


def run_symbol_backend(
    *,
    metadata_store: MetadataStore,
    hints: QueryHints,
    query: str,
    repo_root: Path,
    constraints: SearchConstraints,
    config: SearchRouterConfig,
) -> BackendResult:
    """Run symbol-name deterministic backend using existing metadata index."""
    started = time.perf_counter()

    top_k = max(1, constraints.max_snippets or config.symbol_top_k)
    try:
        symbols = metadata_store.list_symbols()
    except Exception as exc:
        return BackendResult(
            name="symbol",
            hits=(),
            timing_ms=int((time.perf_counter() - started) * 1000),
            error=f"metadata unavailable: {exc}",
            commands=("metadata list symbols",),
        )

    query_tokens = tuple(token for token in query.lower().split() if token)
    symbol_candidates = hints.symbols

    hits: list[BackendHit] = []
    for symbol in symbols:
        path = symbol.file_path
        if os.path.isabs(path):
            path = os.path.abspath(path)
        else:
            path = os.path.abspath(str(repo_root / path))
        if _is_ignored(path, config.ignore_globs):
            continue
        if not _path_matches_prefix(path, constraints.path_prefix, repo_root=repo_root):
            continue
        if constraints.language and symbol.language and symbol.language != constraints.language:
            continue

        score = _symbol_match_score(
            symbol.name,
            symbol_candidates,
            hints.identifier_tokens or query_tokens,
        )
        if score <= 0.0:
            continue

        snippet = _load_snippet(
            repo_root,
            path,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            radius=2,
            max_chars=config.max_snippet_chars,
        )
        hits.append(
            BackendHit(
                backend="symbol",
                path=path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                snippet=snippet,
                raw_score=max(0.0, min(1.0, score)),
                tags=("symbol_match",),
            )
        )

    ordered = tuple(
        sorted(
            hits,
            key=lambda item: (-item.raw_score, item.path, item.start_line),
        )[:top_k]
    )
    return BackendResult(
        name="symbol",
        hits=ordered,
        timing_ms=int((time.perf_counter() - started) * 1000),
        commands=("symbol index scan",),
    )
