from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from gloggur.search.hybrid_search import HybridSearch
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.types import BackendHit, BackendResult, QueryHints, SearchConstraints
from gloggur.symbol_index.models import SymbolOccurrence
from gloggur.symbol_index.store import SymbolIndexStore


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


def _path_matches_any_prefix(path: str, prefixes: tuple[str, ...], *, repo_root: Path) -> bool:
    if not prefixes:
        return True
    return any(_path_matches_prefix(path, prefix, repo_root=repo_root) for prefix in prefixes)


def _glob_candidates(path: str, *, repo_root: Path) -> tuple[str, ...]:
    normalized = path.replace("\\", "/")
    candidates: set[str] = {normalized}
    try:
        relative = os.path.relpath(path, str(repo_root))
    except ValueError:
        relative = path
    candidates.add(relative.replace("\\", "/"))
    return tuple(sorted(candidates))


def _matches_include_globs(path: str, globs: tuple[str, ...], *, repo_root: Path) -> bool:
    if not globs:
        return True
    candidates = _glob_candidates(path, repo_root=repo_root)
    for pattern in globs:
        for candidate in candidates:
            if fnmatch.fnmatch(candidate, pattern):
                return True
    return False


def _matches_exclude_globs(path: str, globs: tuple[str, ...], *, repo_root: Path) -> bool:
    if not globs:
        return False
    candidates = _glob_candidates(path, repo_root=repo_root)
    for pattern in globs:
        for candidate in candidates:
            if fnmatch.fnmatch(candidate, pattern):
                return True
    return False


def _path_allowed(
    path: str,
    *,
    constraints: SearchConstraints,
    config: SearchRouterConfig,
    repo_root: Path,
) -> bool:
    if _matches_exclude_globs(path, config.ignore_globs, repo_root=repo_root):
        return False
    if not _path_matches_prefix(path, constraints.path_prefix, repo_root=repo_root):
        return False
    if not _path_matches_any_prefix(path, constraints.path_filters, repo_root=repo_root):
        return False
    if not _matches_include_globs(path, constraints.include_globs, repo_root=repo_root):
        return False
    if _matches_exclude_globs(path, constraints.exclude_globs, repo_root=repo_root):
        return False
    return True


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
    if constraints.path_prefix and not constraints.path_filters:
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
        if constraints.case_mode == "ignore":
            cmd.append("-i")
        elif constraints.case_mode == "smart":
            cmd.append("-S")
        if constraints.word_match:
            cmd.append("-w")
        if constraints.fixed_string:
            cmd.append("-F")
        for include_glob in constraints.include_globs:
            cmd.extend(["--glob", include_glob])
        for exclude_glob in constraints.exclude_globs:
            cmd.extend(["--glob", f"!{exclude_glob}"])
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
            absolute_path = relative_path
            if os.path.isabs(absolute_path):
                absolute_path = os.path.abspath(absolute_path)
            else:
                absolute_path = os.path.abspath(str(repo_root / relative_path))
            if not _path_allowed(
                absolute_path,
                constraints=constraints,
                config=config,
                repo_root=repo_root,
            ):
                continue

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
        if not _path_allowed(
            normalized_path,
            constraints=constraints,
            config=config,
            repo_root=repo_root,
        ):
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


_USAGE_INTENT_RE = re.compile(r"(?i)\b(call|calls|called|use|used|usage|reference|references)\b")


def _usage_intent(query: str) -> bool:
    lowered = query.lower()
    return "who calls" in lowered or bool(_USAGE_INTENT_RE.search(query))


def _resolve_case_sensitive(
    *,
    symbol_candidates: tuple[str, ...],
    constraints: SearchConstraints,
) -> bool:
    if constraints.case_mode == "ignore":
        return False
    if constraints.case_mode == "smart":
        return any(any(char.isupper() for char in candidate) for candidate in symbol_candidates)
    return False


def _symbol_match_score(
    symbol_name: str,
    symbol_candidates: tuple[str, ...],
    *,
    query_tokens: tuple[str, ...],
    case_sensitive: bool,
    fixed_string: bool,
    word_match: bool,
) -> float:
    """Score one indexed symbol token against extracted query hints."""
    normalized_symbol = symbol_name if case_sensitive else symbol_name.lower()
    best = 0.0
    for candidate in symbol_candidates:
        normalized_candidate = candidate if case_sensitive else candidate.lower()
        tail = normalized_candidate.split(".")[-1].split(":")[-1].split("#")[-1]
        if normalized_symbol == normalized_candidate:
            best = max(best, 1.0)
            continue
        if normalized_symbol == tail:
            best = max(best, 0.96)
            continue
        if word_match or fixed_string:
            continue
        if tail and (tail in normalized_symbol or normalized_symbol in tail):
            best = max(best, 0.74)
    for token in query_tokens:
        normalized_token = token if case_sensitive else token.lower()
        if len(normalized_token) < 3:
            continue
        if normalized_symbol == normalized_token:
            best = max(best, 0.82)
        elif not fixed_string and not word_match and normalized_token in normalized_symbol:
            best = max(best, 0.62)
    return best


def _kind_bonus(*, occurrence_kind: str, usage_intent: bool) -> float:
    if usage_intent:
        return 0.08 if occurrence_kind == "ref" else -0.12
    return 0.08 if occurrence_kind == "def" else 0.0


def run_symbol_backend(
    *,
    symbol_store: SymbolIndexStore,
    hints: QueryHints,
    query: str,
    repo_root: Path,
    constraints: SearchConstraints,
    config: SearchRouterConfig,
) -> BackendResult:
    """Run symbol occurrence backend over .gloggur/index/symbols.db."""
    started = time.perf_counter()

    top_k = max(1, constraints.max_snippets or config.symbol_top_k)
    if not symbol_store.available:
        return BackendResult(
            name="symbol",
            hits=(),
            timing_ms=int((time.perf_counter() - started) * 1000),
            error="symbol index unavailable: missing or unreadable symbols.db",
            commands=("symbol index lookup",),
        )
    try:
        prefix_filters: tuple[str, ...] = ()
        if constraints.path_prefix:
            prefix_filters = (constraints.path_prefix,)
        if constraints.path_filters:
            prefix_filters = tuple(list(prefix_filters) + list(constraints.path_filters))
        occurrences = symbol_store.list_occurrences(
            path_prefixes=prefix_filters,
            language=constraints.language,
        )
    except Exception as exc:
        return BackendResult(
            name="symbol",
            hits=(),
            timing_ms=int((time.perf_counter() - started) * 1000),
            error=f"symbol index unavailable: {exc}",
            commands=("symbol index lookup",),
        )

    if not occurrences:
        return BackendResult(
            name="symbol",
            hits=(),
            timing_ms=int((time.perf_counter() - started) * 1000),
            commands=("symbol index lookup",),
        )

    symbol_candidates = hints.symbols or tuple(
        token for token in hints.identifier_tokens if len(token) >= 3
    )
    if not symbol_candidates:
        return BackendResult(
            name="symbol",
            hits=(),
            timing_ms=int((time.perf_counter() - started) * 1000),
            commands=("symbol index lookup",),
        )

    case_sensitive = _resolve_case_sensitive(
        symbol_candidates=symbol_candidates,
        constraints=constraints,
    )
    query_tokens = tuple(token for token in hints.identifier_tokens if token)
    usage_intent = _usage_intent(query)

    hits: list[BackendHit] = []
    file_hit_counts: dict[str, int] = {}
    scored_occurrences: list[tuple[float, SymbolOccurrence, str]] = []
    for occurrence in occurrences:
        path = occurrence.path
        if not os.path.isabs(path):
            path = os.path.abspath(str(repo_root / path))
        if not _path_allowed(
            path,
            constraints=constraints,
            config=config,
            repo_root=repo_root,
        ):
            continue

        score = _symbol_match_score(
            occurrence.symbol,
            symbol_candidates,
            query_tokens=query_tokens,
            case_sensitive=case_sensitive,
            fixed_string=constraints.fixed_string,
            word_match=constraints.word_match,
        )
        if score <= 0.0:
            continue
        score = max(0.0, min(1.0, score + _kind_bonus(occurrence_kind=occurrence.kind, usage_intent=usage_intent)))
        scored_occurrences.append((score, occurrence, path))
        file_hit_counts[path] = file_hit_counts.get(path, 0) + 1

    for base_score, occurrence, path in scored_occurrences:
        group_count = file_hit_counts.get(path, 1)
        group_bonus = min(0.06, 0.02 * (group_count - 1))
        score = max(0.0, min(1.0, base_score + group_bonus))
        snippet = _load_snippet(
            repo_root,
            path,
            start_line=occurrence.line,
            end_line=occurrence.line,
            radius=2,
            max_chars=config.max_snippet_chars,
        )
        tags = ("symbol_def",) if occurrence.kind == "def" else ("symbol_ref",)
        hits.append(
            BackendHit(
                backend="symbol",
                path=path,
                start_line=occurrence.line,
                end_line=occurrence.line,
                snippet=snippet,
                raw_score=score,
                tags=tags,
            )
        )

    ordered = tuple(
        sorted(
            hits,
            key=lambda item: (-item.raw_score, item.path, item.start_line, item.end_line),
        )[:top_k]
    )
    return BackendResult(
        name="symbol",
        hits=ordered,
        timing_ms=int((time.perf_counter() - started) * 1000),
        commands=("symbol index lookup",),
    )
