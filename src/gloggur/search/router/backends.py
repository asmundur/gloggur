from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from gloggur.byte_spans import LineByteSpanIndex
from gloggur.search.hybrid_search import HybridSearch
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.types import (
    BackendHit,
    BackendResult,
    ExecutionHints,
    QueryHints,
    SearchIntent,
)
from gloggur.symbol_index.models import SymbolOccurrence
from gloggur.symbol_index.store import SymbolIndexStore

_SOURCE_CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".m",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".swift",
    ".ts",
    ".tsx",
}
_DOC_NAME_MARKERS = ("readme", "changelog", "changes", "history", "news", "release-notes")
_DEFINITION_LINE_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|func|interface|struct|trait|enum)\b"
)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _ripgrep_env() -> dict[str, str]:
    """Return a deterministic subprocess environment for ripgrep."""
    env = dict(os.environ)
    env.pop("RIPGREP_CONFIG_PATH", None)
    return env


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
    repo_root_name = _normalize_path(repo_root.name)
    if repo_root_name and raw_prefix == repo_root_name:
        return True
    if repo_root_name and raw_prefix.startswith(repo_root_name + "/"):
        stripped_prefix = raw_prefix[len(repo_root_name) + 1 :]
        if not stripped_prefix:
            return True
        for path_candidate in path_candidates:
            if path_candidate == stripped_prefix:
                return True
            if path_candidate.startswith(stripped_prefix + "/"):
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
    intent: SearchIntent,
    execution_hints: ExecutionHints,
    config: SearchRouterConfig,
    repo_root: Path,
) -> bool:
    if _matches_exclude_globs(path, config.ignore_globs, repo_root=repo_root):
        return False
    if not _path_matches_prefix(path, intent.path_prefix, repo_root=repo_root):
        return False
    if not _path_matches_any_prefix(path, intent.path_filters, repo_root=repo_root):
        return False
    if not _matches_include_globs(path, execution_hints.include_globs, repo_root=repo_root):
        return False
    if _matches_exclude_globs(path, execution_hints.exclude_globs, repo_root=repo_root):
        return False
    return True


def _is_source_code_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    return Path(lowered).suffix in _SOURCE_CODE_EXTENSIONS


def _is_docs_or_changelog_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    basename = Path(lowered).name
    stem = Path(lowered).stem
    if "/docs/" in lowered or "/doc/" in lowered:
        return True
    if basename.endswith((".md", ".rst", ".txt")) and stem in _DOC_NAME_MARKERS:
        return True
    return any(marker in stem for marker in _DOC_NAME_MARKERS)


def _is_test_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    basename = Path(lowered).name
    return (
        "/test/" in lowered
        or "/tests/" in lowered
        or basename.startswith("test_")
        or basename.endswith("_test.py")
        or ".spec." in basename
    )


def _exact_ranking_adjustment(path: str, matched_line: str) -> float:
    adjustment = 0.0
    if _DEFINITION_LINE_RE.search(matched_line):
        adjustment += 0.10
    if _is_source_code_path(path):
        adjustment += 0.08
    if _is_docs_or_changelog_path(path):
        adjustment -= 0.08
    if _is_test_path(path):
        adjustment -= 0.05
    return adjustment


def _sorted_exact_hits(
    hit_map: dict[tuple[str, int, int], BackendHit],
    top_k: int,
) -> tuple[BackendHit, ...]:
    return tuple(
        sorted(
            hit_map.values(),
            key=lambda item: (-item.raw_score, item.path, item.start_line),
        )[:top_k]
    )


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
        with absolute_path.open(encoding="utf8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ""

    safe_start = max(1, start_line)
    safe_end = max(safe_start, end_line)
    view_start = max(1, safe_start - radius)
    view_end = min(len(lines), safe_end + radius)
    snippet = "".join(lines[view_start - 1 : view_end]).strip()
    return _clip_text(snippet, max_chars)


def _line_byte_span(
    cache: dict[str, LineByteSpanIndex],
    path: str,
    *,
    start_line: int,
    end_line: int,
) -> tuple[int, int] | None:
    """Return the raw-byte span for a line-aligned source range."""
    index = cache.get(path)
    if index is None:
        try:
            raw_bytes = Path(path).read_bytes()
        except OSError:
            return None
        index = LineByteSpanIndex.from_bytes(raw_bytes)
        cache[path] = index
    try:
        return index.span_for_lines(start_line, end_line)
    except ValueError:
        return None


def _iter_fallback_search_files(
    *,
    repo_root: Path,
    search_target: str,
    max_files: int | None,
) -> tuple[str, ...]:
    """Return deterministic file candidates for exact backend fallback scans."""
    target = Path(search_target)
    if not target.is_absolute():
        target = repo_root / target
    target = target.resolve()
    if target.is_file():
        return (str(target),)
    if not target.exists() or not target.is_dir():
        return ()
    files: list[str] = []
    for root, dirs, names in os.walk(target):
        # Match ripgrep defaults: ignore hidden directories/files unless explicitly requested.
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        dirs.sort()
        names.sort()
        for name in names:
            if name.startswith("."):
                continue
            files.append(str(Path(root) / name))
            if max_files is not None and max_files > 0 and len(files) >= max_files:
                return tuple(files)
    return tuple(files)


def _compile_exact_matcher(
    pattern: str,
    execution_hints: ExecutionHints,
) -> re.Pattern[str] | None:
    """Compile a Python matcher that approximates ripgrep option semantics."""
    case_mode = (execution_hints.case_mode or "sensitive").strip().lower()
    flags = 0
    if case_mode == "ignore":
        flags |= re.IGNORECASE
    elif case_mode == "smart" and not any(char.isupper() for char in pattern):
        flags |= re.IGNORECASE

    if execution_hints.fixed_string:
        expression = re.escape(pattern)
    else:
        expression = pattern
    if execution_hints.word_match:
        expression = rf"\b{expression}\b"

    try:
        return re.compile(expression, flags)
    except re.error:
        if execution_hints.fixed_string:
            return None
        escaped = re.escape(pattern)
        if execution_hints.word_match:
            escaped = rf"\b{escaped}\b"
        return re.compile(escaped, flags)


def _is_probably_binary(path: str) -> bool:
    """Best-effort binary detector for fallback scans."""
    try:
        with Path(path).open("rb") as handle:
            sample = handle.read(2048)
    except OSError:
        return True
    return b"\x00" in sample


def _run_exact_backend_fallback_scan(
    *,
    deduped_patterns: tuple[tuple[str, tuple[str, ...], float], ...],
    hints: QueryHints,
    repo_root: Path,
    intent: SearchIntent,
    execution_hints: ExecutionHints,
    config: SearchRouterConfig,
    search_target: str,
    top_k: int,
    deadline: float,
) -> tuple[BackendHit, ...]:
    """Fallback exact scan used when ripgrep invocation is unavailable."""
    compiled_patterns: list[tuple[str, re.Pattern[str], tuple[str, ...], float]] = []
    for pattern, tags, base_score in deduped_patterns:
        matcher = _compile_exact_matcher(pattern, execution_hints)
        if matcher is None:
            continue
        compiled_patterns.append((pattern, matcher, tags, base_score))
    if not compiled_patterns:
        return ()

    file_candidates = _iter_fallback_search_files(
        repo_root=repo_root,
        search_target=search_target,
        max_files=intent.max_files,
    )
    hit_map: dict[tuple[str, int, int], BackendHit] = {}
    byte_span_cache: dict[str, LineByteSpanIndex] = {}
    for absolute_path in file_candidates:
        if len(hit_map) >= top_k * 3:
            break
        if deadline - time.perf_counter() < 0.02:
            break
        if not _path_allowed(
            absolute_path,
            intent=intent,
            execution_hints=execution_hints,
            config=config,
            repo_root=repo_root,
        ):
            continue
        if _is_probably_binary(absolute_path):
            continue
        try:
            with Path(absolute_path).open(encoding="utf8", errors="replace") as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for line_number, source_line in enumerate(lines, start=1):
            if len(hit_map) >= top_k * 3:
                break
            for pattern, matcher, tags, base_score in compiled_patterns:
                if not matcher.search(source_line):
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
                score = max(
                    0.0,
                    min(
                        1.0,
                        base_score + bonus + _exact_ranking_adjustment(absolute_path, source_line),
                    ),
                )
                existing = hit_map.get(key)
                byte_span = _line_byte_span(
                    byte_span_cache,
                    absolute_path,
                    start_line=line_number,
                    end_line=line_number,
                )
                candidate = BackendHit(
                    backend="exact",
                    path=absolute_path,
                    start_line=line_number,
                    end_line=line_number,
                    snippet=snippet,
                    raw_score=score,
                    start_byte=byte_span[0] if byte_span is not None else None,
                    end_byte=byte_span[1] if byte_span is not None else None,
                    tags=tags,
                )
                if existing is None or candidate.raw_score > existing.raw_score:
                    hit_map[key] = candidate
                break
    return _sorted_exact_hits(hit_map, top_k)


def run_exact_backend(
    *,
    query: str,
    hints: QueryHints,
    repo_root: Path,
    intent: SearchIntent,
    execution_hints: ExecutionHints,
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

    top_k = max(1, intent.max_snippets or config.exact_top_k)
    cmd_fragments: list[str] = []
    hit_map: dict[tuple[str, int, int], BackendHit] = {}
    byte_span_cache: dict[str, LineByteSpanIndex] = {}

    time_budget_ms = intent.time_budget_ms or config.default_time_budget_ms
    time_budget_seconds = max(time_budget_ms / 1000.0, 0.1)
    deadline = started + time_budget_seconds

    search_target = "."
    if intent.path_prefix and not intent.path_filters:
        candidate = (repo_root / intent.path_prefix).resolve()
        if candidate.exists():
            if candidate.is_file():
                search_target = str(Path(intent.path_prefix))
            else:
                search_target = str(Path(intent.path_prefix))

    ripgrep_unavailable = False
    for pattern, tags, base_score in deduped_patterns:
        if len(hit_map) >= top_k * 3:
            break
        remaining_time = deadline - time.perf_counter()
        if remaining_time < 0.05:
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
        if execution_hints.case_mode == "ignore":
            cmd.append("-i")
        elif execution_hints.case_mode == "smart":
            cmd.append("-S")
        if execution_hints.word_match:
            cmd.append("-w")
        if execution_hints.fixed_string:
            cmd.append("-F")
        for include_glob in execution_hints.include_globs:
            cmd.extend(["--glob", include_glob])
        for exclude_glob in execution_hints.exclude_globs:
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
                timeout=remaining_time,
                env=_ripgrep_env(),
            )
        except subprocess.TimeoutExpired:
            continue
        except OSError:
            ripgrep_unavailable = True
            break

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
                intent=intent,
                execution_hints=execution_hints,
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
            score = max(
                0.0,
                min(
                    1.0,
                    base_score + bonus + _exact_ranking_adjustment(absolute_path, parts[2].strip()),
                ),
            )
            existing = hit_map.get(key)
            byte_span = _line_byte_span(
                byte_span_cache,
                absolute_path,
                start_line=line_number,
                end_line=line_number,
            )
            candidate = BackendHit(
                backend="exact",
                path=absolute_path,
                start_line=line_number,
                end_line=line_number,
                snippet=snippet,
                raw_score=score,
                start_byte=byte_span[0] if byte_span is not None else None,
                end_byte=byte_span[1] if byte_span is not None else None,
                tags=tags,
            )
            if existing is None or candidate.raw_score > existing.raw_score:
                hit_map[key] = candidate

    if ripgrep_unavailable and not hit_map:
        cmd_fragments.append("python_fallback_exact_scan")
        fallback_hits = _run_exact_backend_fallback_scan(
            deduped_patterns=tuple(deduped_patterns),
            hints=hints,
            repo_root=repo_root,
            intent=intent,
            execution_hints=execution_hints,
            config=config,
            search_target=search_target,
            top_k=top_k,
            deadline=deadline,
        )
        if fallback_hits:
            return BackendResult(
                name="exact",
                hits=fallback_hits,
                timing_ms=int((time.perf_counter() - started) * 1000),
                commands=tuple(cmd_fragments),
            )

    hits = _sorted_exact_hits(hit_map, top_k)
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
    intent: SearchIntent,
    execution_hints: ExecutionHints,
    config: SearchRouterConfig,
) -> BackendResult:
    """Run semantic backend over existing HybridSearch."""
    started = time.perf_counter()
    semantic_query = (
        intent.semantic_query.strip()
        if isinstance(intent.semantic_query, str) and intent.semantic_query.strip()
        else query
    )

    filters: dict[str, str] = {
        "ranking_mode": "balanced",
        "mode": intent.search_mode or "semantic",
    }
    if intent.path_prefix:
        filters["file"] = intent.path_prefix
    if intent.language:
        filters["language"] = intent.language

    top_k = max(1, intent.max_snippets or config.semantic_top_k)
    payload = searcher.search(
        semantic_query,
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
            intent=intent,
            execution_hints=execution_hints,
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
                start_byte=(
                    int(item.get("start_byte")) if isinstance(item.get("start_byte"), int) else None
                ),
                end_byte=(
                    int(item.get("end_byte")) if isinstance(item.get("end_byte"), int) else None
                ),
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
    execution_hints: ExecutionHints,
) -> bool:
    if execution_hints.case_mode == "ignore":
        return False
    if execution_hints.case_mode == "smart":
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
    intent: SearchIntent,
    execution_hints: ExecutionHints,
    config: SearchRouterConfig,
) -> BackendResult:
    """Run symbol occurrence backend over .gloggur/index/symbols.db."""
    started = time.perf_counter()

    top_k = max(1, intent.max_snippets or config.symbol_top_k)
    if not symbol_store.available:
        reason = symbol_store.unavailability_reason or "missing or unreadable symbols.db"
        return BackendResult(
            name="symbol",
            hits=(),
            timing_ms=int((time.perf_counter() - started) * 1000),
            error=f"symbol index unavailable: {reason}",
            commands=("symbol index lookup",),
        )
    try:
        prefix_filters: tuple[str, ...] = ()
        if intent.path_prefix:
            prefix_filters = (intent.path_prefix,)
        if intent.path_filters:
            prefix_filters = tuple(list(prefix_filters) + list(intent.path_filters))
        occurrences = symbol_store.list_occurrences(
            path_prefixes=prefix_filters,
            language=intent.language,
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
        execution_hints=execution_hints,
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
            intent=intent,
            execution_hints=execution_hints,
            config=config,
            repo_root=repo_root,
        ):
            continue

        score = _symbol_match_score(
            occurrence.symbol,
            symbol_candidates,
            query_tokens=query_tokens,
            case_sensitive=case_sensitive,
            fixed_string=execution_hints.fixed_string,
            word_match=execution_hints.word_match,
        )
        if score <= 0.0:
            continue
        score = max(
            0.0,
            min(
                1.0,
                score + _kind_bonus(occurrence_kind=occurrence.kind, usage_intent=usage_intent),
            ),
        )
        scored_occurrences.append((score, occurrence, path))
        file_hit_counts[path] = file_hit_counts.get(path, 0) + 1

    for base_score, occurrence, path in scored_occurrences:
        group_count = file_hit_counts.get(path, 1)
        group_bonus = min(0.06, 0.02 * (group_count - 1))
        score = max(0.0, min(1.0, base_score + group_bonus))
        snippet = _load_snippet(
            repo_root,
            path,
            start_line=occurrence.start_line,
            end_line=occurrence.end_line,
            radius=2,
            max_chars=config.max_snippet_chars,
        )
        tags = ("symbol_def",) if occurrence.kind == "def" else ("symbol_ref",)
        hits.append(
            BackendHit(
                backend="symbol",
                path=path,
                start_line=occurrence.start_line,
                end_line=occurrence.end_line,
                snippet=snippet,
                raw_score=score,
                start_byte=occurrence.start_byte,
                end_byte=occurrence.end_byte,
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
