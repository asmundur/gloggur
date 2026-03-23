from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from gloggur.byte_spans import LineByteSpanIndex
from gloggur.search.hybrid_search import HybridSearch
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.path_priors import (
    is_authority_path,
    is_docs_authority_path,
    is_docs_path,
    is_source_code_path,
    is_test_path,
    is_workflow_config_path,
    path_domain_score,
)
from gloggur.search.router.types import (
    BackendHit,
    BackendResult,
    ExecutionHints,
    QueryHints,
    SearchIntent,
)
from gloggur.symbol_index.models import SymbolOccurrence
from gloggur.symbol_index.store import SymbolIndexStore

_DEFINITION_LINE_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|func|interface|struct|trait|enum)\b"
)
_IMPORT_LINE_RE = re.compile(
    r"^\s*(?:from\s+\S+\s+import\b|import\b|use\b|#include\b|require(?:\(|\s))"
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


def _exact_ranking_adjustment(path: str, matched_line: str, *, hints: QueryHints) -> float:
    adjustment = 0.0
    if _DEFINITION_LINE_RE.search(matched_line):
        adjustment += 0.10
    if hints.query_domain == "docs_policy":
        if is_docs_authority_path(path):
            adjustment += 0.12
        elif is_docs_path(path):
            adjustment += 0.08
        elif is_workflow_config_path(path):
            adjustment += 0.06
        elif is_test_path(path):
            adjustment -= 0.08
        elif is_source_code_path(path):
            adjustment -= 0.05
    elif hints.query_domain == "workflow_config":
        if is_workflow_config_path(path):
            adjustment += 0.12
        elif is_docs_authority_path(path):
            adjustment += 0.10
        elif is_docs_path(path):
            adjustment += 0.06
        elif is_test_path(path):
            adjustment -= 0.08
        elif is_source_code_path(path):
            adjustment -= 0.04
    else:
        source_bonus = 0.08
        test_adjustment = -0.05
        if hints.query_kind == "mixed":
            source_bonus = 0.06
            test_adjustment = 0.04
        if is_source_code_path(path):
            adjustment += source_bonus
        if is_docs_path(path):
            adjustment -= 0.08
        if is_workflow_config_path(path):
            adjustment -= 0.10
        if is_test_path(path):
            adjustment += test_adjustment
    return adjustment


def _definition_ordering_enabled(hints: QueryHints) -> bool:
    return hints.query_domain == "code" and hints.query_kind in {"identifier", "declaration"}


def _definition_only_exact_query(hints: QueryHints) -> bool:
    return hints.query_domain == "code" and hints.query_kind == "declaration"


def _symbol_name_tails(symbols: tuple[str, ...]) -> tuple[str, ...]:
    tails: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        tail = symbol.split(".")[-1].split(":")[-1].split("#")[-1].strip()
        if len(tail) < 3 or tail in seen:
            continue
        seen.add(tail)
        tails.append(tail)
    return tuple(tails)


def _identifier_candidates(hints: QueryHints) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in (*_symbol_name_tails(hints.symbols), *hints.identifier_tokens):
        normalized = candidate.strip()
        if len(normalized) < 3 or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _line_mentions_identifier(line: str, identifier: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
            line,
        )
    )


def _line_mentions_query_identifier(line: str, hints: QueryHints) -> bool:
    candidates = _identifier_candidates(hints)
    if not candidates:
        return False
    return any(_line_mentions_identifier(line, candidate) for candidate in candidates)


def _classify_exact_match_role(*, path: str, matched_line: str, hints: QueryHints) -> str:
    stripped = matched_line.strip()
    if hints.query_domain == "code" and (
        is_docs_path(path) or is_test_path(path) or is_workflow_config_path(path)
    ):
        return "auxiliary"
    if _DEFINITION_LINE_RE.search(stripped) and _line_mentions_query_identifier(stripped, hints):
        return "definition"
    if _IMPORT_LINE_RE.search(stripped) and _line_mentions_query_identifier(stripped, hints):
        return "import"
    return "reference"


def _exact_role_priority(role: str | None) -> int:
    order = {
        "definition": 0,
        "same_file_context": 1,
        "reference": 2,
        "import": 3,
        "auxiliary": 4,
    }
    return order.get(role or "reference", 5)


def _finalize_exact_hit_roles(
    hit_map: dict[tuple[str, int, int], BackendHit],
    *,
    hints: QueryHints,
) -> dict[tuple[str, int, int], BackendHit]:
    if not hit_map or not _definition_ordering_enabled(hints):
        return hit_map
    definition_paths = {hit.path for hit in hit_map.values() if hit.match_role == "definition"}
    if not definition_paths:
        return hit_map
    finalized: dict[tuple[str, int, int], BackendHit] = {}
    for key, hit in hit_map.items():
        role = hit.match_role or "reference"
        if role == "reference" and hit.path in definition_paths:
            role = "same_file_context"
        finalized[key] = replace(hit, match_role=role)
    return finalized


def _final_exact_hits(
    hit_map: dict[tuple[str, int, int], BackendHit],
    *,
    hints: QueryHints,
    top_k: int,
) -> tuple[BackendHit, ...]:
    if not hit_map:
        return ()
    ordered_hits = list(_finalize_exact_hit_roles(hit_map, hints=hints).values())
    if _definition_only_exact_query(hints):
        definition_hits = [hit for hit in ordered_hits if hit.match_role == "definition"]
        if definition_hits:
            ordered_hits = definition_hits
    if _definition_ordering_enabled(hints):
        return tuple(
            sorted(
                ordered_hits,
                key=lambda item: (
                    _exact_role_priority(item.match_role),
                    -item.raw_score,
                    item.path,
                    item.start_line,
                ),
            )[:top_k]
        )
    ranked_hits = sorted(
        ordered_hits,
        key=lambda item: (-item.raw_score, item.path, item.start_line),
    )
    if hints.query_domain == "code" and hints.query_kind == "mixed":
        ranked_hits = _rebalance_mixed_code_exact_hits(ranked_hits)
    return tuple(ranked_hits[:top_k])


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


def _path_subtree_tokens(path: str) -> tuple[str, ...]:
    generic = {"src", "tests", "test", "python", "lib"}
    return tuple(
        segment.lower()
        for segment in Path(path).parts
        if segment and segment not in {"."} and segment.lower() not in generic
    )


def _mixed_code_test_relevance(*, implementation_path: str, candidate_path: str) -> int:
    implementation_segments = set(_path_subtree_tokens(implementation_path))
    candidate_segments = set(_path_subtree_tokens(candidate_path))
    shared_segments = len(implementation_segments & candidate_segments)

    implementation_stem = Path(implementation_path).stem.lower()
    candidate_stem = Path(candidate_path).stem.lower()
    normalized_candidate_stem = candidate_stem.removeprefix("test_").removesuffix("_test")
    if implementation_stem and implementation_stem == normalized_candidate_stem:
        shared_segments += 3
    elif implementation_stem and implementation_stem in normalized_candidate_stem:
        shared_segments += 2
    return shared_segments


def _rebalance_mixed_code_exact_hits(ordered_hits: list[BackendHit]) -> list[BackendHit]:
    """Promote one implementation hit plus one nearby test for mixed exact queries."""
    if len(ordered_hits) < 2:
        return ordered_hits

    implementation_candidates = [
        (index, hit)
        for index, hit in enumerate(ordered_hits)
        if is_source_code_path(hit.path)
        and not is_test_path(hit.path)
        and hit.match_role != "import"
    ]
    if not implementation_candidates:
        return ordered_hits

    implementation_index, implementation_hit = sorted(
        implementation_candidates,
        key=lambda item: (
            -item[1].raw_score,
            0 if item[1].match_role == "definition" else 1,
            item[0],
        ),
    )[0]

    ranked_tests = sorted(
        (
            (
                _mixed_code_test_relevance(
                    implementation_path=implementation_hit.path,
                    candidate_path=hit.path,
                ),
                hit.raw_score,
                -index,
                index,
            )
            for index, hit in enumerate(ordered_hits)
            if is_test_path(hit.path)
        ),
        reverse=True,
    )
    if not ranked_tests or ranked_tests[0][0] <= 0:
        return ordered_hits

    test_index = ranked_tests[0][-1]
    if implementation_index == 0 and test_index == 1:
        return ordered_hits

    ordered_indices = [implementation_index, test_index]
    ordered_indices.extend(
        index
        for index in range(len(ordered_hits))
        if index not in {implementation_index, test_index}
    )
    return [ordered_hits[index] for index in ordered_indices]


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
    include_hidden: bool,
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
        # Match ripgrep defaults unless the query explicitly needs hidden authority paths.
        if not include_hidden:
            dirs[:] = [name for name in dirs if not name.startswith(".")]
        dirs.sort()
        names.sort()
        for name in names:
            if not include_hidden and name.startswith("."):
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
    include_hidden: bool,
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
        include_hidden=include_hidden,
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
                        base_score
                        + bonus
                        + _exact_ranking_adjustment(
                            absolute_path,
                            source_line,
                            hints=hints,
                        ),
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
                    match_role=_classify_exact_match_role(
                        path=absolute_path,
                        matched_line=source_line,
                        hints=hints,
                    ),
                )
                if existing is None or candidate.raw_score > existing.raw_score:
                    hit_map[key] = candidate
                break
    return _final_exact_hits(hit_map, hints=hints, top_k=top_k)


def _dedupe_exact_patterns(
    patterns: list[tuple[str, tuple[str, ...], float]],
) -> list[tuple[str, tuple[str, ...], float]]:
    seen_patterns: set[str] = set()
    deduped_patterns: list[tuple[str, tuple[str, ...], float]] = []
    for pattern, tags, base_score in patterns:
        normalized = pattern.strip()
        if not normalized or normalized in seen_patterns:
            continue
        seen_patterns.add(normalized)
        deduped_patterns.append((normalized, tags, base_score))
    return deduped_patterns


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
    stripped_query = query.strip()
    locator_verbatim_mode = bool(intent.result_profile == "locator" and hints.verbatim_literal)
    locator_literal_mode = bool(intent.result_profile == "locator" and hints.literal_first)
    include_hidden = bool(
        hints.query_domain == "workflow_config"
        or ".github" in query
        or any(".github" in value for value in (*hints.path_hints, *hints.literals))
    )

    primary_patterns: list[tuple[str, tuple[str, ...], float]] = []
    if hints.verbatim_literal is not None:
        primary_patterns.append((hints.verbatim_literal, ("literal_match",), 1.0))
    elif hints.literal_first and stripped_query:
        primary_patterns.append((stripped_query, ("literal_match", "literal_first"), 0.98))

    fallback_patterns: list[tuple[str, tuple[str, ...], float]] = []
    if not locator_verbatim_mode and not locator_literal_mode:
        for literal in hints.literals:
            if literal == hints.verbatim_literal:
                continue
            fallback_patterns.append((literal, ("literal_match",), 1.0))
        for symbol in hints.symbols:
            fallback_patterns.append((symbol, ("symbol_match", "literal_match"), 0.9))
        if not fallback_patterns and stripped_query and stripped_query != hints.verbatim_literal:
            fallback_patterns.append((stripped_query, ("literal_match",), 0.7))

    if not primary_patterns and not fallback_patterns and stripped_query:
        fallback_patterns.append((stripped_query, ("literal_match",), 0.7))

    deduped_primary_patterns = _dedupe_exact_patterns(primary_patterns)
    deduped_fallback_patterns = _dedupe_exact_patterns(fallback_patterns)
    primary_execution_hints = (
        replace(execution_hints, fixed_string=True)
        if hints.verbatim_literal is not None or hints.literal_first
        else execution_hints
    )

    top_k = max(1, intent.max_snippets or config.exact_top_k)
    cmd_fragments: list[str] = []
    hit_map: dict[tuple[str, int, int], BackendHit] = {}
    hit_pattern_map: dict[tuple[str, int, int], set[str]] = {}
    file_pattern_map: dict[str, set[str]] = {}
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

    def _execute_pattern_batch(
        patterns: tuple[tuple[str, tuple[str, ...], float], ...],
        *,
        batch_hints: ExecutionHints,
    ) -> None:
        nonlocal ripgrep_unavailable
        for pattern, tags, base_score in patterns:
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
            if batch_hints.case_mode == "ignore":
                cmd.append("-i")
            elif batch_hints.case_mode == "smart":
                cmd.append("-S")
            if batch_hints.word_match:
                cmd.append("-w")
            if batch_hints.fixed_string:
                cmd.append("-F")
            if include_hidden:
                cmd.append("--hidden")
            for include_glob in batch_hints.include_globs:
                cmd.extend(["--glob", include_glob])
            for exclude_glob in batch_hints.exclude_globs:
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
                    execution_hints=batch_hints,
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
                bonus = (
                    0.05 if pattern in hints.literals or pattern == hints.verbatim_literal else 0.0
                )
                score = max(
                    0.0,
                    min(
                        1.0,
                        base_score
                        + bonus
                        + _exact_ranking_adjustment(
                            absolute_path,
                            parts[2].strip(),
                            hints=hints,
                        ),
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
                    match_role=_classify_exact_match_role(
                        path=absolute_path,
                        matched_line=parts[2].strip(),
                        hints=hints,
                    ),
                )
                file_pattern_map.setdefault(absolute_path, set()).add(pattern)
                hit_pattern_map.setdefault(key, set()).add(pattern)
                if existing is None or candidate.raw_score > existing.raw_score:
                    hit_map[key] = candidate

    if deduped_primary_patterns:
        _execute_pattern_batch(tuple(deduped_primary_patterns), batch_hints=primary_execution_hints)
    if not hit_map and not ripgrep_unavailable and deduped_fallback_patterns:
        _execute_pattern_batch(tuple(deduped_fallback_patterns), batch_hints=execution_hints)

    if ripgrep_unavailable and not hit_map:
        cmd_fragments.append("python_fallback_exact_scan")
        fallback_hits: tuple[BackendHit, ...] = ()
        pattern_batches: list[
            tuple[tuple[tuple[str, tuple[str, ...], float], ...], ExecutionHints]
        ] = []
        if deduped_primary_patterns:
            pattern_batches.append((tuple(deduped_primary_patterns), primary_execution_hints))
        if deduped_fallback_patterns:
            pattern_batches.append((tuple(deduped_fallback_patterns), execution_hints))
        for pattern_batch, batch_hints in pattern_batches:
            fallback_hits = _run_exact_backend_fallback_scan(
                deduped_patterns=pattern_batch,
                hints=hints,
                repo_root=repo_root,
                intent=intent,
                execution_hints=batch_hints,
                config=config,
                search_target=search_target,
                top_k=top_k,
                deadline=deadline,
                include_hidden=include_hidden,
            )
            if fallback_hits:
                break
        if fallback_hits:
            return BackendResult(
                name="exact",
                hits=fallback_hits,
                timing_ms=int((time.perf_counter() - started) * 1000),
                commands=tuple(cmd_fragments),
            )

    if hit_map:
        anchor_path = sorted(
            hit_map.values(),
            key=lambda item: (
                -len(file_pattern_map.get(item.path, ())),
                -item.raw_score,
                item.path,
                item.start_line,
            ),
        )[0].path
        anchor_parent = str(Path(anchor_path).parent)
        parent_hit_counts: dict[str, int] = {}
        for candidate in hit_map.values():
            parent = str(Path(candidate.path).parent)
            if parent in {"", "."}:
                continue
            parent_hit_counts[parent] = parent_hit_counts.get(parent, 0) + 1
        for key, hit in list(hit_map.items()):
            score = hit.raw_score
            file_pattern_count = len(file_pattern_map.get(hit.path, ()))
            hit_pattern_count = len(hit_pattern_map.get(key, ()))
            if file_pattern_count > 1:
                score += min(0.16, 0.05 * (file_pattern_count - 1))
            if hit_pattern_count > 1:
                score += min(0.08, 0.03 * (hit_pattern_count - 1))
            if hints.query_domain == "code" and hints.query_kind == "mixed":
                if hit.match_role == "import":
                    score -= 0.12
                elif hit.match_role == "definition" and not is_test_path(hit.path):
                    score -= 0.02
                elif hit.match_role == "reference" and is_source_code_path(hit.path):
                    score += 0.03
                if is_test_path(hit.path):
                    score += 0.04
            if (
                not _definition_ordering_enabled(hints)
                and hints.query_domain == "code"
                and anchor_parent not in {"", "."}
                and str(Path(hit.path).parent) == anchor_parent
                and parent_hit_counts.get(anchor_parent, 0) >= 2
            ):
                score += 0.02 if hints.query_kind == "mixed" else 0.05
            if not _definition_ordering_enabled(hints) and is_authority_path(
                hit.path, query_domain=hints.query_domain
            ):
                authority_bonus = 0.03
                if hints.query_domain == "code" and hints.query_kind == "mixed":
                    authority_bonus = 0.01
                score += authority_bonus
            hit_map[key] = replace(hit, raw_score=max(0.0, min(1.0, score)))

    hits = _final_exact_hits(hit_map, hints=hints, top_k=top_k)
    return BackendResult(
        name="exact",
        hits=hits,
        timing_ms=int((time.perf_counter() - started) * 1000),
        commands=tuple(cmd_fragments),
    )


def run_semantic_backend(
    *,
    query: str,
    hints: QueryHints,
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
    ranking_mode = "balanced"
    if intent.result_profile == "locator":
        ranking_mode = (
            intent.ranking_mode.strip().lower()
            if isinstance(intent.ranking_mode, str) and intent.ranking_mode.strip()
            else "balanced"
        )

    filters: dict[str, str] = {
        "ranking_mode": ranking_mode,
        "mode": intent.search_mode or "semantic",
    }
    if intent.path_prefix:
        filters["file"] = intent.path_prefix
    if intent.language:
        filters["language"] = intent.language

    top_k = max(1, intent.max_snippets or config.semantic_top_k)
    command_label = f"semantic search mode={filters['mode']} top_k={top_k}"
    try:
        payload = searcher.search(
            semantic_query,
            filters=filters,
            top_k=top_k,
            context_radius=8,
        )
    except Exception as exc:
        return BackendResult(
            name="semantic",
            hits=(),
            timing_ms=int((time.perf_counter() - started) * 1000),
            error=f"semantic backend failed: {type(exc).__name__}: {exc}",
            commands=(command_label,),
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
        commands=(command_label,),
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


def _symbol_role_priority(
    *,
    occurrence_kind: str,
    hints: QueryHints,
    usage_intent: bool,
) -> int:
    if usage_intent:
        return 0 if occurrence_kind == "ref" else 1
    if hints.query_domain != "code" or hints.query_kind not in {"identifier", "declaration"}:
        return 0 if occurrence_kind == "def" else 1
    if hints.query_kind == "declaration":
        return 0 if occurrence_kind == "def" else 2
    return 0 if occurrence_kind == "def" else 2


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
                score
                + max(
                    0.0,
                    path_domain_score(path, query_domain=hints.query_domain) - 0.5,
                )
                * 0.08,
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
                match_role="definition" if occurrence.kind == "def" else "reference",
            )
        )

    if _definition_only_exact_query(hints):
        definition_hits = [hit for hit in hits if hit.match_role == "definition"]
        if definition_hits:
            hits = definition_hits

    ordered = tuple(
        sorted(
            hits,
            key=lambda item: (
                _symbol_role_priority(
                    occurrence_kind="def" if item.match_role == "definition" else "ref",
                    hints=hints,
                    usage_intent=usage_intent,
                ),
                -item.raw_score,
                item.path,
                item.start_line,
                item.end_line,
            ),
        )[:top_k]
    )
    return BackendResult(
        name="symbol",
        hits=ordered,
        timing_ms=int((time.perf_counter() - started) * 1000),
        commands=("symbol index lookup",),
    )
