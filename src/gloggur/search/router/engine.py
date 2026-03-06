from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from gloggur.byte_spans import to_repo_relative_path
from gloggur.search.hybrid_search import HybridSearch
from gloggur.search.router.backends import (
    run_exact_backend,
    run_semantic_backend,
    run_symbol_backend,
)
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.hints import extract_query_hints
from gloggur.search.router.query_compat import ParsedQueryCompat, parse_query_compat
from gloggur.search.router.telemetry import log_router_event
from gloggur.search.router.types import (
    BackendHit,
    BackendResult,
    ContextHit,
    ContextPack,
    ContextSpan,
    QueryHints,
    RouterOutcome,
    SearchConstraints,
)
from gloggur.storage.metadata_store import MetadataStore
from gloggur.symbol_index.store import SymbolIndexStore

_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_BACKEND_TIE_PRIORITY = {
    "exact": 0,
    "symbol": 1,
    "semantic": 2,
}


def _normalize_string_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        candidate = str(raw).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return tuple(ordered)


def _normalize_constraints(
    constraints: SearchConstraints | None,
    config: SearchRouterConfig,
) -> SearchConstraints:
    """Resolve runtime constraints with config defaults."""
    base = constraints or SearchConstraints()
    max_files = (
        base.max_files
        if isinstance(base.max_files, int) and base.max_files > 0
        else config.max_files
    )
    max_snippets = (
        base.max_snippets
        if isinstance(base.max_snippets, int) and base.max_snippets > 0
        else config.max_snippets
    )
    time_budget_ms = (
        base.time_budget_ms
        if isinstance(base.time_budget_ms, int) and base.time_budget_ms > 0
        else config.default_time_budget_ms
    )
    language = (
        base.language.strip() if isinstance(base.language, str) and base.language.strip() else None
    )
    path_prefix = (
        base.path_prefix.strip()
        if isinstance(base.path_prefix, str) and base.path_prefix.strip()
        else None
    )
    search_mode = (
        base.search_mode.strip().lower()
        if isinstance(base.search_mode, str) and base.search_mode.strip()
        else "semantic"
    )
    if search_mode not in {"semantic", "by_fqname", "by_fqname_regex", "by_path"}:
        search_mode = "semantic"
    path_filters = _normalize_string_tuple(base.path_filters)
    include_globs = _normalize_string_tuple(base.include_globs)
    exclude_globs = _normalize_string_tuple(base.exclude_globs)
    case_mode: str | None = None
    if isinstance(base.case_mode, str):
        normalized_case = base.case_mode.strip().lower()
        if normalized_case in {"ignore", "smart"}:
            case_mode = normalized_case
    return SearchConstraints(
        search_mode=search_mode,
        language=language,
        path_prefix=path_prefix,
        path_filters=path_filters,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        case_mode=case_mode,
        word_match=bool(base.word_match),
        fixed_string=bool(base.fixed_string),
        max_files=max_files,
        max_snippets=max_snippets,
        time_budget_ms=time_budget_ms,
    )


def _merge_constraints_with_query_compat(
    constraints: SearchConstraints,
    *,
    parsed: ParsedQueryCompat,
) -> SearchConstraints:
    if parsed.source != "grep_compat" or parsed.fallback_used:
        return constraints
    merged_path_filters = list(constraints.path_filters)
    for candidate in parsed.path_filters:
        if candidate not in merged_path_filters:
            merged_path_filters.append(candidate)
    include_globs = list(constraints.include_globs)
    for candidate in parsed.include_globs:
        if candidate not in include_globs:
            include_globs.append(candidate)
    exclude_globs = list(constraints.exclude_globs)
    for candidate in parsed.exclude_globs:
        if candidate not in exclude_globs:
            exclude_globs.append(candidate)
    return SearchConstraints(
        search_mode=constraints.search_mode,
        language=constraints.language,
        path_prefix=constraints.path_prefix,
        path_filters=tuple(merged_path_filters),
        include_globs=tuple(include_globs),
        exclude_globs=tuple(exclude_globs),
        case_mode=parsed.case_mode if parsed.case_mode is not None else constraints.case_mode,
        word_match=constraints.word_match or parsed.word_match,
        fixed_string=constraints.fixed_string or parsed.fixed_string,
        max_files=constraints.max_files,
        max_snippets=constraints.max_snippets,
        time_budget_ms=constraints.time_budget_ms,
    )


def _span_length(hit: BackendHit) -> int:
    return max(1, hit.end_line - hit.start_line + 1)


def _spans_overlap(left: BackendHit, right: BackendHit) -> bool:
    if left.path != right.path:
        return False
    return not (left.end_line < right.start_line or right.end_line < left.start_line)


def _token_overlap_score(*, text: str, hints: QueryHints) -> float:
    tokens = {token.lower() for token in _TOKEN_RE.findall(text)}
    if not tokens:
        return 0.0
    hint_tokens = set(hints.identifier_tokens)
    if not hint_tokens:
        return 0.0
    overlap = tokens & hint_tokens
    return min(len(overlap) / max(len(hint_tokens), 1), 1.0)


def _path_quality(path: str) -> float:
    lowered = path.replace("\\", "/").lower()
    if "/dist/" in lowered or "/vendor/" in lowered or "/node_modules/" in lowered:
        return 0.0
    if "/src/" in lowered:
        return 1.0
    return 0.7


def _quality_exact(
    result: BackendResult, hints: QueryHints, constraints: SearchConstraints
) -> float:
    if not result.hits:
        return 0.0
    match_strength = max(hit.raw_score for hit in result.hits)
    hit_density = min(len(result.hits) / max(1, constraints.max_snippets or 1), 1.0)
    path_quality = sum(_path_quality(hit.path) for hit in result.hits) / len(result.hits)
    hint_overlap = sum(
        _token_overlap_score(text=f"{hit.path}\n{hit.snippet}", hints=hints) for hit in result.hits
    ) / len(result.hits)
    return max(
        0.0,
        min(
            1.0,
            0.45 * match_strength + 0.25 * hit_density + 0.20 * path_quality + 0.10 * hint_overlap,
        ),
    )


def _quality_symbol(
    result: BackendResult, hints: QueryHints, constraints: SearchConstraints
) -> float:
    if not result.hits:
        return 0.0
    name_match = max(hit.raw_score for hit in result.hits)
    ref_density = min(len(result.hits) / max(1, constraints.max_snippets or 1), 1.0)
    path_quality = sum(_path_quality(hit.path) for hit in result.hits) / len(result.hits)
    hint_overlap = sum(
        _token_overlap_score(text=f"{hit.path}\n{hit.snippet}", hints=hints) for hit in result.hits
    ) / len(result.hits)
    return max(
        0.0,
        min(
            1.0, 0.40 * name_match + 0.25 * ref_density + 0.20 * path_quality + 0.15 * hint_overlap
        ),
    )


def _quality_semantic(result: BackendResult, hints: QueryHints) -> float:
    if not result.hits:
        return 0.0
    ordered = sorted((hit.raw_score for hit in result.hits), reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    margin = max(0.0, min(1.0, top1 - top2))

    directory_counts: dict[str, int] = {}
    for hit in result.hits:
        parent = str(Path(hit.path).parent)
        directory_counts[parent] = directory_counts.get(parent, 0) + 1
    dir_coherence = max(directory_counts.values()) / len(result.hits)

    hint_overlap = sum(
        _token_overlap_score(text=f"{hit.path}\n{hit.snippet}", hints=hints) for hit in result.hits
    ) / len(result.hits)
    return max(
        0.0, min(1.0, 0.45 * top1 + 0.25 * margin + 0.20 * dir_coherence + 0.10 * hint_overlap)
    )


def _compute_quality(
    result: BackendResult,
    *,
    hints: QueryHints,
    constraints: SearchConstraints,
) -> float:
    if result.error:
        return 0.0
    if result.name == "exact":
        return _quality_exact(result, hints, constraints)
    if result.name == "symbol":
        return _quality_symbol(result, hints, constraints)
    return _quality_semantic(result, hints)


def _threshold_for(backend: str, config: SearchRouterConfig) -> float:
    if backend == "exact":
        return config.threshold_exact
    if backend == "symbol":
        return config.threshold_symbol
    return config.threshold_semantic


def _route_auto(results: list[BackendResult], config: SearchRouterConfig) -> RouterOutcome:
    qualified = [
        result
        for result in results
        if result.hits and result.quality_score >= _threshold_for(result.name, config)
    ]
    if qualified:
        chosen = sorted(
            qualified,
            key=lambda item: (
                -item.quality_score,
                _BACKEND_TIE_PRIORITY.get(item.name, 99),
                item.timing_ms,
            ),
        )[0]
        return RouterOutcome(
            strategy=chosen.name,
            reason="quality_threshold_met",
            winner=chosen.name,
            considered=tuple(result.name for result in results),
            backend_scores={result.name: result.quality_score for result in results},
        )

    return RouterOutcome(
        strategy="hybrid",
        reason="no_backend_met_threshold",
        winner=None,
        considered=tuple(result.name for result in results),
        backend_scores={result.name: result.quality_score for result in results},
    )


def _merge_backend_hits(
    results_by_name: dict[str, BackendResult], max_snippets: int
) -> list[BackendHit]:
    """Merge best exact+semantic hits with deterministic fallback."""
    exact_hits = list(results_by_name.get("exact", BackendResult("exact", (), 0)).hits)
    semantic_hits = list(results_by_name.get("semantic", BackendResult("semantic", (), 0)).hits)

    primary_count = max(1, math.ceil(max_snippets * 0.6))
    secondary_count = max(1, max_snippets - primary_count)
    merged = exact_hits[:primary_count] + semantic_hits[:secondary_count]

    if len(merged) < max_snippets:
        symbol_hits = list(results_by_name.get("symbol", BackendResult("symbol", (), 0)).hits)
        for item in symbol_hits:
            if len(merged) >= max_snippets:
                break
            merged.append(item)
    return merged[:max_snippets]


def _pack_hits(
    hits: list[BackendHit],
    *,
    repo_root: Path,
    constraints: SearchConstraints,
    config: SearchRouterConfig,
) -> tuple[ContextHit, ...]:
    """Dedupe and cap context hits with deterministic ordering."""
    sorted_hits = sorted(hits, key=lambda item: (-item.raw_score, item.path, item.start_line))

    deduped: list[BackendHit] = []
    for candidate in sorted_hits:
        replaced = False
        for index, existing in enumerate(deduped):
            if not _spans_overlap(candidate, existing):
                continue
            # Prefer smaller spans, then higher score.
            if _span_length(candidate) < _span_length(existing) or (
                _span_length(candidate) == _span_length(existing)
                and candidate.raw_score > existing.raw_score
            ):
                deduped[index] = candidate
            replaced = True
            break
        if not replaced:
            deduped.append(candidate)

    by_path: dict[str, list[BackendHit]] = {}
    for hit in deduped:
        bucket = by_path.setdefault(hit.path, [])
        bucket.append(hit)
    for bucket in by_path.values():
        bucket.sort(key=lambda item: (-item.raw_score, item.start_line, item.end_line))

    max_files = constraints.max_files or config.max_files
    max_snippets = constraints.max_snippets or config.max_snippets
    selected_files = sorted(by_path.keys(), key=lambda path: -by_path[path][0].raw_score)[
        :max_files
    ]

    ordered: list[BackendHit] = []
    cursor = {path: 0 for path in selected_files}
    while len(ordered) < max_snippets:
        advanced = False
        for path in selected_files:
            index = cursor[path]
            bucket = by_path[path]
            if index >= len(bucket):
                continue
            ordered.append(bucket[index])
            cursor[path] += 1
            advanced = True
            if len(ordered) >= max_snippets:
                break
        if not advanced:
            break

    context_hits: list[ContextHit] = []
    total_chars = 0
    for hit in ordered:
        snippet = hit.snippet.strip()
        if len(snippet) > config.max_snippet_chars:
            snippet = snippet[: config.max_snippet_chars].rstrip() + " ..."

        relative_path = to_repo_relative_path(repo_root, hit.path)
        projected = total_chars + len(relative_path) + len(snippet)
        if context_hits and projected > config.max_chars:
            break
        total_chars = projected
        context_hits.append(
            ContextHit(
                path=relative_path,
                span=ContextSpan(start_line=hit.start_line, end_line=hit.end_line),
                snippet=snippet,
                score=max(0.0, min(1.0, float(hit.raw_score))),
                start_byte=hit.start_byte,
                end_byte=hit.end_byte,
                tags=tuple(sorted(set(hit.tags))),
            )
        )

    return tuple(context_hits)


class SearchRouter:
    """Deterministic retrieval router used by `gloggur search`."""

    def __init__(
        self,
        *,
        repo_root: Path,
        searcher: HybridSearch | None,
        metadata_store: MetadataStore | None,
        config: SearchRouterConfig,
        symbol_store: SymbolIndexStore | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.searcher = searcher
        self.metadata_store = metadata_store
        self.symbol_store = symbol_store
        self.config = config

    def _resolve_backends(self, mode: str) -> tuple[str, ...]:
        enabled = tuple(
            name for name in self.config.enabled_backends if name in {"exact", "semantic", "symbol"}
        )
        if self.searcher is None:
            enabled = tuple(name for name in enabled if name != "semantic")
        if self.symbol_store is None:
            enabled = tuple(name for name in enabled if name != "symbol")
        if mode == "exact":
            return tuple(name for name in enabled if name == "exact") or ("exact",)
        if mode == "semantic":
            return tuple(name for name in enabled if name == "semantic") or ("semantic",)
        if mode == "hybrid":
            pair = tuple(name for name in enabled if name in {"exact", "semantic"})
            return pair if pair else enabled
        return enabled or ("exact", "semantic")

    def search(
        self,
        *,
        query: str,
        constraints: SearchConstraints | None = None,
        mode: str = "auto",
        include_debug: bool = False,
    ) -> ContextPack:
        """Run routed retrieval and return ContextPack v2."""
        started = time.perf_counter()
        normalized_mode = mode.strip().lower() if mode else "auto"
        if normalized_mode not in {"auto", "exact", "semantic", "hybrid"}:
            normalized_mode = "auto"

        resolved_constraints = _normalize_constraints(constraints, self.config)
        parsed_query = parse_query_compat(query)
        effective_query = query
        if (
            parsed_query.source == "grep_compat"
            and not parsed_query.fallback_used
            and isinstance(parsed_query.pattern, str)
            and parsed_query.pattern.strip()
        ):
            effective_query = parsed_query.pattern.strip()
        resolved_constraints = _merge_constraints_with_query_compat(
            resolved_constraints,
            parsed=parsed_query,
        )
        hints = extract_query_hints(effective_query)
        if (
            parsed_query.source == "grep_compat"
            and not parsed_query.fallback_used
            and parsed_query.pattern_quoted
            and isinstance(parsed_query.pattern, str)
            and parsed_query.pattern.strip()
        ):
            quoted_pattern = parsed_query.pattern.strip()
            symbols = list(hints.symbols)
            if quoted_pattern not in symbols:
                symbols.append(quoted_pattern)
            identifier_tokens = list(hints.identifier_tokens)
            lowered = quoted_pattern.lower()
            if lowered not in identifier_tokens:
                identifier_tokens.append(lowered)
            hints = replace(
                hints,
                symbols=tuple(symbols),
                identifier_tokens=tuple(identifier_tokens),
            )
        backend_names = self._resolve_backends(normalized_mode)

        def _run_backend(name: str) -> BackendResult:
            if name == "exact":
                return run_exact_backend(
                    query=effective_query,
                    hints=hints,
                    repo_root=self.repo_root,
                    constraints=resolved_constraints,
                    config=self.config,
                )
            if name == "semantic":
                if self.searcher is None:
                    return BackendResult(
                        name="semantic",
                        hits=(),
                        timing_ms=0,
                        error="semantic backend unavailable",
                    )
                return run_semantic_backend(
                    query=effective_query,
                    searcher=self.searcher,
                    repo_root=self.repo_root,
                    constraints=resolved_constraints,
                    config=self.config,
                )
            if self.symbol_store is None:
                return BackendResult(
                    name="symbol",
                    hits=(),
                    timing_ms=0,
                    error="symbol backend unavailable",
                )
            return run_symbol_backend(
                symbol_store=self.symbol_store,
                hints=hints,
                query=effective_query,
                repo_root=self.repo_root,
                constraints=resolved_constraints,
                config=self.config,
            )

        backend_results: list[BackendResult] = []
        if len(backend_names) == 1:
            backend_results.append(_run_backend(backend_names[0]))
        else:
            timeout_seconds = max((resolved_constraints.time_budget_ms or 1) / 1000.0, 0.1)
            with ThreadPoolExecutor(max_workers=len(backend_names)) as executor:
                futures = {executor.submit(_run_backend, name): name for name in backend_names}
                try:
                    for future in as_completed(futures, timeout=timeout_seconds):
                        backend_results.append(future.result())
                except TimeoutError:
                    for future in futures:
                        future.cancel()

        enriched_results: list[BackendResult] = []
        for result in backend_results:
            quality = _compute_quality(result, hints=hints, constraints=resolved_constraints)
            enriched_results.append(replace(result, quality_score=quality))

        results_by_name = {result.name: result for result in enriched_results}
        outcome: RouterOutcome
        selected_hits: list[BackendHit]
        if normalized_mode == "hybrid":
            outcome = RouterOutcome(
                strategy="hybrid",
                reason="forced_mode",
                winner=None,
                considered=tuple(result.name for result in enriched_results),
                backend_scores={result.name: result.quality_score for result in enriched_results},
            )
            selected_hits = _merge_backend_hits(
                results_by_name, resolved_constraints.max_snippets or self.config.max_snippets
            )
        elif normalized_mode in {"exact", "semantic"}:
            forced = results_by_name.get(normalized_mode)
            if forced is None:
                forced = BackendResult(
                    name=normalized_mode, hits=(), timing_ms=0, error="backend_not_available"
                )
            outcome = RouterOutcome(
                strategy=normalized_mode,
                reason="forced_mode",
                winner=normalized_mode,
                considered=tuple(result.name for result in enriched_results),
                backend_scores={result.name: result.quality_score for result in enriched_results},
            )
            selected_hits = list(forced.hits)
        else:
            outcome = _route_auto(enriched_results, self.config)
            if outcome.strategy == "hybrid":
                selected_hits = _merge_backend_hits(
                    results_by_name,
                    resolved_constraints.max_snippets or self.config.max_snippets,
                )
            else:
                winner_result = results_by_name.get(outcome.strategy)
                selected_hits = list(winner_result.hits if winner_result is not None else ())

        packed_hits = _pack_hits(
            selected_hits,
            repo_root=self.repo_root,
            constraints=resolved_constraints,
            config=self.config,
        )

        total_ms = int((time.perf_counter() - started) * 1000)
        summary: dict[str, object] = {
            "strategy": outcome.strategy,
            "reason": outcome.reason,
            "winner": outcome.winner,
            "hits": len(packed_hits),
        }

        debug_payload: dict[str, object] = {
            "timings": {
                "total_ms": total_ms,
                "backend_ms": {result.name: result.timing_ms for result in enriched_results},
            },
            "thresholds": {
                "exact": self.config.threshold_exact,
                "semantic": self.config.threshold_semantic,
                "symbol": self.config.threshold_symbol,
            },
            "backend_scores": {result.name: result.quality_score for result in enriched_results},
            "backend_errors": {
                result.name: result.error for result in enriched_results if result.error is not None
            },
            "commands": {
                result.name: list(result.commands) for result in enriched_results if result.commands
            },
            "constraints": {
                "search_mode": resolved_constraints.search_mode,
                "language": resolved_constraints.language,
                "path_prefix": resolved_constraints.path_prefix,
                "path_filters": list(resolved_constraints.path_filters),
                "include_globs": list(resolved_constraints.include_globs),
                "exclude_globs": list(resolved_constraints.exclude_globs),
                "case_mode": resolved_constraints.case_mode,
                "word_match": resolved_constraints.word_match,
                "fixed_string": resolved_constraints.fixed_string,
                "max_files": resolved_constraints.max_files,
                "max_snippets": resolved_constraints.max_snippets,
                "time_budget_ms": resolved_constraints.time_budget_ms,
            },
            "parsed_query": parsed_query.to_debug_payload(),
        }

        log_router_event(
            config=self.config,
            repo_root=self.repo_root,
            query=query,
            payload={
                "strategy": outcome.strategy,
                "winner": outcome.winner,
                "reason": outcome.reason,
                "backend_scores": {
                    result.name: result.quality_score for result in enriched_results
                },
                "backend_timings": {result.name: result.timing_ms for result in enriched_results},
                "pack": {
                    "hits": len(packed_hits),
                    "files": len({hit.path for hit in packed_hits}),
                    "chars": sum(len(hit.path) + len(hit.snippet) for hit in packed_hits),
                },
                "duration_ms": total_ms,
            },
        )

        return ContextPack(
            query=query,
            summary=summary,
            hits=packed_hits,
            debug=debug_payload,
        )
