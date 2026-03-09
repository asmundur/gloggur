from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
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
from gloggur.search.router.locality import (
    LocalityAnchor,
    LocalityState,
    build_anchor_hit,
    derive_candidate_test_paths,
    is_source_path,
    is_test_path,
    load_locality_state,
    persist_locality_state,
)
from gloggur.search.router.query_compat import ParsedQueryCompat, parse_query_compat
from gloggur.search.router.telemetry import log_router_event
from gloggur.search.router.types import (
    BackendHit,
    BackendResult,
    ContextHit,
    ContextPack,
    ContextSpan,
    ExecutionHints,
    QueryHints,
    RouterOutcome,
    SearchConstraints,
    SearchIntent,
)
from gloggur.storage.metadata_store import MetadataStore
from gloggur.symbol_index.store import SymbolIndexStore

_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_BACKEND_TIE_PRIORITY = {
    "exact": 0,
    "symbol": 1,
    "semantic": 2,
}
_RANK_FUSION_K = 60
_LOCALITY_NEARBY_RADIUS = 24
_LOCALITY_STRONG_SCORE = 0.9


@dataclass(frozen=True)
class LocalityOutcome:
    """Resolved locality behavior for one router search."""

    mode: str = "none"
    origin: str | None = None
    anchor: LocalityAnchor | None = None
    ambiguity_streak: int = 0
    candidate_test_paths: tuple[str, ...] = ()
    local_hit_counts: dict[str, int] | None = None
    confirmation_suppressed: bool = False
    forced_local_only: bool = False


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


def _normalize_search_mode(value: object) -> str:
    search_mode = (
        str(value).strip().lower() if isinstance(value, str) and str(value).strip() else "semantic"
    )
    if search_mode not in {"semantic", "by_fqname", "by_fqname_regex", "by_path"}:
        return "semantic"
    return search_mode


def _normalize_search_intent(
    intent: SearchIntent | None,
    config: SearchRouterConfig,
) -> SearchIntent:
    """Resolve runtime intent values with config defaults."""
    base = intent or SearchIntent()
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
    path_filters = _normalize_string_tuple(base.path_filters)
    return SearchIntent(
        search_mode=_normalize_search_mode(base.search_mode),
        language=language,
        path_prefix=path_prefix,
        path_filters=path_filters,
        max_files=max_files,
        max_snippets=max_snippets,
        time_budget_ms=time_budget_ms,
    )


def _normalize_execution_hints(hints: ExecutionHints | None) -> ExecutionHints:
    include_globs = _normalize_string_tuple(hints.include_globs if hints is not None else ())
    exclude_globs = _normalize_string_tuple(hints.exclude_globs if hints is not None else ())
    case_mode: str | None = None
    if hints is not None and isinstance(hints.case_mode, str):
        normalized_case = hints.case_mode.strip().lower()
        if normalized_case in {"ignore", "smart"}:
            case_mode = normalized_case
    return ExecutionHints(
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        case_mode=case_mode,
        word_match=bool(hints.word_match) if hints is not None else False,
        fixed_string=bool(hints.fixed_string) if hints is not None else False,
    )


def _intent_and_hints_from_constraints(
    constraints: SearchConstraints | None,
    *,
    config: SearchRouterConfig,
) -> tuple[SearchIntent, ExecutionHints]:
    """Convert compatibility constraints into SearchIntent + internal execution hints."""
    if constraints is None:
        return _normalize_search_intent(None, config), _normalize_execution_hints(None)
    intent = SearchIntent(
        search_mode=constraints.search_mode,
        language=constraints.language,
        path_prefix=constraints.path_prefix,
        path_filters=constraints.path_filters,
        max_files=constraints.max_files,
        max_snippets=constraints.max_snippets,
        time_budget_ms=constraints.time_budget_ms,
    )
    hints = ExecutionHints(
        include_globs=constraints.include_globs,
        exclude_globs=constraints.exclude_globs,
        case_mode=constraints.case_mode,
        word_match=constraints.word_match,
        fixed_string=constraints.fixed_string,
    )
    return _normalize_search_intent(intent, config), _normalize_execution_hints(hints)


def _merge_execution_hints_with_query_compat(
    execution_hints: ExecutionHints,
    *,
    parsed: ParsedQueryCompat,
) -> tuple[ExecutionHints, tuple[str, ...]]:
    merged_path_filters: list[str] = []
    if parsed.source != "grep_compat" or parsed.fallback_used:
        return execution_hints, ()
    merged_path_filters = []
    for candidate in parsed.path_filters:
        if candidate not in merged_path_filters:
            merged_path_filters.append(candidate)
    include_globs = list(execution_hints.include_globs)
    for candidate in parsed.include_globs:
        if candidate not in include_globs:
            include_globs.append(candidate)
    exclude_globs = list(execution_hints.exclude_globs)
    for candidate in parsed.exclude_globs:
        if candidate not in exclude_globs:
            exclude_globs.append(candidate)
    return (
        ExecutionHints(
            include_globs=tuple(include_globs),
            exclude_globs=tuple(exclude_globs),
            case_mode=(
                parsed.case_mode if parsed.case_mode is not None else execution_hints.case_mode
            ),
            word_match=execution_hints.word_match or parsed.word_match,
            fixed_string=execution_hints.fixed_string or parsed.fixed_string,
        ),
        tuple(merged_path_filters),
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


def _quality_exact(result: BackendResult, hints: QueryHints, intent: SearchIntent) -> float:
    if not result.hits:
        return 0.0
    match_strength = max(hit.raw_score for hit in result.hits)
    hit_density = min(len(result.hits) / max(1, intent.max_snippets or 1), 1.0)
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


def _quality_symbol(result: BackendResult, hints: QueryHints, intent: SearchIntent) -> float:
    if not result.hits:
        return 0.0
    name_match = max(hit.raw_score for hit in result.hits)
    ref_density = min(len(result.hits) / max(1, intent.max_snippets or 1), 1.0)
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
    intent: SearchIntent,
) -> float:
    if result.error:
        return 0.0
    if result.name == "exact":
        return _quality_exact(result, hints, intent)
    if result.name == "symbol":
        return _quality_symbol(result, hints, intent)
    return _quality_semantic(result, hints)


def _threshold_for(backend: str, config: SearchRouterConfig) -> float:
    if backend == "exact":
        return config.threshold_exact
    if backend == "symbol":
        return config.threshold_symbol
    return config.threshold_semantic


def _evidence_kind_for_backend(name: str) -> str:
    if name == "exact":
        return "lexical"
    if name == "symbol":
        return "symbol"
    return "semantic"


def _backend_threshold_status(
    results: list[BackendResult],
    config: SearchRouterConfig,
) -> dict[str, dict[str, object]]:
    """Return per-backend threshold/debug status for summary/debug payloads."""
    status: dict[str, dict[str, object]] = {}
    for result in results:
        top_hit_score = max((hit.raw_score for hit in result.hits), default=0.0)
        threshold = _threshold_for(result.name, config)
        status[result.name] = {
            "top_hit_score": top_hit_score,
            "quality_score": result.quality_score,
            "threshold": threshold,
            "threshold_met": bool(result.hits and result.quality_score >= threshold),
            "evidence_kind": _evidence_kind_for_backend(result.name),
        }
    return status


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

    has_non_semantic_evidence = any(
        result.name in {"exact", "symbol"} and result.hits for result in results
    )
    if not has_non_semantic_evidence:
        return RouterOutcome(
            strategy="suppressed",
            reason="semantic_only_below_threshold",
            winner=None,
            considered=tuple(result.name for result in results),
            backend_scores={result.name: result.quality_score for result in results},
            warning_codes=("ungrounded_results_suppressed",),
        )

    return RouterOutcome(
        strategy="hybrid",
        reason="no_backend_met_threshold",
        winner=None,
        considered=tuple(result.name for result in results),
        backend_scores={result.name: result.quality_score for result in results},
    )


def _lexical_support_for_backend(result: BackendResult, *, hints: QueryHints) -> float:
    if not result.hits:
        return 0.0
    overlap = sum(
        _token_overlap_score(text=f"{hit.path}\n{hit.snippet}", hints=hints) for hit in result.hits
    ) / len(result.hits)
    if result.name == "exact":
        tagged = sum(
            1 for hit in result.hits if "literal_match" in hit.tags or "symbol_match" in hit.tags
        )
        tag_signal = tagged / len(result.hits)
        return max(0.0, min(1.0, 0.50 + 0.30 * overlap + 0.20 * tag_signal))
    if result.name == "symbol":
        tagged = sum(
            1 for hit in result.hits if "symbol_def" in hit.tags or "symbol_ref" in hit.tags
        )
        tag_signal = tagged / len(result.hits)
        return max(0.0, min(1.0, 0.42 + 0.35 * overlap + 0.23 * tag_signal))
    tagged = sum(1 for hit in result.hits if "semantic_high_conf" in hit.tags)
    tag_signal = tagged / len(result.hits)
    return max(0.0, min(1.0, 0.25 + 0.55 * overlap + 0.20 * tag_signal))


def _adaptive_backend_weight(
    result: BackendResult,
    *,
    hints: QueryHints,
    intent: SearchIntent,
    threshold_met: bool,
    max_backend_ms: int,
) -> float:
    if not result.hits:
        return 0.0
    hit_density = min(len(result.hits) / max(1, intent.max_snippets or 1), 1.0)
    lexical_support = _lexical_support_for_backend(result, hints=hints)
    timing_penalty = min(max(result.timing_ms / max(1, max_backend_ms), 0.0), 1.0)
    threshold_factor = 1.0 if threshold_met else 0.65
    raw_weight = (
        0.55 * result.quality_score
        + 0.20 * hit_density
        + 0.15 * lexical_support
        + 0.10 * (1.0 - timing_penalty)
    )
    return max(0.0, min(1.0, raw_weight * threshold_factor))


def _adaptive_fusion_selection(
    *,
    results_by_name: dict[str, BackendResult],
    intent: SearchIntent,
    hints: QueryHints,
    config: SearchRouterConfig,
) -> tuple[list[BackendHit], dict[str, int], dict[str, float], dict[str, bool], dict[str, float]]:
    available = {name: result for name, result in results_by_name.items() if result.hits}
    if not available:
        return [], {}, {}, {}, {}

    ordered_backends = sorted(
        available.keys(),
        key=lambda name: (
            _BACKEND_TIE_PRIORITY.get(name, 99),
            available[name].timing_ms,
            name,
        ),
    )
    max_snippets = max(1, intent.max_snippets or config.max_snippets)
    eligibility = {
        name: bool(
            available[name].hits and available[name].quality_score >= _threshold_for(name, config)
        )
        for name in ordered_backends
    }
    max_backend_ms = max((available[name].timing_ms for name in ordered_backends), default=1)
    backend_weights = {
        name: _adaptive_backend_weight(
            available[name],
            hints=hints,
            intent=intent,
            threshold_met=eligibility[name],
            max_backend_ms=max_backend_ms,
        )
        for name in ordered_backends
    }

    allocation = {name: 0 for name in ordered_backends}
    for name in ordered_backends:
        if not eligibility.get(name):
            continue
        if allocation[name] >= len(available[name].hits):
            continue
        if sum(allocation.values()) >= max_snippets:
            break
        allocation[name] += 1

    remaining = max_snippets - sum(allocation.values())
    proportional_shares = {name: 0.0 for name in ordered_backends}
    if remaining > 0:
        contributors = [
            name
            for name in ordered_backends
            if len(available[name].hits) > allocation[name] and backend_weights.get(name, 0.0) > 0.0
        ]
        if not contributors:
            contributors = [
                name for name in ordered_backends if len(available[name].hits) > allocation[name]
            ]
        if contributors:
            total_weight = sum(backend_weights.get(name, 0.0) for name in contributors)
            if total_weight <= 0.0:
                total_weight = float(len(contributors))
            fractional: dict[str, float] = {}
            distributed = 0
            for name in contributors:
                share = (
                    backend_weights.get(name, 0.0) / total_weight
                    if total_weight > 0.0
                    else 1.0 / len(contributors)
                )
                proportional_shares[name] = share
                capacity = len(available[name].hits) - allocation[name]
                whole = min(capacity, int(math.floor(remaining * share)))
                allocation[name] += whole
                distributed += whole
                fractional[name] = (remaining * share) - whole

            leftover = remaining - distributed
            if leftover > 0:
                remainder_order = sorted(
                    contributors,
                    key=lambda name: (
                        -fractional.get(name, 0.0),
                        -backend_weights.get(name, 0.0),
                        _BACKEND_TIE_PRIORITY.get(name, 99),
                        available[name].timing_ms,
                        name,
                    ),
                )
                while leftover > 0:
                    progressed = False
                    for name in remainder_order:
                        if allocation[name] >= len(available[name].hits):
                            continue
                        allocation[name] += 1
                        leftover -= 1
                        progressed = True
                        if leftover <= 0:
                            break
                    if not progressed:
                        break

    ranked_candidates = {
        name: list(available[name].hits)[: allocation.get(name, 0)] for name in ordered_backends
    }
    vote_scores: dict[tuple[str, int, int], float] = {}
    first_seen: dict[tuple[str, int, int], int] = {}
    representatives: dict[tuple[str, int, int], BackendHit] = {}
    order_counter = 0
    for name in ordered_backends:
        candidates = ranked_candidates.get(name, [])
        for rank, hit in enumerate(candidates, start=1):
            key = (hit.path, hit.start_line, hit.end_line)
            vote_scores[key] = vote_scores.get(key, 0.0) + (1.0 / (_RANK_FUSION_K + rank))
            if key not in first_seen:
                first_seen[key] = order_counter
            order_counter += 1
            existing = representatives.get(key)
            if existing is None:
                representatives[key] = hit
                continue
            merged_tags = tuple(sorted(set(existing.tags) | set(hit.tags)))
            preferred = existing if existing.raw_score >= hit.raw_score else hit
            if preferred.tags != merged_tags:
                preferred = replace(preferred, tags=merged_tags)
            representatives[key] = preferred

    ordered_keys = sorted(
        vote_scores,
        key=lambda key: (
            -vote_scores[key],
            first_seen.get(key, 0),
            key[0],
            key[1],
            key[2],
        ),
    )
    selected_hits = [representatives[key] for key in ordered_keys]
    return (
        selected_hits[:max_snippets],
        allocation,
        backend_weights,
        eligibility,
        proportional_shares,
    )


def _suggested_path_prefix(hits: tuple[ContextHit, ...]) -> str | None:
    top_hits = hits[:4]
    if len(top_hits) < 2:
        return None

    path_counts: dict[str, int] = {}
    for hit in top_hits:
        path_counts[hit.path] = path_counts.get(hit.path, 0) + 1
    path, path_count = sorted(
        path_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    if path_count >= 2:
        return path

    dir_counts: dict[str, int] = {}
    for hit in top_hits:
        parent = str(Path(hit.path).parent)
        if parent in {"", "."}:
            continue
        dir_counts[parent] = dir_counts.get(parent, 0) + 1
    if not dir_counts:
        return None
    parent, parent_count = sorted(
        dir_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    if parent_count >= 2:
        return parent
    return None


def _decisive_non_semantic_result(
    *,
    hits: tuple[ContextHit, ...],
    outcome: RouterOutcome,
    backend_thresholds: dict[str, dict[str, object]],
    query_kind: str,
    locality: LocalityOutcome,
    repo_root: Path,
) -> bool:
    if outcome.strategy not in {"exact", "symbol"} or not hits:
        return False
    winner_status = backend_thresholds.get(outcome.strategy, {})
    if not bool(winner_status.get("threshold_met")):
        return False

    if (
        locality.anchor is not None
        and locality.mode == "reused"
        and hits[0].path == to_repo_relative_path(repo_root, locality.anchor.path)
    ):
        if locality.forced_local_only or locality.local_hit_counts is None:
            return True
        if int(locality.local_hit_counts.get("nonlocal", 0)) <= 0:
            return True

    top_score = hits[0].score
    if len(hits) == 1:
        return top_score >= 0.80

    second_score = hits[1].score
    if hits[0].path == hits[1].path and abs(top_score - second_score) < 0.05:
        if query_kind in {"identifier", "declaration"} and _hit_has_definition_signal(hits[0].tags):
            return True
        return False
    if query_kind in {"identifier", "declaration"}:
        return top_score >= 0.85 and (
            top_score - second_score >= 0.05 or hits[0].path != hits[1].path
        )
    return top_score >= 0.90 and top_score - second_score >= 0.10


def _resolve_next_action(
    *,
    hits: tuple[ContextHit, ...],
    outcome: RouterOutcome,
    backend_thresholds: dict[str, dict[str, object]],
    query_kind: str,
    locality: LocalityOutcome,
    repo_root: Path,
) -> tuple[bool, str, str | None]:
    decisive = _decisive_non_semantic_result(
        hits=hits,
        outcome=outcome,
        backend_thresholds=backend_thresholds,
        query_kind=query_kind,
        locality=locality,
        repo_root=repo_root,
    )
    if decisive and not (locality.mode == "derived" and query_kind == "mixed"):
        return True, "open_hit_1", None

    if locality.anchor is not None and locality.mode in {"derived", "reused"} and hits:
        anchor_path = to_repo_relative_path(repo_root, locality.anchor.path)
        if hits[0].path == anchor_path and locality.forced_local_only:
            return True, "open_hit_1", None

    suggested_path_prefix = _suggested_path_prefix(hits)
    if suggested_path_prefix is not None:
        return False, "narrow_by_path", suggested_path_prefix
    return False, "refine_query", None


def _pack_hits(
    hits: list[BackendHit],
    *,
    repo_root: Path,
    intent: SearchIntent,
    config: SearchRouterConfig,
    preserve_ranked_order: bool = False,
) -> tuple[ContextHit, ...]:
    """Dedupe and cap context hits with deterministic ordering."""
    sorted_hits = (
        list(hits)
        if preserve_ranked_order
        else sorted(hits, key=lambda item: (-item.raw_score, item.path, item.start_line))
    )

    deduped: list[BackendHit] = []
    for candidate in sorted_hits:
        replaced = False
        for index, existing in enumerate(deduped):
            if not _spans_overlap(candidate, existing):
                continue
            if preserve_ranked_order:
                # Keep first-seen hit when order is already pre-ranked by fusion.
                pass
            else:
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
    if not preserve_ranked_order:
        for bucket in by_path.values():
            bucket.sort(key=lambda item: (-item.raw_score, item.start_line, item.end_line))

    max_files = intent.max_files or config.max_files
    max_snippets = intent.max_snippets or config.max_snippets
    ordered: list[BackendHit]
    if preserve_ranked_order:
        ordered = []
        selected_files: list[str] = []
        selected_file_set: set[str] = set()
        for hit in deduped:
            path = hit.path
            if path not in selected_file_set:
                if len(selected_files) >= max_files:
                    continue
                selected_files.append(path)
                selected_file_set.add(path)
            ordered.append(hit)
            if len(ordered) >= max_snippets:
                break
    else:
        selected_files = sorted(by_path.keys(), key=lambda path: -by_path[path][0].raw_score)[
            :max_files
        ]
        ordered = []
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


def _hit_has_definition_signal(tags: tuple[str, ...]) -> bool:
    return any(
        tag in tags for tag in ("symbol_def", "literal_match", "symbol_match", "locality_anchor")
    )


def _derive_locality_anchor(
    *,
    hits: list[BackendHit],
    outcome: RouterOutcome,
    query_kind: str,
) -> LocalityAnchor | None:
    if not hits or outcome.strategy not in {"exact", "symbol"}:
        return None

    scored_candidates: list[tuple[float, BackendHit]] = []
    for hit in hits:
        if not is_source_path(hit.path):
            continue
        score = hit.raw_score
        if _hit_has_definition_signal(hit.tags):
            score += 0.12
        if hit.backend in {"exact", "symbol"}:
            score += 0.06
        if query_kind in {"identifier", "declaration"}:
            score += 0.04
        scored_candidates.append((score, hit))
    if not scored_candidates:
        return None

    anchor_score, anchor_hit = sorted(
        scored_candidates,
        key=lambda item: (-item[0], item[1].path, item[1].start_line, item[1].end_line),
    )[0]
    minimum_score = 0.74 if query_kind in {"identifier", "declaration"} else 0.80
    if anchor_score < minimum_score:
        return None
    return LocalityAnchor(
        path=anchor_hit.path,
        start_line=anchor_hit.start_line,
        end_line=anchor_hit.end_line,
        score=max(0.0, min(1.0, anchor_hit.raw_score)),
        backend=anchor_hit.backend,
        tags=anchor_hit.tags,
    )


def _same_span(hit: BackendHit, anchor: LocalityAnchor) -> bool:
    return (
        hit.path == anchor.path
        and hit.start_line == anchor.start_line
        and hit.end_line == anchor.end_line
    )


def _same_file(hit: BackendHit, anchor: LocalityAnchor) -> bool:
    return hit.path == anchor.path


def _span_distance(hit: BackendHit, anchor: LocalityAnchor) -> int:
    if hit.path != anchor.path:
        return 1_000_000
    if hit.end_line < anchor.start_line:
        return anchor.start_line - hit.end_line
    if anchor.end_line < hit.start_line:
        return hit.start_line - anchor.end_line
    return 0


def _is_near_anchor(hit: BackendHit, anchor: LocalityAnchor) -> bool:
    return _same_file(hit, anchor) and _span_distance(hit, anchor) <= _LOCALITY_NEARBY_RADIUS


def _is_local_neighborhood_hit(
    hit: BackendHit,
    *,
    anchor: LocalityAnchor,
    candidate_test_paths: tuple[str, ...],
) -> bool:
    normalized_tests = {path.replace("\\", "/") for path in candidate_test_paths}
    hit_path = hit.path.replace("\\", "/")
    return (
        _same_file(hit, anchor)
        or _is_near_anchor(hit, anchor)
        or (is_test_path(hit.path) and hit_path in normalized_tests)
    )


def _locality_adjusted_score(
    hit: BackendHit,
    *,
    anchor: LocalityAnchor,
    candidate_test_paths: tuple[str, ...],
    confirmation_penalty: bool,
) -> float:
    score = hit.raw_score
    if _same_file(hit, anchor):
        score += 0.35
    if _is_near_anchor(hit, anchor):
        score += 0.12
    if is_test_path(hit.path) and hit.path.replace("\\", "/") in {
        path.replace("\\", "/") for path in candidate_test_paths
    }:
        score += 0.18
    elif is_test_path(hit.path):
        score -= 0.04
    if confirmation_penalty and not _is_local_neighborhood_hit(
        hit,
        anchor=anchor,
        candidate_test_paths=candidate_test_paths,
    ):
        score -= 0.38
    return score


def _has_local_non_semantic_hits(
    results: list[BackendResult],
    *,
    anchor: LocalityAnchor | None,
) -> bool:
    if anchor is None:
        return False
    for result in results:
        if result.name not in {"exact", "symbol"}:
            continue
        for hit in result.hits:
            if _same_file(hit, anchor) or is_test_path(hit.path):
                return True
    return False


def _apply_locality_mode(
    *,
    selected_hits: list[BackendHit],
    enriched_results: list[BackendResult],
    hints: QueryHints,
    outcome: RouterOutcome,
    repo_root: Path,
    config: SearchRouterConfig,
    persisted_anchor: LocalityAnchor | None,
    persisted_ambiguity_streak: int,
    allow_reuse: bool,
) -> tuple[list[BackendHit], LocalityOutcome]:
    all_hits: list[BackendHit] = []
    for result in enriched_results:
        all_hits.extend(result.hits)

    derived_anchor = _derive_locality_anchor(
        hits=selected_hits or all_hits,
        outcome=outcome,
        query_kind=hints.query_kind,
    )
    anchor = derived_anchor
    locality_mode = "derived" if derived_anchor is not None else "none"
    locality_origin = "current_query" if derived_anchor is not None else None
    ambiguity_streak = 0
    if anchor is None and allow_reuse and persisted_anchor is not None:
        anchor = persisted_anchor
        locality_mode = "reused"
        locality_origin = "persisted"
        ambiguity_streak = max(0, persisted_ambiguity_streak)
    if anchor is None:
        return selected_hits, LocalityOutcome()

    observed_test_paths = tuple(
        sorted({hit.path.replace("\\", "/") for hit in all_hits if is_test_path(hit.path)})
    )
    candidate_test_paths = derive_candidate_test_paths(
        repo_root=repo_root,
        anchor_path=anchor.path,
        observed_test_paths=observed_test_paths,
        query_tokens=hints.identifier_tokens,
    )
    anchor_hit = next(
        (hit for hit in [*selected_hits, *all_hits] if _same_span(hit, anchor)),
        None,
    )
    if anchor_hit is None:
        anchor_hit = build_anchor_hit(anchor=anchor, repo_root=repo_root, config=config)

    seen_keys: set[tuple[str, int, int, str]] = set()
    candidates: list[BackendHit] = []
    for hit in [anchor_hit, *selected_hits, *all_hits]:
        key = (hit.path, hit.start_line, hit.end_line, hit.backend)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(hit)

    selected_nonlocal = any(
        not _is_local_neighborhood_hit(
            hit,
            anchor=anchor,
            candidate_test_paths=candidate_test_paths,
        )
        for hit in selected_hits[:4]
    )
    if locality_mode == "reused" and selected_nonlocal:
        ambiguity_streak += 1
    elif locality_mode == "reused":
        ambiguity_streak = 0

    confirmation_suppressed = locality_mode == "reused" and selected_nonlocal
    forced_local_only = confirmation_suppressed and ambiguity_streak >= 2

    ranked = sorted(
        candidates,
        key=lambda hit: (
            -_locality_adjusted_score(
                hit,
                anchor=anchor,
                candidate_test_paths=candidate_test_paths,
                confirmation_penalty=confirmation_suppressed,
            ),
            hit.path,
            hit.start_line,
            hit.end_line,
        ),
    )
    if forced_local_only:
        ranked = [
            hit
            for hit in ranked
            if _is_local_neighborhood_hit(
                hit,
                anchor=anchor,
                candidate_test_paths=candidate_test_paths,
            )
        ]
        if not ranked:
            ranked = [anchor_hit]

    local_counts = {
        "same_file": sum(1 for hit in ranked if _same_file(hit, anchor)),
        "nearby_same_file": sum(1 for hit in ranked if _is_near_anchor(hit, anchor)),
        "candidate_tests": sum(
            1
            for hit in ranked
            if is_test_path(hit.path)
            and hit.path.replace("\\", "/")
            in {path.replace("\\", "/") for path in candidate_test_paths}
        ),
        "nonlocal": sum(
            1
            for hit in ranked
            if not _is_local_neighborhood_hit(
                hit,
                anchor=anchor,
                candidate_test_paths=candidate_test_paths,
            )
        ),
    }
    return ranked, LocalityOutcome(
        mode=locality_mode,
        origin=locality_origin,
        anchor=anchor,
        ambiguity_streak=ambiguity_streak,
        candidate_test_paths=candidate_test_paths,
        local_hit_counts=local_counts,
        confirmation_suppressed=confirmation_suppressed,
        forced_local_only=forced_local_only,
    )


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
        searcher_factory: Callable[[], tuple[HybridSearch | None, str | None]] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.searcher = searcher
        self.metadata_store = metadata_store
        self.symbol_store = symbol_store
        self.config = config
        self._searcher_factory = searcher_factory
        self._searcher_initialized = searcher is not None
        self._semantic_init_error: str | None = None

    def _semantic_backend_configured(self) -> bool:
        return self.searcher is not None or self._searcher_factory is not None

    def _get_searcher(self) -> tuple[HybridSearch | None, str | None]:
        if self.searcher is not None:
            return self.searcher, self._semantic_init_error
        if self._searcher_initialized:
            return None, self._semantic_init_error

        self._searcher_initialized = True
        if self._searcher_factory is None:
            self._semantic_init_error = "semantic backend unavailable"
            return None, self._semantic_init_error

        try:
            searcher, init_error = self._searcher_factory()
        except Exception as exc:
            self._semantic_init_error = str(exc)
            return None, self._semantic_init_error

        self.searcher = searcher
        self._semantic_init_error = init_error
        if self.searcher is None and self._semantic_init_error is None:
            self._semantic_init_error = "semantic backend unavailable"
        return self.searcher, self._semantic_init_error

    def _resolve_backends(self, mode: str) -> tuple[str, ...]:
        enabled = tuple(
            name for name in self.config.enabled_backends if name in {"exact", "semantic", "symbol"}
        )
        if not self._semantic_backend_configured():
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
        return enabled or ("exact",)

    def search(
        self,
        *,
        query: str,
        intent: SearchIntent | None = None,
        constraints: SearchConstraints | None = None,
        mode: str = "auto",
        include_debug: bool = False,
    ) -> ContextPack:
        """Run routed retrieval and return ContextPack v2."""
        started = time.perf_counter()
        normalized_mode = mode.strip().lower() if mode else "auto"
        if normalized_mode not in {"auto", "exact", "semantic", "hybrid"}:
            normalized_mode = "auto"

        if intent is not None:
            resolved_intent = _normalize_search_intent(intent, self.config)
            resolved_execution_hints = _normalize_execution_hints(None)
        else:
            resolved_intent, resolved_execution_hints = _intent_and_hints_from_constraints(
                constraints,
                config=self.config,
            )
        parsed_query = parse_query_compat(query)
        effective_query = query
        if (
            parsed_query.source == "grep_compat"
            and not parsed_query.fallback_used
            and isinstance(parsed_query.pattern, str)
            and parsed_query.pattern.strip()
        ):
            effective_query = parsed_query.pattern.strip()
        (
            resolved_execution_hints,
            parsed_path_filters,
        ) = _merge_execution_hints_with_query_compat(
            resolved_execution_hints,
            parsed=parsed_query,
        )
        if parsed_path_filters:
            merged_path_filters = list(resolved_intent.path_filters)
            for candidate in parsed_path_filters:
                if candidate not in merged_path_filters:
                    merged_path_filters.append(candidate)
            resolved_intent = replace(
                resolved_intent,
                path_filters=tuple(merged_path_filters),
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
        explicit_regex_requested = (
            parsed_query.source == "grep_compat"
            and not parsed_query.fallback_used
            and not parsed_query.fixed_string
        )
        backend_execution_hints = resolved_execution_hints
        if hints.query_kind in {"identifier", "declaration"} and not explicit_regex_requested:
            backend_execution_hints = replace(resolved_execution_hints, fixed_string=True)
        backend_names = self._resolve_backends(normalized_mode)
        reuse_locality_allowed = (
            normalized_mode == "auto"
            and not resolved_intent.path_prefix
            and not resolved_intent.path_filters
        )
        persisted_locality_state = (
            load_locality_state(repo_root=self.repo_root, config=self.config)
            if reuse_locality_allowed
            else None
        )

        def _run_backend(name: str) -> BackendResult:
            if name == "exact":
                return run_exact_backend(
                    query=effective_query,
                    hints=hints,
                    repo_root=self.repo_root,
                    intent=resolved_intent,
                    execution_hints=backend_execution_hints,
                    config=self.config,
                )
            if name == "semantic":
                searcher, semantic_init_error = self._get_searcher()
                if searcher is None:
                    return BackendResult(
                        name="semantic",
                        hits=(),
                        timing_ms=0,
                        error=semantic_init_error or "semantic backend unavailable",
                    )
                return run_semantic_backend(
                    query=effective_query,
                    searcher=searcher,
                    repo_root=self.repo_root,
                    intent=resolved_intent,
                    execution_hints=backend_execution_hints,
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
                intent=resolved_intent,
                execution_hints=backend_execution_hints,
                config=self.config,
            )

        backend_results: list[BackendResult] = []
        if normalized_mode == "hybrid":
            if len(backend_names) == 1:
                backend_results.append(_run_backend(backend_names[0]))
            else:
                timeout_seconds = max((resolved_intent.time_budget_ms or 1) / 1000.0, 0.1)
                with ThreadPoolExecutor(max_workers=len(backend_names)) as executor:
                    futures = {executor.submit(_run_backend, name): name for name in backend_names}
                    try:
                        for future in as_completed(futures, timeout=timeout_seconds):
                            backend_results.append(future.result())
                    except TimeoutError:
                        for future in futures:
                            future.cancel()
        elif normalized_mode in {"exact", "semantic"}:
            backend_results.append(_run_backend(normalized_mode))
        else:
            non_semantic_results: list[BackendResult] = []
            for backend_name in ("exact", "symbol"):
                if backend_name in backend_names:
                    non_semantic_results.append(_run_backend(backend_name))
            backend_results.extend(non_semantic_results)

            preliminary_results = [
                replace(
                    result,
                    quality_score=_compute_quality(
                        result,
                        hints=hints,
                        intent=resolved_intent,
                    ),
                )
                for result in non_semantic_results
            ]
            preliminary_outcome = _route_auto(preliminary_results, self.config)
            has_non_semantic_hits = any(result.hits for result in non_semantic_results)
            needs_semantic_fallback = False
            if "semantic" in backend_names:
                if hints.query_kind in {"identifier", "declaration"}:
                    needs_semantic_fallback = not has_non_semantic_hits
                else:
                    needs_semantic_fallback = preliminary_outcome.strategy in {
                        "hybrid",
                        "suppressed",
                    }
                    if needs_semantic_fallback and _has_local_non_semantic_hits(
                        non_semantic_results,
                        anchor=(
                            persisted_locality_state.anchor
                            if persisted_locality_state is not None
                            else None
                        ),
                    ):
                        needs_semantic_fallback = False
            if needs_semantic_fallback:
                backend_results.append(_run_backend("semantic"))

        enriched_results: list[BackendResult] = []
        for result in backend_results:
            quality = _compute_quality(result, hints=hints, intent=resolved_intent)
            enriched_results.append(replace(result, quality_score=quality))

        results_by_name = {result.name: result for result in enriched_results}
        max_backend_ms = max((result.timing_ms for result in enriched_results), default=1)
        eligibility = {
            result.name: bool(
                result.hits and result.quality_score >= _threshold_for(result.name, self.config)
            )
            for result in enriched_results
        }
        backend_weights = {
            result.name: _adaptive_backend_weight(
                result,
                hints=hints,
                intent=resolved_intent,
                threshold_met=eligibility[result.name],
                max_backend_ms=max_backend_ms,
            )
            for result in enriched_results
        }
        fusion_allocation: dict[str, int] = {}
        fusion_proportional_shares: dict[str, float] = {}
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
            (
                selected_hits,
                fusion_allocation,
                backend_weights,
                eligibility,
                fusion_proportional_shares,
            ) = _adaptive_fusion_selection(
                results_by_name=results_by_name,
                intent=resolved_intent,
                hints=hints,
                config=self.config,
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
                (
                    selected_hits,
                    fusion_allocation,
                    backend_weights,
                    eligibility,
                    fusion_proportional_shares,
                ) = _adaptive_fusion_selection(
                    results_by_name=results_by_name,
                    intent=resolved_intent,
                    hints=hints,
                    config=self.config,
                )
            elif outcome.strategy == "suppressed":
                selected_hits = []
            else:
                winner_result = results_by_name.get(outcome.strategy)
                selected_hits = list(winner_result.hits if winner_result is not None else ())

        selected_hits, locality = _apply_locality_mode(
            selected_hits=selected_hits,
            enriched_results=enriched_results,
            hints=hints,
            outcome=outcome,
            repo_root=self.repo_root,
            config=self.config,
            persisted_anchor=(
                persisted_locality_state.anchor if persisted_locality_state is not None else None
            ),
            persisted_ambiguity_streak=(
                persisted_locality_state.ambiguity_streak
                if persisted_locality_state is not None
                else 0
            ),
            allow_reuse=reuse_locality_allowed,
        )
        if locality.anchor is not None:
            persist_locality_state(
                repo_root=self.repo_root,
                config=self.config,
                state=LocalityState(
                    anchor=locality.anchor,
                    updated_at_epoch=time.time(),
                    ambiguity_streak=locality.ambiguity_streak,
                ),
            )

        packed_hits = _pack_hits(
            selected_hits,
            repo_root=self.repo_root,
            intent=resolved_intent,
            config=self.config,
            preserve_ranked_order=outcome.strategy == "hybrid",
        )

        total_ms = int((time.perf_counter() - started) * 1000)
        backend_thresholds = _backend_threshold_status(enriched_results, self.config)
        warning_codes = list(outcome.warning_codes)
        requested_snippets = resolved_intent.max_snippets or self.config.max_snippets
        if (
            hints.query_kind in {"identifier", "declaration"}
            and requested_snippets > 8
            and "identifier_query_high_top_k" not in warning_codes
        ):
            warning_codes.append("identifier_query_high_top_k")
        if locality.mode in {"derived", "reused"} and "locality_mode_active" not in warning_codes:
            warning_codes.append("locality_mode_active")
        if (
            locality.confirmation_suppressed
            and "confirmation_query_downranked" not in warning_codes
        ):
            warning_codes.append("confirmation_query_downranked")
        if locality.ambiguity_streak > 0 and "ambiguous_followup_detected" not in warning_codes:
            warning_codes.append("ambiguous_followup_detected")
        decisive, next_action, suggested_path_prefix = _resolve_next_action(
            hits=packed_hits,
            outcome=outcome,
            backend_thresholds=backend_thresholds,
            query_kind=hints.query_kind,
            locality=locality,
            repo_root=self.repo_root,
        )
        summary: dict[str, object] = {
            "strategy": outcome.strategy,
            "reason": outcome.reason,
            "winner": outcome.winner,
            "hits": len(packed_hits),
            "warning_codes": warning_codes,
            "backend_thresholds": backend_thresholds,
            "query_kind": hints.query_kind,
            "decisive": decisive,
            "next_action": next_action,
            "locality_mode": locality.mode,
        }
        if suggested_path_prefix is not None:
            summary["suggested_path_prefix"] = suggested_path_prefix
        if locality.anchor is not None:
            summary["anchor_path"] = to_repo_relative_path(self.repo_root, locality.anchor.path)

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
            "backend_thresholds": backend_thresholds,
            "backend_scores": {result.name: result.quality_score for result in enriched_results},
            "backend_errors": {
                result.name: result.error for result in enriched_results if result.error is not None
            },
            "commands": {
                result.name: list(result.commands) for result in enriched_results if result.commands
            },
            "backend_weights": backend_weights,
            "fusion_budget": {
                "max_snippets": resolved_intent.max_snippets or self.config.max_snippets,
                "allocation": fusion_allocation,
                "proportional_shares": fusion_proportional_shares,
            },
            "eligibility": eligibility,
            "constraints": {
                "search_mode": resolved_intent.search_mode,
                "language": resolved_intent.language,
                "path_prefix": resolved_intent.path_prefix,
                "path_filters": list(resolved_intent.path_filters),
                "include_globs": list(resolved_execution_hints.include_globs),
                "exclude_globs": list(resolved_execution_hints.exclude_globs),
                "case_mode": backend_execution_hints.case_mode,
                "word_match": backend_execution_hints.word_match,
                "fixed_string": backend_execution_hints.fixed_string,
                "max_files": resolved_intent.max_files,
                "max_snippets": resolved_intent.max_snippets,
                "time_budget_ms": resolved_intent.time_budget_ms,
            },
            "query_kind": hints.query_kind,
            "declaration_terms": list(hints.declaration_terms),
            "parsed_query": parsed_query.to_debug_payload(),
            "locality": {
                "mode": locality.mode,
                "origin": locality.origin,
                "ambiguity_streak": locality.ambiguity_streak,
                "confirmation_suppressed": locality.confirmation_suppressed,
                "forced_local_only": locality.forced_local_only,
                "candidate_test_paths": [
                    to_repo_relative_path(self.repo_root, path)
                    for path in locality.candidate_test_paths
                ],
                "local_hit_counts": locality.local_hit_counts or {},
                "anchor": (
                    locality.anchor.to_payload(repo_root=self.repo_root)
                    if locality.anchor is not None
                    else None
                ),
            },
        }

        log_router_event(
            config=self.config,
            repo_root=self.repo_root,
            query=query,
            payload={
                "query_kind": hints.query_kind,
                "strategy": outcome.strategy,
                "winner": outcome.winner,
                "reason": outcome.reason,
                "next_action": next_action,
                "decisive": decisive,
                "backend_scores": {
                    result.name: result.quality_score for result in enriched_results
                },
                "backend_weights": backend_weights,
                "backend_timings": {result.name: result.timing_ms for result in enriched_results},
                "fusion_budget": {
                    "allocation": fusion_allocation,
                    "proportional_shares": fusion_proportional_shares,
                    "eligibility": eligibility,
                },
                "pack": {
                    "hits": len(packed_hits),
                    "files": len({hit.path for hit in packed_hits}),
                    "chars": sum(len(hit.path) + len(hit.snippet) for hit in packed_hits),
                },
                "locality": {
                    "mode": locality.mode,
                    "origin": locality.origin,
                    "ambiguity_streak": locality.ambiguity_streak,
                    "confirmation_suppressed": locality.confirmation_suppressed,
                    "forced_local_only": locality.forced_local_only,
                    "anchor_path": (
                        to_repo_relative_path(self.repo_root, locality.anchor.path)
                        if locality.anchor is not None
                        else None
                    ),
                    "candidate_test_paths": [
                        to_repo_relative_path(self.repo_root, path)
                        for path in locality.candidate_test_paths
                    ],
                    "local_hit_counts": locality.local_hit_counts or {},
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
