from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import wrap_embedding_error
from gloggur.storage.metadata_store import MetadataStore
from gloggur.storage.vector_store import VectorStore


@dataclass
class SearchResult:
    """Dataclass for search hits: symbol_id and similarity_score."""

    symbol_id: str
    similarity_score: float


@dataclass(frozen=True)
class QueryIntent:
    """Parsed query intent signals used by ranking heuristics."""

    is_identifier_query: bool
    identifier_tail: str | None
    explicit_test_intent: bool


@dataclass(frozen=True)
class RankingPolicy:
    """Scoring deltas for one ranking mode."""

    exact_identifier_tail_bonus: float
    source_path_bonus: float
    test_path_penalty: float
    test_symbol_penalty: float


@dataclass(frozen=True)
class RankingContext:
    """Resolved ranking context for one query execution."""

    ranking_mode: str
    query_intent: QueryIntent
    explicit_test_intent: bool
    test_penalty_applied: bool
    policy: RankingPolicy


RANKING_MODE_BALANCED = "balanced"
RANKING_MODE_SOURCE_FIRST = "source-first"
_RANKING_MODES = {RANKING_MODE_BALANCED, RANKING_MODE_SOURCE_FIRST}
_IDENTIFIER_QUERY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:[.:#][A-Za-z_][A-Za-z0-9_]*)+$")
_TEST_INTENT_RE = re.compile(r"(?i)\b(test|tests|pytest|fixture|unittest|spec)\b")
_TEST_SYMBOL_NAME_RE = re.compile(r"^test(_|$)")
_UNFILTERED_BASE_FETCH_MULTIPLIER = 2
_BALANCED_IDENTIFIER_FETCH_MULTIPLIER = 8
_SOURCE_FIRST_FETCH_MULTIPLIER = 10
_MAX_EXPANDED_CANDIDATES = 250
_DEFAULT_CONTEXT_RADIUS = 12
_FILE_FILTER_MATCH_MODE = "exact_or_prefix"
_FILE_FILTER_NO_MATCH_WARNING = "file_filter_no_match"
_BALANCED_POLICY = RankingPolicy(
    exact_identifier_tail_bonus=0.22,
    source_path_bonus=0.06,
    test_path_penalty=-0.20,
    test_symbol_penalty=-0.08,
)
_SOURCE_FIRST_POLICY = RankingPolicy(
    exact_identifier_tail_bonus=0.22,
    source_path_bonus=0.12,
    test_path_penalty=-0.35,
    test_symbol_penalty=-0.15,
)


class HybridSearch:
    """Hybrid search using embeddings, vector store, and metadata store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        """Initialize search with embedding, vector, and metadata stores."""
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def search(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int = 10,
        context_radius: int = _DEFAULT_CONTEXT_RADIUS,
    ) -> dict[str, object]:
        """Search for symbols matching the query and filters."""
        start = time.time()
        filters = filters or {}
        try:
            query_vector = self.embedding_provider.embed_text(query)
        except Exception as exc:
            raise wrap_embedding_error(
                exc,
                provider=getattr(self.embedding_provider, "provider", "unknown"),
                operation="embed query for search",
            ) from exc
        ranking_context = self._build_ranking_context(query, filters)
        metadata_filters = self._metadata_filters(filters)
        if metadata_filters:
            results = self._search_filtered(
                query_vector,
                metadata_filters,
                top_k,
                ranking_context=ranking_context,
                context_radius=context_radius,
            )
        else:
            results = self._search_unfiltered(
                query_vector,
                top_k,
                ranking_context=ranking_context,
                context_radius=context_radius,
            )
        duration_ms = int((time.time() - start) * 1000)
        metadata = {
            "total_results": len(results),
            "search_time_ms": duration_ms,
            "ranking_mode": ranking_context.ranking_mode,
            "query_intent": self._query_intent_label(ranking_context.query_intent),
            "explicit_test_intent": ranking_context.explicit_test_intent,
            "test_penalty_applied": ranking_context.test_penalty_applied,
            "context_radius": context_radius,
        }
        metadata.update(self._file_filter_metadata(filters=filters, results=results))
        return {
            "query": query,
            "results": results,
            "metadata": metadata,
        }

    def get_semantic_neighborhood(
        self,
        symbol_id: str,
        radius: int = 5,
        top_k: int = 5,
    ) -> dict[str, object]:
        """Get structural and semantic neighbors for a given symbol."""
        start = time.time()

        symbol = self.metadata_store.get_symbol(symbol_id)
        if not symbol:
            return {"error": f"Symbol not found: {symbol_id}"}

        # 1. Structural neighbors (same file)
        structural: list[dict[str, object]] = []
        file_symbols = self.metadata_store.filter_symbols(file_path=symbol.file_path)
        for s in file_symbols:
            if s.id == symbol_id:
                continue
            # Calculate distance in lines
            line_distance = min(
                abs(s.start_line - symbol.end_line), abs(symbol.start_line - s.end_line)
            )
            # If they overlap or enclose, distance is 0
            if s.start_line <= symbol.end_line and s.end_line >= symbol.start_line:
                line_distance = 0

            if line_distance <= radius * 10:  # heuristic radius multiplier for lines
                structural.append(self._serialize_result(s, similarity_score=1.0))

        # 2. Semantic neighbors (vector search)
        semantic: list[dict[str, object]] = []
        if symbol.embedding_vector:
            # We use unfiltered search to find conceptually related symbols across the codebase
            vector_results = self._search_unfiltered(
                symbol.embedding_vector,
                top_k=top_k + 1,
                ranking_context=self._build_ranking_context("", {}),
                context_radius=_DEFAULT_CONTEXT_RADIUS,
            )
            for res in vector_results:
                if res["symbol_id"] != symbol_id:
                    semantic.append(res)
                    if len(semantic) >= top_k:
                        break

        duration_ms = int((time.time() - start) * 1000)
        return {
            "symbol_id": symbol_id,
            "structural_neighbors": structural,
            "semantic_neighbors": semantic,
            "metadata": {
                "structural_count": len(structural),
                "semantic_count": len(semantic),
                "compute_time_ms": duration_ms,
            },
        }

    @classmethod
    def build_ranking_metadata(
        cls,
        query: str,
        filters: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """Return ranking metadata fields for one query/filter combination."""
        ranking_context = cls._build_ranking_context(query, filters or {})
        file_filter: str | None = None
        if isinstance((filters or {}).get("file"), str):
            raw_file_filter = str((filters or {}).get("file") or "").strip()
            if raw_file_filter:
                file_filter = raw_file_filter
        return {
            "ranking_mode": ranking_context.ranking_mode,
            "query_intent": cls._query_intent_label(ranking_context.query_intent),
            "explicit_test_intent": ranking_context.explicit_test_intent,
            "test_penalty_applied": ranking_context.test_penalty_applied,
            "file_filter": file_filter,
            "file_filter_match_mode": _FILE_FILTER_MATCH_MODE,
            "file_filter_warning_codes": [],
        }

    def _search_unfiltered(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        ranking_context: RankingContext,
        context_radius: int,
    ) -> list[dict[str, object]]:
        """Search via vector index without any filters."""
        if top_k < 1:
            return []
        candidate_count = self._candidate_fetch_size(top_k, ranking_context)
        hits = self.vector_store.search(query_vector, k=candidate_count)
        scored: list[tuple[float, float, str, int, str, object]] = []
        for symbol_id, distance in hits:
            symbol = self.metadata_store.get_symbol(symbol_id)
            if not symbol:
                continue
            similarity_score = self._score_from_distance(distance)
            ranking_score = self._ranking_score(
                symbol,
                similarity_score=similarity_score,
                ranking_context=ranking_context,
            )
            scored.append(
                (
                    ranking_score,
                    similarity_score,
                    symbol.file_path,
                    symbol.start_line,
                    symbol.id,
                    symbol,
                )
            )
        scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4]))
        results = []
        for ranking_score, similarity_score, _file, _line, _symbol_id, symbol in scored[:top_k]:
            results.append(
                self._serialize_result(
                    symbol,
                    similarity_score=similarity_score,
                    ranking_score=ranking_score,
                    context_radius=context_radius,
                )
            )
        return results

    def _search_filtered(
        self,
        query_vector: list[float],
        filters: dict[str, str],
        top_k: int,
        *,
        ranking_context: RankingContext,
        context_radius: int,
    ) -> list[dict[str, object]]:
        """Search within metadata-filtered symbols and rank by similarity."""
        if top_k < 1:
            return []
        candidates = self._filter_symbols(filters)
        if not candidates:
            return []
        scored: list[tuple[float, float, str, int, str, object]] = []
        for symbol in candidates:
            similarity_score = self._score_symbol(query_vector, symbol)
            if similarity_score is None:
                similarity_score = 0.0
            ranking_score = self._ranking_score(
                symbol,
                similarity_score=similarity_score,
                ranking_context=ranking_context,
            )
            scored.append(
                (
                    ranking_score,
                    similarity_score,
                    symbol.file_path,
                    symbol.start_line,
                    symbol.id,
                    symbol,
                )
            )
        scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4]))
        results = []
        for ranking_score, similarity_score, _file, _line, _symbol_id, symbol in scored[:top_k]:
            results.append(
                self._serialize_result(
                    symbol,
                    similarity_score=similarity_score,
                    ranking_score=ranking_score,
                    context_radius=context_radius,
                )
            )
        return results

    def _filter_symbols(self, filters: dict[str, str]):
        """Return symbols matching metadata filters with path normalization."""
        kinds = [filters["kind"]] if filters.get("kind") else None
        language = filters.get("language")
        file_path = filters.get("file")
        if not file_path:
            return self.metadata_store.filter_symbols(kinds=kinds, language=language)
        file_matcher = getattr(self.metadata_store, "filter_symbols_by_file_match", None)
        if callable(file_matcher):
            return file_matcher(file_path=file_path, kinds=kinds, language=language)
        for candidate in self._file_path_candidates(file_path):
            symbols = self.metadata_store.filter_symbols(
                kinds=kinds,
                file_path=candidate,
                language=language,
            )
            if symbols:
                return symbols
        return []

    @staticmethod
    def _metadata_filters(filters: dict[str, str] | None) -> dict[str, str]:
        """Extract metadata-store filters from full query-time filter payload."""
        if not filters:
            return {}
        filtered: dict[str, str] = {}
        for key in ("kind", "file", "language"):
            value = filters.get(key)
            if isinstance(value, str) and value.strip():
                filtered[key] = value
        return filtered

    @classmethod
    def _build_ranking_context(
        cls,
        query: str,
        filters: dict[str, str] | None,
    ) -> RankingContext:
        """Resolve ranking mode, query intent, and test-penalty policy for one query."""
        active_filters = filters or {}
        ranking_mode = cls._resolve_ranking_mode(active_filters.get("ranking_mode"))
        query_intent = cls._parse_query_intent(query)
        file_filter = active_filters.get("file")
        explicit_file_test_intent = False
        if isinstance(file_filter, str) and file_filter.strip():
            explicit_file_test_intent = cls._is_test_path(file_filter)
        explicit_test_intent = query_intent.explicit_test_intent or explicit_file_test_intent
        test_penalty_applied = cls._should_apply_test_penalty(
            ranking_mode=ranking_mode,
            query_intent=query_intent,
            explicit_test_intent=explicit_test_intent,
        )
        policy = _BALANCED_POLICY
        if ranking_mode == RANKING_MODE_SOURCE_FIRST:
            policy = _SOURCE_FIRST_POLICY
        return RankingContext(
            ranking_mode=ranking_mode,
            query_intent=query_intent,
            explicit_test_intent=explicit_test_intent,
            test_penalty_applied=test_penalty_applied,
            policy=policy,
        )

    @staticmethod
    def _resolve_ranking_mode(raw_mode: str | None) -> str:
        """Normalize ranking mode with a conservative default."""
        if not raw_mode:
            return RANKING_MODE_BALANCED
        normalized = raw_mode.strip().lower()
        if normalized in _RANKING_MODES:
            return normalized
        return RANKING_MODE_BALANCED

    @staticmethod
    def _query_intent_label(query_intent: QueryIntent) -> str:
        """Serialize query intent into stable metadata labels."""
        if query_intent.is_identifier_query:
            return "identifier"
        return "semantic"

    @staticmethod
    def _parse_query_intent(query: str) -> QueryIntent:
        """Parse query into identifier/non-identifier plus explicit test-intent markers."""
        normalized = query.strip()
        if not normalized:
            return QueryIntent(
                is_identifier_query=False,
                identifier_tail=None,
                explicit_test_intent=False,
            )
        lowered = normalized.lower()
        explicit_test_intent = bool(
            _TEST_INTENT_RE.search(normalized)
            or "/tests/" in lowered
            or lowered.startswith("tests/")
            or " test_" in f" {lowered}"
        )
        if re.search(r"\s", normalized):
            return QueryIntent(
                is_identifier_query=False,
                identifier_tail=None,
                explicit_test_intent=explicit_test_intent,
            )
        if not _IDENTIFIER_QUERY_RE.fullmatch(normalized):
            return QueryIntent(
                is_identifier_query=False,
                identifier_tail=None,
                explicit_test_intent=explicit_test_intent,
            )
        tail = re.split(r"[.:#]+", normalized)[-1]
        return QueryIntent(
            is_identifier_query=True,
            identifier_tail=tail,
            explicit_test_intent=explicit_test_intent,
        )

    @staticmethod
    def _should_apply_test_penalty(
        *,
        ranking_mode: str,
        query_intent: QueryIntent,
        explicit_test_intent: bool,
    ) -> bool:
        """Return whether ranking should penalize test paths/symbol names."""
        if explicit_test_intent:
            return False
        if ranking_mode == RANKING_MODE_SOURCE_FIRST:
            return True
        return query_intent.is_identifier_query

    @staticmethod
    def _is_source_path(path: str) -> bool:
        """Return whether a path belongs to a source-focused directory."""
        normalized = os.path.normpath(path).lower()
        segments = {segment for segment in normalized.split(os.sep) if segment}
        return "src" in segments

    @staticmethod
    def _is_test_path(path: str) -> bool:
        """Return whether a path appears to be a test file path."""
        normalized = os.path.normpath(path).lower()
        segments = {segment for segment in normalized.split(os.sep) if segment}
        if "tests" in segments or "test" in segments:
            return True
        basename = os.path.basename(normalized)
        return (
            basename.startswith("test_")
            or basename.endswith("_test.py")
            or basename.endswith("_test.go")
            or basename.endswith(".spec.ts")
            or basename.endswith(".spec.tsx")
            or basename.endswith(".spec.js")
            or basename.endswith(".spec.jsx")
        )

    @staticmethod
    def _is_test_symbol_name(symbol_name: str) -> bool:
        """Return whether a symbol name appears to be test-centric."""
        return bool(_TEST_SYMBOL_NAME_RE.match(symbol_name.lower()))

    @classmethod
    def _candidate_fetch_size(cls, top_k: int, ranking_context: RankingContext) -> int:
        """Return vector candidate count used before reranking."""
        multiplier = _UNFILTERED_BASE_FETCH_MULTIPLIER
        if ranking_context.ranking_mode == RANKING_MODE_SOURCE_FIRST:
            if not ranking_context.explicit_test_intent:
                multiplier = _SOURCE_FIRST_FETCH_MULTIPLIER
        elif (
            ranking_context.ranking_mode == RANKING_MODE_BALANCED
            and ranking_context.query_intent.is_identifier_query
            and not ranking_context.explicit_test_intent
        ):
            multiplier = _BALANCED_IDENTIFIER_FETCH_MULTIPLIER
        candidate_count = top_k * multiplier
        if candidate_count > _MAX_EXPANDED_CANDIDATES:
            candidate_count = _MAX_EXPANDED_CANDIDATES
        if candidate_count < top_k:
            candidate_count = top_k
        return candidate_count

    @classmethod
    def _ranking_score(
        cls,
        symbol,
        *,
        similarity_score: float,
        ranking_context: RankingContext,
    ) -> float:
        """Compute final ranking score from similarity and deterministic path/name deltas."""
        score = similarity_score
        identifier_tail = ranking_context.query_intent.identifier_tail
        if identifier_tail and symbol.name == identifier_tail:
            score += ranking_context.policy.exact_identifier_tail_bonus
        if cls._is_source_path(symbol.file_path):
            score += ranking_context.policy.source_path_bonus
        if ranking_context.test_penalty_applied:
            if cls._is_test_path(symbol.file_path):
                score += ranking_context.policy.test_path_penalty
            if cls._is_test_symbol_name(symbol.name):
                score += ranking_context.policy.test_symbol_penalty
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    @staticmethod
    def _file_path_candidates(file_path: str) -> list[str]:
        """Return candidate file paths to match against stored symbols."""
        candidates: list[str] = []
        for candidate in (file_path, os.path.normpath(file_path)):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if not os.path.isabs(file_path):
            dot_candidate = os.path.join(".", file_path)
            for candidate in (dot_candidate, os.path.normpath(dot_candidate)):
                if candidate not in candidates:
                    candidates.append(candidate)
        abs_candidate = os.path.abspath(file_path)
        if abs_candidate not in candidates:
            candidates.append(abs_candidate)
        return candidates

    @staticmethod
    def _file_filter_metadata(
        *,
        filters: dict[str, str],
        results: list[dict[str, object]],
    ) -> dict[str, object]:
        """Return stable metadata describing file-filter behavior."""
        file_filter: str | None = None
        raw_filter = filters.get("file")
        if isinstance(raw_filter, str) and raw_filter.strip():
            file_filter = raw_filter
        warnings: list[str] = []
        if file_filter and not results:
            warnings.append(_FILE_FILTER_NO_MATCH_WARNING)
        return {
            "file_filter": file_filter,
            "file_filter_match_mode": _FILE_FILTER_MATCH_MODE,
            "file_filter_warning_codes": warnings,
        }

    def _score_symbol(self, query_vector: list[float], symbol) -> float | None:
        """Return similarity score for a symbol or None if scoring fails."""
        if not symbol.embedding_vector:
            return None
        if len(symbol.embedding_vector) != len(query_vector):
            return None
        distance = self._l2_distance(query_vector, symbol.embedding_vector)
        return self._score_from_distance(distance)

    @staticmethod
    def _l2_distance(a: list[float], b: list[float]) -> float:
        """Return squared L2 distance between vectors."""
        return sum((left - right) ** 2 for left, right in zip(a, b, strict=True))

    @staticmethod
    def _score_from_distance(distance: float) -> float:
        """Convert a distance to a bounded similarity score."""
        score = 1.0 - (distance / 2.0)
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    def _serialize_result(
        self,
        symbol,
        *,
        similarity_score: float,
        ranking_score: float | None = None,
        context_radius: int = _DEFAULT_CONTEXT_RADIUS,
    ) -> dict[str, object]:
        """Build the JSON-friendly result payload for a symbol."""
        if ranking_score is None:
            ranking_score = similarity_score
        return {
            "symbol_id": symbol.id,
            "symbol": symbol.name,
            "kind": symbol.kind,
            "file": symbol.file_path,
            "line": symbol.start_line,
            "line_end": symbol.end_line,
            "signature": symbol.signature,
            "docstring": symbol.docstring,
            "similarity_score": similarity_score,
            "ranking_score": ranking_score,
            "context": self._load_context(
                symbol.file_path,
                symbol.start_line,
                radius=context_radius,
            ),
        }

    @staticmethod
    def _load_context(path: str, line: int, radius: int = _DEFAULT_CONTEXT_RADIUS) -> str:
        """Load a small context window around a symbol."""
        if radius < 1:
            radius = 1
        try:
            with open(path, encoding="utf8") as handle:
                lines = handle.readlines()
        except OSError:
            return ""
        start = max(0, line - radius - 1)
        end = min(len(lines), line + radius)
        return "".join(lines[start:end]).strip()
