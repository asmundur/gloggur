from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchIntent:
    """Public caller-facing search intent for ContextPack routing."""

    search_mode: str = "semantic"
    semantic_query: str | None = None
    ranking_mode: str = "balanced"
    result_profile: str = "default"
    semantic_assist_mode: str = "none"
    language: str | None = None
    path_prefix: str | None = None
    path_filters: tuple[str, ...] = ()
    max_files: int | None = None
    max_snippets: int | None = None
    time_budget_ms: int | None = None


@dataclass(frozen=True)
class SearchConstraints:
    """Compatibility adapter for older callers during SearchIntent migration."""

    search_mode: str = "semantic"
    language: str | None = None
    path_prefix: str | None = None
    path_filters: tuple[str, ...] = ()
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    case_mode: str | None = None
    word_match: bool = False
    fixed_string: bool = False
    max_files: int | None = None
    max_snippets: int | None = None
    time_budget_ms: int | None = None


@dataclass(frozen=True)
class ExecutionHints:
    """Internal backend execution hints derived from parsed compatibility inputs."""

    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    case_mode: str | None = None
    word_match: bool = False
    fixed_string: bool = False


@dataclass(frozen=True)
class ContextSpan:
    """Inclusive line span for one context hit."""

    start_line: int
    end_line: int


@dataclass(frozen=True)
class ContextHit:
    """ContextPack hit consumed by agents."""

    path: str
    span: ContextSpan
    snippet: str
    score: float
    start_byte: int | None = None
    end_byte: int | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize hit to JSON-friendly payload."""
        payload: dict[str, object] = {
            "path": self.path,
            "span": {
                "start_line": self.span.start_line,
                "end_line": self.span.end_line,
            },
            "snippet": self.snippet,
            "score": self.score,
            "tags": list(self.tags),
        }
        if self.start_byte is not None:
            payload["start_byte"] = self.start_byte
        if self.end_byte is not None:
            payload["end_byte"] = self.end_byte
        return payload


@dataclass(frozen=True)
class ContextPack:
    """Canonical search payload returned by gloggur search."""

    query: str
    summary: dict[str, object]
    hits: tuple[ContextHit, ...]
    debug: dict[str, object] | None = None
    schema_version: int = 2

    def to_dict(self, *, include_debug: bool = False) -> dict[str, object]:
        """Serialize context pack payload."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "query": self.query,
            "summary": self.summary,
            "hits": [hit.to_dict() for hit in self.hits],
        }
        if include_debug and self.debug is not None:
            payload["debug"] = self.debug
        return payload


@dataclass(frozen=True)
class QueryHints:
    """Deterministic query signal extraction output."""

    symbols: tuple[str, ...] = ()
    literals: tuple[str, ...] = ()
    path_hints: tuple[str, ...] = ()
    stack_locations: tuple[tuple[str, int], ...] = ()
    identifier_tokens: tuple[str, ...] = ()
    declaration_terms: tuple[str, ...] = ()
    query_kind: str = "natural_language"


@dataclass(frozen=True)
class BackendHit:
    """Internal backend hit before pack merge/dedupe."""

    backend: str
    path: str
    start_line: int
    end_line: int
    snippet: str
    raw_score: float
    start_byte: int | None = None
    end_byte: int | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendResult:
    """Backend search execution output."""

    name: str
    hits: tuple[BackendHit, ...]
    timing_ms: int
    quality_score: float = 0.0
    error: str | None = None
    commands: tuple[str, ...] = ()


@dataclass
class RouterOutcome:
    """Final routing decision with debug-friendly metadata."""

    strategy: str
    reason: str
    winner: str | None
    considered: tuple[str, ...] = ()
    backend_scores: dict[str, float] = field(default_factory=dict)
    warning_codes: tuple[str, ...] = ()
