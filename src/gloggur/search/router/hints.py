from __future__ import annotations

import re

from gloggur.search.router.types import QueryHints

_QUOTED_RE = re.compile(r"[\"']([^\"']+)[\"']")
_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:[.:#][A-Za-z_][A-Za-z0-9_]*)*\b")
_PATH_RE = re.compile(r"(?:\.?/?[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)")
_STACK_RE = re.compile(r"(?P<path>[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+):(?P<line>[0-9]{1,7})")
_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    """Return stable-order unique tuple, dropping blanks."""
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def extract_query_hints(query: str) -> QueryHints:
    """Extract deterministic hints used by routing and scoring."""
    raw_literals = _ordered_unique(_QUOTED_RE.findall(query))
    literal_symbol_candidates: list[str] = []
    for literal in raw_literals:
        for match in _SYMBOL_RE.findall(literal):
            normalized = match.strip()
            if not normalized:
                continue
            literal_symbol_candidates.append(normalized)

    raw_symbols: list[str] = []
    for match in _SYMBOL_RE.findall(query):
        # Avoid noisy stopwords and very short tokens.
        lowered = match.lower()
        if lowered in {"the", "and", "for", "with", "from", "error", "line"}:
            continue
        if len(match) < 3:
            continue
        raw_symbols.append(match)
    for match in literal_symbol_candidates:
        lowered = match.lower()
        if lowered in {"the", "and", "for", "with", "from", "error", "line"}:
            continue
        raw_symbols.append(match)

    raw_paths = _ordered_unique(_PATH_RE.findall(query))

    stack_locations: list[tuple[str, int]] = []
    for match in _STACK_RE.finditer(query):
        path = match.group("path").strip()
        try:
            line = int(match.group("line"))
        except ValueError:
            continue
        if line < 1:
            continue
        stack_locations.append((path, line))

    identifier_tokens_list = [
        token.lower() for token in _TOKEN_RE.findall(query) if len(token) >= 3
    ]
    for literal in literal_symbol_candidates:
        token = literal.lower()
        if token not in identifier_tokens_list:
            identifier_tokens_list.append(token)
    identifier_tokens = _ordered_unique(identifier_tokens_list)

    return QueryHints(
        symbols=_ordered_unique(raw_symbols),
        literals=raw_literals,
        path_hints=raw_paths,
        stack_locations=tuple(stack_locations),
        identifier_tokens=identifier_tokens,
    )
