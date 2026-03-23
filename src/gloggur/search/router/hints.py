from __future__ import annotations

import re

from gloggur.search.router.path_priors import QUERY_DOMAINS
from gloggur.search.router.types import QueryHints

_QUOTED_RE = re.compile(r"[\"']([^\"']+)[\"']")
_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:[.:#][A-Za-z_][A-Za-z0-9_]*)*\b")
_PATH_RE = re.compile(r"(?:\.?/?[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)")
_STACK_RE = re.compile(r"(?P<path>[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+):(?P<line>[0-9]{1,7})")
_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_DECLARATION_TERMS = {
    "def",
    "class",
    "function",
    "method",
    "struct",
    "interface",
    "trait",
    "enum",
}
_DECLARATION_MODIFIERS = {"async", "public", "private", "protected", "static", "export"}
_SYMBOL_STOPWORDS = {"the", "and", "for", "with", "from", "error", "line"}
_NATURAL_LANGUAGE_HINTS = {
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "who",
    "should",
    "does",
    "order",
    "tuple",
    "table",
    "safe",
    "safety",
    "return",
    "returns",
    "behavior",
    "behaviour",
}
_BEHAVIOR_PROCESS_HINTS = {
    "absolute",
    "alias",
    "config",
    "configs",
    "footer",
    "forwarded",
    "format",
    "order",
    "parity",
    "preserve",
    "redirect",
    "tuple",
    "uri",
    "workflow",
    "workflows",
}
_IDENTIFIERISH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:[.:#][A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?$")
_LITERAL_FIRST_PUNCTUATION = set("[](){}<>/=+-")
_WORKFLOW_PHRASES = (
    "ci",
    "workflow",
    "workflows",
    ".github",
    "github actions",
    "matrix",
    "label",
    "labels",
    "backport",
    "stable branch",
    "stable/",
    "subject prefix",
    "commit prefix",
    "backport footer",
    "contributing",
    "review stage",
    "trac",
    "lane",
    "migrations workflow",
    "pyproject.toml",
    "tox.ini",
)
_DOCS_POLICY_PHRASES = (
    "docs",
    "docs/internals",
    "readme",
    "changelog",
    "history",
    "release notes",
    "release-notes",
    "deprecation",
    "paper trail",
    "warning",
    "warnings",
)


def _detect_verbatim_literal(query: str) -> str | None:
    candidate = query.strip()
    if not candidate or any(char.isspace() for char in candidate):
        return None
    if "/" not in candidate and "\\" not in candidate:
        return None
    if not any(char.isalnum() for char in candidate):
        return None
    return candidate


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


def _strip_declaration_prefix(query: str) -> tuple[str, tuple[str, ...]]:
    """Remove declaration-oriented prefixes while preserving declaration intent."""
    tokens = query.strip().split()
    if not tokens:
        return "", ()

    index = 0
    while index < len(tokens) and tokens[index].lower() in _DECLARATION_MODIFIERS:
        index += 1

    declaration_terms: list[str] = []
    while index < len(tokens) and tokens[index].lower() in _DECLARATION_TERMS:
        declaration_terms.append(tokens[index].lower())
        index += 1

    if declaration_terms and index < len(tokens):
        return " ".join(tokens[index:]), tuple(declaration_terms)
    return query.strip(), tuple(declaration_terms)


def _looks_identifierish_query(query: str) -> bool:
    candidate = query.strip()
    if not candidate:
        return False
    if candidate.endswith("("):
        candidate = candidate[:-1].rstrip()
    return bool(_IDENTIFIERISH_RE.fullmatch(candidate))


def _has_symbol_shape(symbols: tuple[str, ...]) -> bool:
    for symbol in symbols:
        if any(marker in symbol for marker in ("_", ".", ":", "#")):
            return True
        if any(char.isupper() for char in symbol[1:]):
            return True
    return False


def _classify_query_kind(
    *,
    original_query: str,
    declaration_terms: tuple[str, ...],
    symbols: tuple[str, ...],
    identifier_tokens: tuple[str, ...],
) -> str:
    if declaration_terms and (symbols or identifier_tokens):
        return "declaration"

    stripped_query = original_query.strip()
    if _looks_identifierish_query(stripped_query):
        return "identifier"

    symbol_shaped = _has_symbol_shape(symbols)
    nl_tokens = {
        token.lower()
        for token in _TOKEN_RE.findall(stripped_query)
        if len(token) >= 3 and token.lower() not in _SYMBOL_STOPWORDS
    }
    if (symbols or identifier_tokens) and any(
        token in _BEHAVIOR_PROCESS_HINTS for token in nl_tokens
    ):
        return "mixed"
    if (symbols or identifier_tokens) and any(
        token in _NATURAL_LANGUAGE_HINTS for token in nl_tokens
    ):
        return "mixed"
    if (symbols or identifier_tokens) and symbol_shaped and len(nl_tokens) <= 2:
        return "identifier"
    if symbols or identifier_tokens:
        return "mixed"
    return "natural_language"


def _classify_query_domain(
    *,
    query: str,
    path_hints: tuple[str, ...],
    literals: tuple[str, ...],
) -> str:
    lowered = query.lower()
    workflow_score = 0
    docs_score = 0
    haystacks = [lowered]
    haystacks.extend(path.lower() for path in path_hints)
    haystacks.extend(literal.lower() for literal in literals)

    for phrase in _WORKFLOW_PHRASES:
        if any(phrase in haystack for haystack in haystacks):
            workflow_score += 1
    for phrase in _DOCS_POLICY_PHRASES:
        if any(phrase in haystack for haystack in haystacks):
            docs_score += 1

    if workflow_score > 0 and workflow_score >= docs_score:
        return "workflow_config"
    if docs_score > 0:
        return "docs_policy"
    return "code"


def _literal_first_query(query: str) -> bool:
    candidate = query.strip()
    if not candidate or any(char.isspace() for char in candidate):
        return False
    if _looks_identifierish_query(candidate):
        return False
    if len(candidate) < 3:
        return False
    if not any(char.isalnum() for char in candidate):
        return False
    punctuation = [char for char in candidate if not char.isalnum()]
    if not punctuation:
        return False
    if "/" in candidate or "\\" in candidate:
        return True
    return any(char in _LITERAL_FIRST_PUNCTUATION or char == "." for char in punctuation)


def extract_query_hints(query: str) -> QueryHints:
    """Extract deterministic hints used by routing and scoring."""
    symbol_query, declaration_terms = _strip_declaration_prefix(query)
    verbatim_literal = _detect_verbatim_literal(query)
    raw_literals = _ordered_unique(_QUOTED_RE.findall(query))
    literal_symbol_candidates: list[str] = []
    for literal in raw_literals:
        for match in _SYMBOL_RE.findall(literal):
            normalized = match.strip()
            if not normalized:
                continue
            if normalized.lower() in _DECLARATION_TERMS:
                continue
            literal_symbol_candidates.append(normalized)

    raw_symbols: list[str] = []
    for match in _SYMBOL_RE.findall(symbol_query):
        # Avoid noisy stopwords and very short tokens.
        lowered = match.lower()
        if lowered in _SYMBOL_STOPWORDS or lowered in _DECLARATION_TERMS:
            continue
        if len(match) < 3:
            continue
        raw_symbols.append(match)
    for match in literal_symbol_candidates:
        lowered = match.lower()
        if lowered in _SYMBOL_STOPWORDS or lowered in _DECLARATION_TERMS:
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
        token.lower()
        for token in _TOKEN_RE.findall(symbol_query)
        if len(token) >= 3 and token.lower() not in _DECLARATION_TERMS
    ]
    for literal in literal_symbol_candidates:
        token = literal.lower()
        if token not in identifier_tokens_list:
            identifier_tokens_list.append(token)
    identifier_tokens = _ordered_unique(identifier_tokens_list)
    symbols = _ordered_unique(raw_symbols)
    query_kind = _classify_query_kind(
        original_query=query,
        declaration_terms=declaration_terms,
        symbols=symbols,
        identifier_tokens=identifier_tokens,
    )
    query_domain = _classify_query_domain(
        query=query,
        path_hints=raw_paths,
        literals=raw_literals,
    )
    literal_first = _literal_first_query(query)
    if query_domain not in QUERY_DOMAINS:
        query_domain = "code"

    return QueryHints(
        symbols=symbols,
        literals=raw_literals,
        verbatim_literal=verbatim_literal,
        path_hints=raw_paths,
        stack_locations=tuple(stack_locations),
        identifier_tokens=identifier_tokens,
        declaration_terms=declaration_terms,
        query_kind=query_kind,
        query_domain=query_domain,
        literal_first=literal_first,
    )
