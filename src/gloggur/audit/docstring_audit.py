from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Tuple

from gloggur.models import Symbol
from gloggur.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class DocstringAuditReport:
    """Audit report for a single symbol (warnings and score)."""
    symbol_id: str
    warnings: List[str]
    semantic_score: Optional[float] = None
    score_metadata: Optional[Dict[str, object]] = None


def audit_docstrings(
    symbols: List[Symbol],
    *,
    code_texts: Optional[Dict[str, str]] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    semantic_threshold: Optional[float] = 0.2,
    semantic_min_chars: int = 0,
    semantic_max_chars: int = 4000,
    semantic_min_code_chars: int = 0,
    kind_thresholds: Optional[Dict[str, float]] = None,
) -> List[DocstringAuditReport]:
    """Audit docstrings and compute semantic similarity scores.

    Args:
        symbols: Symbols to audit.
        code_texts: Map of symbol ID to raw source snippet for scoring.
        embedding_provider: Provider for computing embedding vectors.
        semantic_threshold: Global low-similarity threshold (default 0.2).
        semantic_min_chars: Minimum character count for docstring and code body
            to be eligible for scoring.  Set >0 to skip scoring trivially short
            content where similarity signals are unreliable.
        semantic_max_chars: Maximum code-body characters sent to the embedder.
        semantic_min_code_chars: Minimum character count for the code body
            (after stripping the docstring) to be eligible for scoring.  Symbols
            whose implementation is trivially short yield unreliable scores and
            are skipped when this is set >0.
        kind_thresholds: Per-symbol-kind threshold overrides.  Keys are symbol
            kinds (``"function"``, ``"class"``, ``"interface"``); values override
            ``semantic_threshold`` for that kind.  Useful for applying a more
            lenient threshold to abstract class/interface docstrings.
    """
    semantic_scores, skip_reasons = _compute_semantic_scores(
        symbols,
        code_texts=code_texts,
        embedding_provider=embedding_provider,
        min_chars=semantic_min_chars,
        max_chars=semantic_max_chars,
        min_code_chars=semantic_min_code_chars,
    )
    reports: List[DocstringAuditReport] = []
    for symbol in symbols:
        score = semantic_scores.get(symbol.id)
        skip_reason = skip_reasons.get(symbol.id)
        warnings, score_metadata = _assess_symbol(
            symbol,
            score,
            semantic_threshold,
            kind_thresholds=kind_thresholds,
            skip_reason=skip_reason,
        )
        if warnings or score is not None or skip_reason is not None:
            reports.append(
                DocstringAuditReport(
                    symbol_id=symbol.id,
                    warnings=warnings,
                    semantic_score=score,
                    score_metadata=score_metadata,
                )
            )
            if warnings:
                logger.debug("Docstring warnings for %s: %s", symbol.id, warnings)
    warning_count = sum(1 for report in reports if report.warnings)
    logger.info(
        "Docstring audit completed (symbols=%d, warnings=%d)", len(symbols), warning_count
    )
    return reports


def _assess_symbol(
    symbol: Symbol,
    semantic_score: Optional[float],
    semantic_threshold: Optional[float],
    kind_thresholds: Optional[Dict[str, float]] = None,
    skip_reason: Optional[str] = None,
) -> Tuple[List[str], Optional[Dict[str, object]]]:
    """Return (warnings, score_metadata) for a symbol based on docstring content.

    ``score_metadata`` is ``None`` for symbol kinds that are not inspected.
    For inspected symbols it contains:

    * ``symbol_kind`` — the symbol kind string.
    * ``threshold_applied`` — the effective threshold used for comparison.
    * ``scored`` — whether embedding similarity was computed.
    * ``score_value`` — the raw cosine similarity (``None`` when not scored).
    * ``skip_reason`` — why scoring was skipped, if applicable.
    """
    warnings: List[str] = []
    if symbol.kind not in {"function", "class", "interface"}:
        return warnings, None
    if not symbol.docstring:
        warnings.append("Missing docstring")
        return warnings, None

    # Resolve effective threshold: kind-specific override takes priority.
    effective_threshold = semantic_threshold
    if kind_thresholds and symbol.kind in kind_thresholds:
        effective_threshold = kind_thresholds[symbol.kind]

    score_metadata: Dict[str, object] = {
        "symbol_kind": symbol.kind,
        "threshold_applied": effective_threshold,
        "scored": semantic_score is not None,
        "score_value": semantic_score,
    }
    if skip_reason is not None:
        score_metadata["skip_reason"] = skip_reason

    if semantic_score is not None and effective_threshold is not None and effective_threshold > 0:
        if semantic_score < effective_threshold:
            warnings.append(
                "Low semantic similarity "
                f"(score={semantic_score:.3f}, threshold={effective_threshold:.3f})"
            )
    return warnings, score_metadata


def _compute_semantic_scores(
    symbols: List[Symbol],
    *,
    code_texts: Optional[Dict[str, str]],
    embedding_provider: Optional[EmbeddingProvider],
    min_chars: int,
    max_chars: int,
    min_code_chars: int = 0,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Compute docstring-to-code semantic similarity scores via embeddings.

    Returns a tuple of:
    * ``scores`` — map of symbol ID to cosine similarity.
    * ``skip_reasons`` — map of symbol ID to skip reason string for symbols
      that had a docstring but were excluded from scoring.
    """
    if not embedding_provider or not code_texts:
        return {}, {}
    pairs: List[Tuple[str, str, str, Optional[str]]] = []
    skip_reasons: Dict[str, str] = {}
    for symbol in symbols:
        if not symbol.docstring:
            continue
        docstring_len = len(symbol.docstring.strip())
        if min_chars > 0 and docstring_len < min_chars:
            skip_reasons[symbol.id] = (
                f"docstring_too_short (len={docstring_len} < min={min_chars})"
            )
            continue
        code_text = code_texts.get(symbol.id)
        if not code_text:
            continue
        cleaned = _prepare_code_text(code_text, symbol.language, symbol.docstring, max_chars)
        if not cleaned:
            continue
        cleaned_len = len(cleaned.strip())
        if min_chars > 0 and cleaned_len < min_chars:
            skip_reasons[symbol.id] = (
                f"code_body_too_short (len={cleaned_len} < min={min_chars})"
            )
            continue
        if min_code_chars > 0 and cleaned_len < min_code_chars:
            skip_reasons[symbol.id] = (
                f"code_body_too_short (len={cleaned_len} < min_code={min_code_chars})"
            )
            continue
        pairs.append((symbol.id, symbol.docstring, cleaned, symbol.language))
    if not pairs:
        return {}, skip_reasons
    texts: List[str] = []
    for _, docstring, code_text, _ in pairs:
        texts.append(docstring)
        texts.append(code_text)
    vectors = embedding_provider.embed_batch(texts)
    scores: Dict[str, float] = {}
    for idx, (symbol_id, _, _, _) in enumerate(pairs):
        doc_vector = vectors[2 * idx]
        code_vector = vectors[2 * idx + 1]
        scores[symbol_id] = _cosine_similarity(doc_vector, code_vector)
    return scores, skip_reasons


def _prepare_code_text(
    code_text: str,
    language: Optional[str],
    docstring: Optional[str],
    max_chars: int,
) -> str:
    """Prepare code text for similarity scoring (strip docstring, trim)."""
    text = code_text
    if language == "python" and docstring:
        text = _strip_python_docstring(text)
    text = text.strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    return text


def _strip_python_docstring(code_text: str) -> str:
    """Remove the first Python docstring from a code snippet."""
    lines = code_text.splitlines()
    if len(lines) <= 1:
        return code_text
    header = lines[0]
    body = "\n".join(lines[1:])
    body = re.sub(r"^\s*(?P<quote>\"\"\"|''')(?:.|\n)*?(?P=quote)\s*", "", body, count=1)
    body = re.sub(r"^\s*(?P<quote>\"|')(?:.|\n)*?(?P=quote)\s*", "", body, count=1)
    combined = "\n".join([header, body]).strip()
    return combined


def _cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vector_a, vector_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom
