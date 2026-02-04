from __future__ import annotations

import logging
import re
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple

from gloggur.models import Symbol
from gloggur.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class DocstringReport:
    symbol_id: str
    warnings: List[str]
    semantic_score: Optional[float] = None


def validate_docstrings(
    symbols: List[Symbol],
    *,
    code_texts: Optional[Dict[str, str]] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    semantic_threshold: Optional[float] = 0.2,
    semantic_max_chars: int = 4000,
) -> List[DocstringReport]:
    semantic_scores = _compute_semantic_scores(
        symbols,
        code_texts=code_texts,
        embedding_provider=embedding_provider,
        max_chars=semantic_max_chars,
    )
    reports: List[DocstringReport] = []
    for symbol in symbols:
        score = semantic_scores.get(symbol.id)
        warnings = _validate_symbol(symbol, score, semantic_threshold)
        if warnings or score is not None:
            reports.append(DocstringReport(symbol_id=symbol.id, warnings=warnings, semantic_score=score))
            if warnings:
                logger.debug("Docstring warnings for %s: %s", symbol.id, warnings)
    warning_count = sum(1 for report in reports if report.warnings)
    logger.info(
        "Docstring validation completed (symbols=%d, warnings=%d)", len(symbols), warning_count
    )
    return reports


def _validate_symbol(
    symbol: Symbol,
    semantic_score: Optional[float],
    semantic_threshold: Optional[float],
) -> List[str]:
    warnings: List[str] = []
    if symbol.kind not in {"function", "class", "interface"}:
        return warnings
    if not symbol.docstring:
        warnings.append("Missing docstring")
        return warnings
    if semantic_score is not None and semantic_threshold is not None and semantic_threshold > 0:
        if semantic_score < semantic_threshold:
            warnings.append(
                "Low semantic similarity "
                f"(score={semantic_score:.3f}, threshold={semantic_threshold:.3f})"
            )
    return warnings


def _compute_semantic_scores(
    symbols: List[Symbol],
    *,
    code_texts: Optional[Dict[str, str]],
    embedding_provider: Optional[EmbeddingProvider],
    max_chars: int,
) -> Dict[str, float]:
    if not embedding_provider or not code_texts:
        return {}
    pairs: List[Tuple[str, str, str, Optional[str]]] = []
    for symbol in symbols:
        if not symbol.docstring:
            continue
        code_text = code_texts.get(symbol.id)
        if not code_text:
            continue
        cleaned = _prepare_code_text(code_text, symbol.language, symbol.docstring, max_chars)
        if not cleaned:
            continue
        pairs.append((symbol.id, symbol.docstring, cleaned, symbol.language))
    if not pairs:
        return {}
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
    return scores


def _prepare_code_text(
    code_text: str,
    language: Optional[str],
    docstring: Optional[str],
    max_chars: int,
) -> str:
    text = code_text
    if language == "python" and docstring:
        text = _strip_python_docstring(text)
    text = text.strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    return text


def _strip_python_docstring(code_text: str) -> str:
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
