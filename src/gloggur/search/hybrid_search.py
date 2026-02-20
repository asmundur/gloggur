from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.storage.metadata_store import MetadataStore
from gloggur.storage.vector_store import VectorStore


@dataclass
class SearchResult:
    """Dataclass for search hits: symbol_id and similarity_score."""
    symbol_id: str
    similarity_score: float


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
        filters: Optional[Dict[str, str]] = None,
        top_k: int = 10,
    ) -> Dict[str, object]:
        """Search for symbols matching the query and filters."""
        start = time.time()
        filters = filters or {}
        query_vector = self.embedding_provider.embed_text(query)
        if filters:
            results = self._search_filtered(query_vector, filters, top_k)
        else:
            results = self._search_unfiltered(query_vector, top_k)
        duration_ms = int((time.time() - start) * 1000)
        return {
            "query": query,
            "results": results,
            "metadata": {"total_results": len(results), "search_time_ms": duration_ms},
        }

    def _search_unfiltered(self, query_vector: List[float], top_k: int) -> List[Dict[str, object]]:
        """Search via vector index without any filters."""
        hits = self.vector_store.search(query_vector, k=top_k * 2)
        results = []
        for symbol_id, distance in hits:
            symbol = self.metadata_store.get_symbol(symbol_id)
            if not symbol:
                continue
            score = self._score_from_distance(distance)
            results.append(self._serialize_result(symbol, score))
            if len(results) >= top_k:
                break
        return results

    def _search_filtered(
        self,
        query_vector: List[float],
        filters: Dict[str, str],
        top_k: int,
    ) -> List[Dict[str, object]]:
        """Search within metadata-filtered symbols and rank by similarity."""
        candidates = self._filter_symbols(filters)
        if not candidates:
            return []
        scored: List[Tuple[float, int, object]] = []
        for symbol in candidates:
            score = self._score_symbol(query_vector, symbol)
            if score is None:
                score = 0.0
            scored.append((score, symbol.start_line, symbol))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        results = []
        for score, _line, symbol in scored[:top_k]:
            results.append(self._serialize_result(symbol, score))
        return results

    def _filter_symbols(self, filters: Dict[str, str]):
        """Return symbols matching metadata filters with path normalization."""
        kinds = [filters["kind"]] if filters.get("kind") else None
        language = filters.get("language")
        file_path = filters.get("file")
        if not file_path:
            return self.metadata_store.filter_symbols(kinds=kinds, language=language)
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
    def _file_path_candidates(file_path: str) -> List[str]:
        """Return candidate file paths to match against stored symbols."""
        candidates: List[str] = []
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

    def _score_symbol(self, query_vector: List[float], symbol) -> Optional[float]:
        """Return similarity score for a symbol or None if scoring fails."""
        if not symbol.embedding_vector:
            return None
        if len(symbol.embedding_vector) != len(query_vector):
            return None
        distance = self._l2_distance(query_vector, symbol.embedding_vector)
        return self._score_from_distance(distance)

    @staticmethod
    def _l2_distance(a: List[float], b: List[float]) -> float:
        """Return squared L2 distance between vectors."""
        return sum((left - right) ** 2 for left, right in zip(a, b))

    @staticmethod
    def _score_from_distance(distance: float) -> float:
        """Convert a distance to a bounded similarity score."""
        score = 1.0 - (distance / 2.0)
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    def _serialize_result(self, symbol, score: float) -> Dict[str, object]:
        """Build the JSON-friendly result payload for a symbol."""
        return {
            "symbol": symbol.name,
            "kind": symbol.kind,
            "file": symbol.file_path,
            "line": symbol.start_line,
            "signature": symbol.signature,
            "docstring": symbol.docstring,
            "similarity_score": score,
            "context": self._load_context(symbol.file_path, symbol.start_line),
        }

    @staticmethod
    def _load_context(path: str, line: int, radius: int = 3) -> str:
        """Load a small context window around a symbol."""
        try:
            with open(path, "r", encoding="utf8") as handle:
                lines = handle.readlines()
        except OSError:
            return ""
        start = max(0, line - radius - 1)
        end = min(len(lines), line + radius)
        return "".join(lines[start:end]).strip()
