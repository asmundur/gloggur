from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.storage.metadata_store import MetadataStore
from gloggur.storage.vector_store import VectorStore


@dataclass
class SearchResult:
    symbol_id: str
    similarity_score: float


class HybridSearch:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None,
        top_k: int = 10,
    ) -> Dict[str, object]:
        start = time.time()
        filters = filters or {}
        query_vector = self.embedding_provider.embed_text(query)
        hits = self.vector_store.search(query_vector, k=top_k * 2)
        results = []
        for symbol_id, distance in hits:
            symbol = self.metadata_store.get_symbol(symbol_id)
            if not symbol:
                continue
            if "kind" in filters and symbol.kind != filters["kind"]:
                continue
            if "file" in filters and symbol.file_path != filters["file"]:
                continue
            if "language" in filters and symbol.language != filters["language"]:
                continue
            score = 1.0 - (distance / 2.0)
            results.append(
                {
                    "symbol": symbol.name,
                    "kind": symbol.kind,
                    "file": symbol.file_path,
                    "line": symbol.start_line,
                    "signature": symbol.signature,
                    "docstring": symbol.docstring,
                    "similarity_score": score,
                    "context": self._load_context(symbol.file_path, symbol.start_line),
                }
            )
            if len(results) >= top_k:
                break
        duration_ms = int((time.time() - start) * 1000)
        return {
            "query": query,
            "results": results,
            "metadata": {"total_results": len(results), "search_time_ms": duration_ms},
        }

    @staticmethod
    def _load_context(path: str, line: int, radius: int = 3) -> str:
        try:
            with open(path, "r", encoding="utf8") as handle:
                lines = handle.readlines()
        except OSError:
            return ""
        start = max(0, line - radius - 1)
        end = min(len(lines), line + radius)
        return "".join(lines[start:end]).strip()
