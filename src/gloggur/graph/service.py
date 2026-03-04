from __future__ import annotations

import os
import time
from dataclasses import dataclass

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import wrap_embedding_error
from gloggur.models import EdgeRecord
from gloggur.storage.metadata_store import MetadataStore


@dataclass(frozen=True)
class _RankedEdge:
    """Ranked edge wrapper used for stable sorting and payload rendering."""

    edge: EdgeRecord
    endpoint_match_score: int
    module_distance: int
    commit_penalty: int
    similarity_score: float | None = None


class GraphService:
    """Graph retrieval and semantic edge-search APIs."""

    def __init__(
        self,
        metadata_store: MetadataStore,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.embedding_provider = embedding_provider

    def neighbors(
        self,
        symbol_id: str,
        *,
        edge_type: str | None = None,
        direction: str = "both",
        k: int = 20,
    ) -> dict[str, object]:
        """Return ranked neighboring edges for a symbol endpoint."""
        start = time.time()
        edges = self.metadata_store.list_edges_for_symbol(
            symbol_id,
            direction=direction,
            edge_type=edge_type,
            limit=None,
        )
        ranked = self._rank_endpoint_edges(symbol_id=symbol_id, edges=edges, limit=k)
        duration_ms = int((time.time() - start) * 1000)
        return {
            "symbol_id": symbol_id,
            "direction": direction,
            "edge_type": edge_type,
            "results": [self._serialize_ranked_edge(item) for item in ranked],
            "metadata": {
                "total_results": len(ranked),
                "query_time_ms": duration_ms,
            },
        }

    def incoming(
        self,
        symbol_id: str,
        *,
        edge_type: str | None = None,
        k: int = 20,
    ) -> dict[str, object]:
        """Return incoming edges for a symbol."""
        return self.neighbors(symbol_id, edge_type=edge_type, direction="incoming", k=k)

    def outgoing(
        self,
        symbol_id: str,
        *,
        edge_type: str | None = None,
        k: int = 20,
    ) -> dict[str, object]:
        """Return outgoing edges for a symbol."""
        return self.neighbors(symbol_id, edge_type=edge_type, direction="outgoing", k=k)

    def search(
        self,
        query: str,
        *,
        edge_type: str | None = None,
        k: int = 20,
    ) -> dict[str, object]:
        """Semantic search over edge fact text embeddings."""
        start = time.time()
        if self.embedding_provider is None:
            raise RuntimeError("semantic graph search requires an embedding provider")
        try:
            query_vector = self.embedding_provider.embed_text(query)
        except Exception as exc:
            raise wrap_embedding_error(
                exc,
                provider=getattr(self.embedding_provider, "provider", "unknown"),
                operation="embed query for graph search",
            ) from exc
        edges = self.metadata_store.list_edges()
        if edge_type:
            edges = [edge for edge in edges if edge.edge_type == edge_type]

        ranked: list[_RankedEdge] = []
        for edge in edges:
            vector = edge.embedding_vector
            if not vector:
                continue
            if len(vector) != len(query_vector):
                continue
            distance = self._l2_distance(query_vector, vector)
            similarity_score = self._score_from_distance(distance)
            ranked.append(
                _RankedEdge(
                    edge=edge,
                    endpoint_match_score=0,
                    module_distance=0,
                    commit_penalty=0,
                    similarity_score=similarity_score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -(item.similarity_score or 0.0),
                -item.edge.confidence,
                item.edge.file_path,
                item.edge.line,
                item.edge.edge_id,
            )
        )
        ranked = ranked[: max(0, k)]
        duration_ms = int((time.time() - start) * 1000)
        return {
            "query": query,
            "edge_type": edge_type,
            "results": [self._serialize_ranked_edge(item) for item in ranked],
            "metadata": {
                "total_results": len(ranked),
                "query_time_ms": duration_ms,
            },
        }

    def _rank_endpoint_edges(
        self,
        *,
        symbol_id: str,
        edges: list[EdgeRecord],
        limit: int,
    ) -> list[_RankedEdge]:
        """Apply deterministic graph ranking policy for endpoint traversal."""
        if limit < 1:
            return []
        anchor_symbol = self.metadata_store.get_symbol(symbol_id)
        anchor_path = anchor_symbol.file_path if anchor_symbol else None
        latest_commit = self._latest_commit(edges)
        ranked: list[_RankedEdge] = []
        for edge in edges:
            endpoint_match = int(edge.from_id == symbol_id) + int(edge.to_id == symbol_id)
            module_distance = self._module_distance(anchor_path, edge.file_path)
            commit_penalty = 0 if latest_commit and edge.commit == latest_commit else 1
            ranked.append(
                _RankedEdge(
                    edge=edge,
                    endpoint_match_score=endpoint_match,
                    module_distance=module_distance,
                    commit_penalty=commit_penalty,
                )
            )
        ranked.sort(
            key=lambda item: (
                -item.endpoint_match_score,
                -item.edge.confidence,
                item.module_distance,
                item.commit_penalty,
                item.edge.file_path,
                item.edge.line,
                item.edge.edge_id,
            )
        )
        return ranked[:limit]

    @staticmethod
    def _latest_commit(edges: list[EdgeRecord]) -> str | None:
        """Return a stable 'latest' commit marker for deterministic ranking."""
        commits = sorted(
            {
                edge.commit
                for edge in edges
                if isinstance(edge.commit, str) and edge.commit and edge.commit != "unknown"
            }
        )
        if not commits:
            return None
        return commits[-1]

    @staticmethod
    def _module_distance(anchor_path: str | None, other_path: str) -> int:
        """Compute directory-distance heuristic used as a ranking tie-breaker."""
        if not anchor_path:
            return 0
        anchor_parts = [
            segment
            for segment in os.path.normpath(os.path.dirname(anchor_path)).split(os.sep)
            if segment
        ]
        other_parts = [
            segment
            for segment in os.path.normpath(os.path.dirname(other_path)).split(os.sep)
            if segment
        ]
        common = 0
        for left, right in zip(anchor_parts, other_parts, strict=False):
            if left != right:
                break
            common += 1
        return (len(anchor_parts) - common) + (len(other_parts) - common)

    @staticmethod
    def _serialize_ranked_edge(item: _RankedEdge) -> dict[str, object]:
        """Render one ranked edge payload."""
        payload: dict[str, object] = {
            "edge_id": item.edge.edge_id,
            "edge_type": item.edge.edge_type,
            "from_id": item.edge.from_id,
            "to_id": item.edge.to_id,
            "from_kind": item.edge.from_kind,
            "to_kind": item.edge.to_kind,
            "file_path": item.edge.file_path,
            "line": item.edge.line,
            "confidence": item.edge.confidence,
            "repo_id": item.edge.repo_id,
            "commit": item.edge.commit,
            "text": item.edge.text,
            "ranking": {
                "endpoint_match_score": item.endpoint_match_score,
                "module_distance": item.module_distance,
                "commit_penalty": item.commit_penalty,
            },
        }
        if item.similarity_score is not None:
            payload["similarity_score"] = item.similarity_score
        return payload

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
