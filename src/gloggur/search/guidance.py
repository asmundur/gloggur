from __future__ import annotations

from gloggur.search.hybrid_search import HybridSearch


class AgentGuidance:
    """Agent guidance layer providing aggregated context and impact analysis."""

    def __init__(self, searcher: HybridSearch) -> None:
        """Initialize guidance layer with a searcher."""
        self.searcher = searcher

    def get_change_impact_radius(self, symbol_id: str, radius: int = 5) -> dict[str, object]:
        """Estimate the impact radius of changing a symbol.

        Note: Currently uses structural neighborhoods as a baseline since
        Static Call Graph and Coverage Map are pending.
        """
        neighborhood = self.searcher.get_semantic_neighborhood(symbol_id, radius=radius, top_k=5)

        impacted_symbols = neighborhood.get("structural_neighbors", [])
        assert isinstance(impacted_symbols, list)

        return {
            "target_symbol_id": symbol_id,
            "estimated_impact_count": len(impacted_symbols),
            "potentially_impacted_symbols": [
                s.get("symbol_id") for s in impacted_symbols if isinstance(s, dict)
            ],
            "notes": "Impact radius currently limited to structural neighbors.",
        }

    def generate_agent_context(self, symbol_id: str) -> dict[str, object]:
        """Generate an aggregated, LLM-consumable context payload for a specific symbol."""
        symbol = self.searcher.metadata_store.get_symbol(symbol_id)
        if not symbol:
            return {"error": f"Symbol not found: {symbol_id}"}

        neighborhood = self.searcher.get_semantic_neighborhood(symbol_id)
        impact = self.get_change_impact_radius(symbol_id)

        semantic_neighbors = neighborhood.get("semantic_neighbors", [])
        assert isinstance(semantic_neighbors, list)

        return {
            "symbol_id": symbol.id,
            "name": symbol.name,
            "kind": symbol.kind,
            "file": symbol.file_path,
            "invariants": getattr(symbol, "invariants", []),
            "is_serialization_boundary": getattr(symbol, "is_serialization_boundary", False),
            "implicit_contract": getattr(symbol, "implicit_contract", None),
            "semantic_neighbors": [
                {"symbol_id": n.get("symbol_id"), "similarity": n.get("similarity_score")}
                for n in semantic_neighbors
                if isinstance(n, dict)
            ],
            "change_impact": impact,
        }
