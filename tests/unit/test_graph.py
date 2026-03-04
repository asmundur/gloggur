from __future__ import annotations

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.graph.extractor import GraphEdgeExtractor
from gloggur.graph.service import GraphService
from gloggur.models import EdgeRecord, Symbol


def _symbol(
    *,
    symbol_id: str,
    name: str,
    file_path: str,
    start_line: int,
    end_line: int,
    fqname: str | None = None,
    calls: list[str] | None = None,
    covered_by: list[str] | None = None,
) -> Symbol:
    return Symbol(
        id=symbol_id,
        name=name,
        kind="function",
        fqname=fqname,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=f"def {name}():",
        body_hash=f"hash-{symbol_id}",
        language="python",
        calls=calls or [],
        covered_by=covered_by or [],
    )


def test_graph_edge_extractor_emits_core_edge_types_with_unresolved_fallback() -> None:
    extractor = GraphEdgeExtractor("python")
    helper = _symbol(
        symbol_id="sym-helper",
        name="helper",
        fqname="sample.helper",
        file_path="sample.py",
        start_line=3,
        end_line=4,
    )
    target = _symbol(
        symbol_id="sym-target",
        name="target",
        fqname="sample.target",
        file_path="sample.py",
        start_line=6,
        end_line=8,
        calls=["helper", "unknown_api"],
        covered_by=["test_target"],
    )
    test_symbol = _symbol(
        symbol_id="sym-test-target",
        name="test_target",
        fqname="tests.test_sample.test_target",
        file_path="tests/test_sample.py",
        start_line=1,
        end_line=2,
    )
    source = (
        "import math\n\n"
        "def helper(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def target(value: int) -> int:\n"
        "    computed = helper(value)\n"
        "    return unknown_api(computed)\n"
    )

    edges = extractor.extract_edges(
        path="sample.py",
        source=source,
        symbols=[helper, target],
        candidate_symbols=[helper, target, test_symbol],
        repo_id="repo-1",
        commit="abc123",
    )

    edge_types = {edge.edge_type for edge in edges}
    assert {"DEFINES", "CALLS", "IMPORTS", "TESTS"}.issubset(edge_types)
    call_edges = [edge for edge in edges if edge.edge_type == "CALLS"]
    assert any(edge.to_id == "sym-helper" for edge in call_edges)
    assert any(edge.to_id == "unresolved:unknown_api" for edge in call_edges)
    test_edges = [edge for edge in edges if edge.edge_type == "TESTS"]
    assert any(
        edge.from_id == "sym-test-target" and edge.to_id == "sym-target" for edge in test_edges
    )


class _FakeMetadataStore:
    def __init__(self, symbol: Symbol, edges: list[EdgeRecord]) -> None:
        self._symbol = symbol
        self._edges = edges

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        if symbol_id == self._symbol.id:
            return self._symbol
        return None

    def list_edges_for_symbol(
        self,
        symbol_id: str,
        *,
        direction: str = "both",
        edge_type: str | None = None,
        limit: int | None = None,
    ) -> list[EdgeRecord]:
        rows: list[EdgeRecord] = []
        for edge in self._edges:
            if direction == "incoming" and edge.to_id != symbol_id:
                continue
            if direction == "outgoing" and edge.from_id != symbol_id:
                continue
            if direction == "both" and edge.from_id != symbol_id and edge.to_id != symbol_id:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            rows.append(edge)
        rows.sort(key=lambda item: (item.file_path, item.line, item.edge_id))
        if limit is not None:
            return rows[:limit]
        return rows

    def list_edges(self) -> list[EdgeRecord]:
        return list(self._edges)


def _edge(
    *,
    edge_id: str,
    from_id: str,
    to_id: str,
    confidence: float,
    file_path: str,
    line: int,
    commit: str,
    embedding: list[float] | None = None,
) -> EdgeRecord:
    return EdgeRecord(
        edge_id=edge_id,
        edge_type="CALLS",
        from_id=from_id,
        to_id=to_id,
        from_kind="function",
        to_kind="function",
        file_path=file_path,
        line=line,
        confidence=confidence,
        repo_id="repo-1",
        commit=commit,
        text=f"{from_id} -> {to_id}",
        embedding_vector=embedding,
    )


def test_graph_service_neighbors_are_ranked_deterministically() -> None:
    anchor = _symbol(
        symbol_id="sym-anchor",
        name="anchor",
        fqname="pkg.anchor",
        file_path="/repo/src/main.py",
        start_line=1,
        end_line=3,
    )
    edges = [
        _edge(
            edge_id="edge-a",
            from_id="sym-anchor",
            to_id="sym-a",
            confidence=0.8,
            file_path="/repo/src/a.py",
            line=10,
            commit="aaa111",
        ),
        _edge(
            edge_id="edge-b",
            from_id="sym-b",
            to_id="sym-anchor",
            confidence=0.95,
            file_path="/repo/tests/test_main.py",
            line=4,
            commit="bbb222",
        ),
        _edge(
            edge_id="edge-c",
            from_id="sym-anchor",
            to_id="sym-c",
            confidence=0.7,
            file_path="/repo/src/utils/helpers.py",
            line=22,
            commit="bbb222",
        ),
    ]
    service = GraphService(metadata_store=_FakeMetadataStore(anchor, edges))

    payload = service.neighbors("sym-anchor", direction="both", k=3)

    assert [item["edge_id"] for item in payload["results"]] == ["edge-b", "edge-a", "edge-c"]


class _FakeEmbeddingProvider(EmbeddingProvider):
    def embed_text(self, text: str) -> list[float]:
        if "anchor" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        return 2


def test_graph_service_semantic_search_orders_by_similarity() -> None:
    anchor = _symbol(
        symbol_id="sym-anchor",
        name="anchor",
        fqname="pkg.anchor",
        file_path="/repo/src/main.py",
        start_line=1,
        end_line=3,
    )
    edges = [
        _edge(
            edge_id="edge-semantic-best",
            from_id="sym-anchor",
            to_id="sym-a",
            confidence=0.8,
            file_path="/repo/src/a.py",
            line=10,
            commit="aaa111",
            embedding=[1.0, 0.0],
        ),
        _edge(
            edge_id="edge-semantic-worse",
            from_id="sym-anchor",
            to_id="sym-b",
            confidence=0.9,
            file_path="/repo/src/b.py",
            line=20,
            commit="aaa111",
            embedding=[0.0, 1.0],
        ),
    ]
    service = GraphService(
        metadata_store=_FakeMetadataStore(anchor, edges),
        embedding_provider=_FakeEmbeddingProvider(),
    )

    payload = service.search("anchor neighbors", k=2)

    assert [item["edge_id"] for item in payload["results"]] == [
        "edge-semantic-best",
        "edge-semantic-worse",
    ]
