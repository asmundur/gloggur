from __future__ import annotations

import os
from collections.abc import Iterable

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.models import Symbol, SymbolChunk
from gloggur.search.hybrid_search import HybridSearch


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic query-to-vector mapping for ranking tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_text(self, text: str) -> list[float]:
        return list(self._vectors[text])

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        return 2


class FakeVectorStore:
    """Deterministic vector hit ordering keyed by query vector."""

    def __init__(self, hits_by_vector: dict[tuple[float, ...], list[tuple[str, float]]]) -> None:
        self._hits_by_vector = hits_by_vector

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]:
        hits = self._hits_by_vector.get(tuple(query_vector), [])
        return list(hits[:k])


class FakeMetadataStore:
    """In-memory metadata store test double."""

    def __init__(self, symbols: list[Symbol]) -> None:
        self._symbols = {symbol.id: symbol for symbol in symbols}
        self._chunks_by_id: dict[str, SymbolChunk] = {}
        self._chunks_by_symbol_id: dict[str, list[SymbolChunk]] = {}
        for symbol in symbols:
            chunk = SymbolChunk(
                chunk_id=_chunk_id(symbol.id),
                symbol_id=symbol.id,
                chunk_part_index=1,
                chunk_part_total=1,
                text=f"FQNAME: {symbol.name}\n\nBODY:\ndef {symbol.name}():\n    pass",
                file_path=symbol.file_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                language=symbol.language,
                embedding_vector=symbol.embedding_vector,
            )
            self._chunks_by_id[chunk.chunk_id] = chunk
            self._chunks_by_symbol_id[symbol.id] = [chunk]

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        return self._symbols.get(symbol_id)

    def filter_symbols(
        self,
        kinds: list[str] | None = None,
        file_path: str | None = None,
        language: str | None = None,
    ) -> list[Symbol]:
        values = list(self._symbols.values())
        if kinds:
            values = [symbol for symbol in values if symbol.kind in kinds]
        if file_path:
            values = [symbol for symbol in values if symbol.file_path == file_path]
        if language:
            values = [symbol for symbol in values if symbol.language == language]
        return values

    def filter_symbols_by_file_match(
        self,
        *,
        file_path: str,
        kinds: list[str] | None = None,
        language: str | None = None,
    ) -> list[Symbol]:
        values = self.filter_symbols(kinds=kinds, language=language)
        prefix = file_path.strip().rstrip("/\\")
        if not prefix:
            return []
        candidates: list[str] = []
        for candidate in (prefix, os.path.normpath(prefix)):
            normalized = candidate.rstrip("/\\")
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        if not os.path.isabs(prefix):
            dot_candidate = os.path.join(".", prefix)
            for candidate in (dot_candidate, os.path.normpath(dot_candidate)):
                normalized = candidate.rstrip("/\\")
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        matched: list[Symbol] = []
        for symbol in values:
            normalized_path = symbol.file_path.rstrip("/\\")
            for candidate in candidates:
                if (
                    normalized_path == candidate
                    or normalized_path.startswith(f"{candidate}/")
                    or normalized_path.startswith(f"{candidate}\\")
                ):
                    matched.append(symbol)
                    break
        return matched

    def list_symbols(self) -> list[Symbol]:
        return list(self._symbols.values())

    def get_chunk(self, chunk_id: str) -> SymbolChunk | None:
        return self._chunks_by_id.get(chunk_id)

    def list_chunks_for_symbol(self, symbol_id: str) -> list[SymbolChunk]:
        return list(self._chunks_by_symbol_id.get(symbol_id, []))


def _chunk_id(symbol_id: str) -> str:
    return f"{symbol_id}::chunk-1"


def _symbol(
    *,
    symbol_id: str,
    name: str,
    file_path: str,
    start_line: int = 1,
    embedding_vector: list[float] | None = None,
) -> Symbol:
    return Symbol(
        id=symbol_id,
        name=name,
        kind="function",
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + 1,
        signature=f"def {name}()",
        docstring=None,
        body_hash=f"hash-{symbol_id}",
        embedding_vector=embedding_vector or [0.0, 0.0],
        language="python",
    )


def test_identifier_query_prefers_exact_source_symbol_over_higher_similarity_test_hit() -> None:
    src_symbol = _symbol(
        symbol_id="./src/requests/sessions.py:801:mount",
        name="mount",
        file_path="./src/requests/sessions.py",
        start_line=802,
    )
    test_symbol = _symbol(
        symbol_id="./tests/test_sessions.py:20:test_mount_behavior",
        name="test_mount_behavior",
        file_path="./tests/test_sessions.py",
        start_line=21,
    )
    embedding = FakeEmbeddingProvider({"Session.mount": [1.0, 0.0]})
    vector_store = FakeVectorStore(
        {
            (1.0, 0.0): [
                (_chunk_id(test_symbol.id), 0.02),  # higher raw similarity
                (_chunk_id(src_symbol.id), 0.09),
            ]
        }
    )
    metadata_store = FakeMetadataStore([src_symbol, test_symbol])
    searcher = HybridSearch(embedding, vector_store, metadata_store)

    payload = searcher.search("Session.mount", top_k=2)

    assert payload["results"][0]["symbol_id"] == src_symbol.id
    assert payload["results"][0]["ranking_score"] >= payload["results"][0]["similarity_score"]
    metadata = payload["metadata"]
    assert metadata["ranking_mode"] == "balanced"
    assert metadata["query_intent"] == "identifier"
    assert metadata["explicit_test_intent"] is False
    assert metadata["test_penalty_applied"] is True


def test_explicit_test_intent_disables_default_test_penalty() -> None:
    src_symbol = _symbol(
        symbol_id="./src/requests/sessions.py:801:mount",
        name="mount",
        file_path="./src/requests/sessions.py",
        start_line=802,
    )
    test_symbol = _symbol(
        symbol_id="./tests/test_sessions.py:20:test_mount_behavior",
        name="test_mount_behavior",
        file_path="./tests/test_sessions.py",
        start_line=21,
    )
    query = "Session.mount test"
    embedding = FakeEmbeddingProvider({query: [2.0, 0.0]})
    vector_store = FakeVectorStore(
        {
            (2.0, 0.0): [
                (_chunk_id(test_symbol.id), 0.02),  # stays first when penalties are disabled
                (_chunk_id(src_symbol.id), 0.20),
            ]
        }
    )
    metadata_store = FakeMetadataStore([src_symbol, test_symbol])
    searcher = HybridSearch(embedding, vector_store, metadata_store)

    payload = searcher.search(query, top_k=2)

    assert payload["results"][0]["symbol_id"] == test_symbol.id
    metadata = payload["metadata"]
    assert metadata["query_intent"] == "semantic"
    assert metadata["explicit_test_intent"] is True
    assert metadata["test_penalty_applied"] is False


def test_source_first_mode_demotes_tests_for_semantic_queries() -> None:
    src_symbol = _symbol(
        symbol_id="./src/flask/app.py:1223:make_response",
        name="make_response",
        file_path="./src/flask/app.py",
        start_line=1224,
    )
    test_symbol = _symbol(
        symbol_id="./tests/test_json_tag.py:11:test_dump_load_unchanged",
        name="test_dump_load_unchanged",
        file_path="./tests/test_json_tag.py",
        start_line=12,
    )
    query = "response tuple coercion"
    embedding = FakeEmbeddingProvider({query: [3.0, 0.0]})
    vector_store = FakeVectorStore(
        {
            (3.0, 0.0): [
                (_chunk_id(test_symbol.id), 0.02),  # much higher raw similarity than src
                (_chunk_id(src_symbol.id), 0.24),
            ]
        }
    )
    metadata_store = FakeMetadataStore([src_symbol, test_symbol])
    searcher = HybridSearch(embedding, vector_store, metadata_store)

    payload = searcher.search(query, filters={"ranking_mode": "source-first"}, top_k=2)

    assert payload["results"][0]["symbol_id"] == src_symbol.id
    metadata = payload["metadata"]
    assert metadata["ranking_mode"] == "source-first"
    assert metadata["query_intent"] == "semantic"
    assert metadata["test_penalty_applied"] is True


def test_search_ranking_uses_deterministic_file_line_symbol_tiebreaks() -> None:
    symbol_b = _symbol(
        symbol_id="./src/zeta.py:10:alpha",
        name="alpha",
        file_path="./src/zeta.py",
        start_line=10,
    )
    symbol_a = _symbol(
        symbol_id="./src/alpha.py:10:alpha",
        name="alpha",
        file_path="./src/alpha.py",
        start_line=10,
    )
    query = "semantic ranking tie"
    embedding = FakeEmbeddingProvider({query: [4.0, 0.0]})
    vector_store = FakeVectorStore(
        {
            (4.0, 0.0): [
                (_chunk_id(symbol_b.id), 0.20),  # intentionally reverse preferred order
                (_chunk_id(symbol_a.id), 0.20),
            ]
        }
    )
    metadata_store = FakeMetadataStore([symbol_a, symbol_b])
    searcher = HybridSearch(embedding, vector_store, metadata_store)

    payload = searcher.search(query, top_k=2)

    assert [item["symbol_id"] for item in payload["results"]] == [symbol_a.id, symbol_b.id]


def test_file_filter_into_tests_suppresses_test_penalty_even_in_source_first_mode() -> None:
    test_symbol = _symbol(
        symbol_id="./tests/test_api.py:8:test_handles_response",
        name="test_handles_response",
        file_path="./tests/test_api.py",
        start_line=9,
        embedding_vector=[0.0, 0.0],
    )
    query = "handles_response"
    embedding = FakeEmbeddingProvider({query: [0.0, 0.0]})
    vector_store = FakeVectorStore({})
    metadata_store = FakeMetadataStore([test_symbol])
    searcher = HybridSearch(embedding, vector_store, metadata_store)

    payload = searcher.search(
        query,
        filters={"file": "./tests/test_api.py", "ranking_mode": "source-first"},
        top_k=1,
    )

    metadata = payload["metadata"]
    assert metadata["explicit_test_intent"] is True
    assert metadata["test_penalty_applied"] is False
    result = payload["results"][0]
    assert result["ranking_score"] == result["similarity_score"]


def test_file_filter_prefix_matches_src_but_not_src2() -> None:
    src_symbol = _symbol(
        symbol_id="./src/requests/sessions.py:801:mount",
        name="mount",
        file_path="./src/requests/sessions.py",
        start_line=802,
        embedding_vector=[0.0, 0.0],
    )
    src2_symbol = _symbol(
        symbol_id="./src2/requests/sessions.py:801:mount_alt",
        name="mount_alt",
        file_path="./src2/requests/sessions.py",
        start_line=802,
        embedding_vector=[0.0, 0.0],
    )
    query = "Session.mount"
    embedding = FakeEmbeddingProvider({query: [0.0, 0.0]})
    vector_store = FakeVectorStore({})
    metadata_store = FakeMetadataStore([src_symbol, src2_symbol])
    searcher = HybridSearch(embedding, vector_store, metadata_store)

    payload = searcher.search(query, filters={"file": "src"}, top_k=10)

    assert [item["symbol_id"] for item in payload["results"]] == [src_symbol.id]
    metadata = payload["metadata"]
    assert metadata["file_filter"] == "src"
    assert metadata["file_filter_match_mode"] == "exact_or_prefix"
    assert metadata["file_filter_warning_codes"] == []


def test_file_filter_no_match_emits_warning_code() -> None:
    src_symbol = _symbol(
        symbol_id="./src/requests/sessions.py:801:mount",
        name="mount",
        file_path="./src/requests/sessions.py",
        start_line=802,
        embedding_vector=[0.0, 0.0],
    )
    query = "Session.mount"
    embedding = FakeEmbeddingProvider({query: [0.0, 0.0]})
    vector_store = FakeVectorStore({})
    metadata_store = FakeMetadataStore([src_symbol])
    searcher = HybridSearch(embedding, vector_store, metadata_store)

    payload = searcher.search(query, filters={"file": "missing"}, top_k=10)

    assert payload["results"] == []
    metadata = payload["metadata"]
    assert metadata["file_filter"] == "missing"
    assert metadata["file_filter_match_mode"] == "exact_or_prefix"
    assert metadata["file_filter_warning_codes"] == ["file_filter_no_match"]
