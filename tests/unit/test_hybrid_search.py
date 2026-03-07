from __future__ import annotations

from pathlib import Path

import pytest

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import EmbeddingProviderError
from gloggur.models import Symbol, SymbolChunk
from gloggur.search.hybrid_search import HybridSearch


class FakeEmbeddingProvider(EmbeddingProvider):
    provider = "test"

    def __init__(
        self,
        *,
        vectors: dict[str, list[float]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.vectors = vectors or {}
        self.error = error

    def embed_text(self, text: str) -> list[float]:
        if self.error is not None:
            raise self.error
        return list(self.vectors.get(text, [0.0, 0.0]))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        return 2


class FakeVectorStore:
    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self.hits = hits
        self.calls: list[dict[str, object]] = []

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]:
        self.calls.append({"query_vector": list(query_vector), "k": k})
        return list(self.hits[:k])


class FakeMetadataStore:
    def __init__(
        self,
        *,
        symbols: list[Symbol],
        chunks: list[SymbolChunk],
    ) -> None:
        self._symbols = {symbol.id: symbol for symbol in symbols}
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._chunks_by_symbol: dict[str, list[SymbolChunk]] = {}
        for chunk in chunks:
            self._chunks_by_symbol.setdefault(chunk.symbol_id, []).append(chunk)

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        return self._symbols.get(symbol_id)

    def filter_symbols(
        self,
        kinds: list[str] | None = None,
        file_path: str | None = None,
        language: str | None = None,
    ) -> list[Symbol]:
        matches = list(self._symbols.values())
        if kinds:
            matches = [symbol for symbol in matches if symbol.kind in kinds]
        if file_path:
            matches = [symbol for symbol in matches if symbol.file_path == file_path]
        if language:
            matches = [symbol for symbol in matches if symbol.language == language]
        return sorted(matches, key=lambda symbol: (symbol.file_path, symbol.start_line, symbol.id))

    def filter_symbols_by_file_match(
        self,
        *,
        file_path: str,
        kinds: list[str] | None = None,
        language: str | None = None,
    ) -> list[Symbol]:
        prefix = file_path.rstrip("/\\")
        matches = []
        for symbol in self.filter_symbols(kinds=kinds, language=language):
            if symbol.file_path == prefix or symbol.file_path.startswith(f"{prefix}/"):
                matches.append(symbol)
        return matches

    def get_chunk(self, chunk_id: str) -> SymbolChunk | None:
        return self._chunks.get(chunk_id)

    def list_chunks_for_symbol(self, symbol_id: str) -> list[SymbolChunk]:
        return list(self._chunks_by_symbol.get(symbol_id, []))


def _symbol(
    symbol_id: str,
    *,
    name: str,
    file_path: str,
    start_line: int,
    embedding_vector: list[float] | None = None,
) -> Symbol:
    return Symbol(
        id=symbol_id,
        name=name,
        kind="function",
        fqname=f"pkg.{name}",
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + 1,
        body_hash=f"hash-{symbol_id}",
        signature=f"def {name}():",
        language="python",
        embedding_vector=embedding_vector,
    )


def _chunk(
    chunk_id: str,
    *,
    symbol_id: str,
    file_path: str,
    start_line: int,
    text: str = "",
    embedding_vector: list[float] | None = None,
) -> SymbolChunk:
    return SymbolChunk(
        chunk_id=chunk_id,
        symbol_id=symbol_id,
        chunk_part_index=0,
        chunk_part_total=1,
        text=text,
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + 1,
        embedding_vector=embedding_vector,
        language="python",
    )


def test_semantic_search_deduplicates_chunk_hits_and_applies_ranking(tmp_path: Path) -> None:
    src_file = tmp_path / "src" / "service.py"
    test_file = tmp_path / "tests" / "test_service.py"
    src_file.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    src_file.write_text("def add():\n    return 1\n", encoding="utf8")
    test_file.write_text("def test_add():\n    assert add() == 1\n", encoding="utf8")
    src_symbol = _symbol("sym-src", name="add", file_path=str(src_file), start_line=1)
    test_symbol = _symbol("sym-test", name="test_add", file_path=str(test_file), start_line=1)
    searcher = HybridSearch(
        FakeEmbeddingProvider(vectors={"pkg.service.add": [0.0, 0.0]}),
        FakeVectorStore(
            [
                ("chunk-src-1", 0.2),
                ("chunk-src-2", 0.1),
                ("chunk-test-1", 0.05),
            ]
        ),
        FakeMetadataStore(
            symbols=[src_symbol, test_symbol],
            chunks=[
                _chunk("chunk-src-1", symbol_id="sym-src", file_path=str(src_file), start_line=1),
                _chunk("chunk-src-2", symbol_id="sym-src", file_path=str(src_file), start_line=1),
                _chunk(
                    "chunk-test-1", symbol_id="sym-test", file_path=str(test_file), start_line=1
                ),
            ],
        ),
    )

    payload = searcher.search(
        "pkg.service.add",
        filters={"ranking_mode": "source-first"},
        top_k=2,
        context_radius=1,
    )

    assert [item["symbol_id"] for item in payload["results"]] == ["sym-src", "sym-test"]
    assert payload["results"][0]["chunk_id"] == "chunk-src-2"
    assert payload["metadata"]["ranking_mode"] == "source-first"
    assert payload["metadata"]["query_intent"] == "identifier"
    assert payload["metadata"]["total_results"] == 2
    assert payload["metadata"]["warning_codes"] == [
        "legacy_search_contract",
        "legacy_search_surface",
    ]


def test_semantic_search_uses_metadata_fallback_when_vector_hits_are_empty(tmp_path: Path) -> None:
    source_file = tmp_path / "src" / "service.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("def alpha():\n    return 1\n", encoding="utf8")
    symbol = _symbol("sym-alpha", name="alpha", file_path="src/service.py", start_line=1)
    searcher = HybridSearch(
        FakeEmbeddingProvider(vectors={"alpha": [0.0, 0.0]}),
        FakeVectorStore([]),
        FakeMetadataStore(
            symbols=[symbol],
            chunks=[
                _chunk(
                    "chunk-alpha",
                    symbol_id="sym-alpha",
                    file_path="src/service.py",
                    start_line=1,
                    embedding_vector=[0.0, 0.0],
                )
            ],
        ),
    )

    payload = searcher.search(
        "alpha",
        filters={"file": "src/service.py", "kind": "function"},
        top_k=1,
    )

    assert [item["symbol_id"] for item in payload["results"]] == ["sym-alpha"]
    assert payload["results"][0]["similarity_score"] == pytest.approx(1.0)
    assert payload["metadata"]["file_filter"] == "src/service.py"
    assert payload["metadata"]["file_filter_warning_codes"] == []


def test_semantic_search_reports_file_filter_warning_when_no_candidates_match() -> None:
    searcher = HybridSearch(
        FakeEmbeddingProvider(vectors={"needle": [0.0, 0.0]}),
        FakeVectorStore([]),
        FakeMetadataStore(symbols=[], chunks=[]),
    )

    payload = searcher.search("needle", filters={"file": "missing.py"}, top_k=3)

    assert payload["results"] == []
    assert payload["metadata"]["file_filter"] == "missing.py"
    assert payload["metadata"]["file_filter_warning_codes"] == ["file_filter_no_match"]


def test_semantic_search_blocks_when_health_requires_reindex() -> None:
    searcher = HybridSearch(
        FakeEmbeddingProvider(vectors={"needle": [0.0, 0.0]}),
        FakeVectorStore([]),
        FakeMetadataStore(symbols=[], chunks=[]),
        health_evaluator=lambda: {
            "entrypoint": "hybrid_search_legacy",
            "contract_version": "legacy",
            "needs_reindex": True,
            "resume_reason_codes": ["missing_index_metadata"],
            "warning_codes": ["reindex_required"],
            "semantic_search_allowed": True,
            "search_integrity": {"status": "blocked"},
        },
    )

    payload = searcher.search("needle")

    assert payload["results"] == []
    assert payload["metadata"]["needs_reindex"] is True
    assert payload["metadata"]["resume_reason_codes"] == ["missing_index_metadata"]
    assert payload["metadata"]["warning_codes"] == ["reindex_required"]


def test_semantic_search_disabled_returns_blocked_payload_with_explicit_warning() -> None:
    searcher = HybridSearch(
        FakeEmbeddingProvider(vectors={"needle": [0.0, 0.0]}),
        FakeVectorStore([]),
        FakeMetadataStore(symbols=[], chunks=[]),
        health_evaluator=lambda: {
            "entrypoint": "hybrid_search_legacy",
            "contract_version": "legacy",
            "needs_reindex": False,
            "resume_reason_codes": [],
            "warning_codes": ["legacy_search_surface"],
            "semantic_search_allowed": False,
            "search_integrity": {"status": "degraded"},
        },
    )

    payload = searcher.search("needle")

    assert payload["results"] == []
    assert payload["metadata"]["semantic_search_allowed"] is False
    assert "semantic_search_disabled" in payload["metadata"]["warning_codes"]
    assert "Prefer `gloggur search --json`" in payload["metadata"]["deprecation_notice"]


@pytest.mark.parametrize(
    ("filters", "query", "expected_ids"),
    [
        ({"mode": "by_fqname"}, "pkg.alpha", ["sym-alpha"]),
        ({"mode": "by_fqname_regex"}, r"pkg\.(alpha|beta)", ["sym-alpha", "sym-beta"]),
        ({"mode": "by_path", "file": "src"}, "", ["sym-alpha", "sym-beta"]),
        ({"mode": "unknown"}, "pkg.alpha", []),
        ({"mode": "by_fqname_regex"}, "(", []),
    ],
)
def test_structured_search_modes_are_deterministic(
    filters: dict[str, str],
    query: str,
    expected_ids: list[str],
) -> None:
    alpha = _symbol("sym-alpha", name="alpha", file_path="src/a.py", start_line=1)
    beta = _symbol("sym-beta", name="beta", file_path="src/b.py", start_line=4)
    searcher = HybridSearch(
        FakeEmbeddingProvider(vectors={}),
        FakeVectorStore([]),
        FakeMetadataStore(
            symbols=[alpha, beta],
            chunks=[
                _chunk("chunk-alpha-1", symbol_id="sym-alpha", file_path="src/a.py", start_line=1),
                _chunk("chunk-alpha-2", symbol_id="sym-alpha", file_path="src/a.py", start_line=2),
                _chunk("chunk-beta-1", symbol_id="sym-beta", file_path="src/b.py", start_line=4),
            ],
        ),
    )
    alpha = alpha.model_copy(update={"fqname": "pkg.alpha"})
    beta = beta.model_copy(update={"fqname": "pkg.beta"})
    searcher.metadata_store = FakeMetadataStore(
        symbols=[alpha, beta],
        chunks=[
            _chunk("chunk-alpha-1", symbol_id="sym-alpha", file_path="src/a.py", start_line=1),
            _chunk("chunk-alpha-2", symbol_id="sym-alpha", file_path="src/a.py", start_line=2),
            _chunk("chunk-beta-1", symbol_id="sym-beta", file_path="src/b.py", start_line=4),
        ],
    )

    payload = searcher.search(query, filters=filters, top_k=5)

    assert [item["symbol_id"] for item in payload["results"]] == expected_ids
    assert payload["metadata"]["search_mode"] == str(filters["mode"]).lower()


def test_search_wraps_embedding_provider_failures() -> None:
    searcher = HybridSearch(
        FakeEmbeddingProvider(error=RuntimeError("offline provider")),
        FakeVectorStore([]),
        FakeMetadataStore(symbols=[], chunks=[]),
    )

    with pytest.raises(EmbeddingProviderError) as error:
        searcher.search("needle")

    assert error.value.provider == "test"
    assert error.value.operation == "embed query for search"
    assert "offline provider" in error.value.detail
