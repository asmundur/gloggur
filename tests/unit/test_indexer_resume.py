from __future__ import annotations

import tempfile

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import EmbeddingProviderError
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.embedding_ledger import EmbeddingLedger, embedding_text_hash
from gloggur.indexer.indexer import Indexer
from gloggur.models import SymbolChunk
from gloggur.parsers.registry import ParserRegistry


def _make_chunks(total: int) -> list[SymbolChunk]:
    return [
        SymbolChunk(
            chunk_id=f"chunk-{index}",
            symbol_id=f"symbol-{index}",
            chunk_part_index=1,
            chunk_part_total=1,
            text=f"def fn_{index}():\n    return {index}",
            file_path="sample.py",
            start_line=1,
            end_line=2,
        )
        for index in range(total)
    ]


def test_apply_embeddings_reuses_ledger_after_partial_failure() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(cache_dir=cache_dir, embedding_provider="gemini")
    cache = CacheManager(CacheConfig(cache_dir))
    ledger = EmbeddingLedger(cache_dir)

    class InterruptingProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self._chunk_size = 2
            self.calls: list[list[str]] = []

        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.1, 0.2]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            payload = list(texts)
            self.calls.append(payload)
            if len(self.calls) > 1:
                raise RuntimeError("timeout")
            return [[0.1, 0.2] for _ in payload]

        def get_dimension(self) -> int:
            return 2

    first_provider = InterruptingProvider()
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=first_provider,
        embedding_ledger=ledger,
    )
    chunks = _make_chunks(4)

    with pytest.raises(EmbeddingProviderError):
        indexer._apply_embeddings(chunks)

    first_hashes = [embedding_text_hash(chunk.text) for chunk in chunks[:2]]
    assert ledger.get_vectors(
        embedding_profile=config.embedding_profile(),
        record_kind="chunk",
        text_hashes=first_hashes,
    ) == {
        first_hashes[0]: [0.1, 0.2],
        first_hashes[1]: [0.1, 0.2],
    }

    class RecordingProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self._chunk_size = 2
            self.calls: list[list[str]] = []

        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.3, 0.4]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            payload = list(texts)
            self.calls.append(payload)
            return [[0.3, 0.4] for _ in payload]

        def get_dimension(self) -> int:
            return 2

    second_provider = RecordingProvider()
    second_indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=second_provider,
        embedding_ledger=ledger,
    )
    resumed_chunks = second_indexer._apply_embeddings(_make_chunks(4))

    assert second_provider.calls == [[chunk.text for chunk in _make_chunks(4)[2:]]]
    assert resumed_chunks[0].embedding_vector == [0.1, 0.2]
    assert resumed_chunks[1].embedding_vector == [0.1, 0.2]
    assert resumed_chunks[2].embedding_vector == [0.3, 0.4]
    assert resumed_chunks[3].embedding_vector == [0.3, 0.4]
