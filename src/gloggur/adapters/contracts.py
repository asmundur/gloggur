from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.models import Symbol, SymbolChunk
from gloggur.parsers.base import ParsedFile


class ParserAdapter(Protocol):
    """Contract for file-language parser adapters."""

    def parse_file(self, path: str, source: str) -> ParsedFile:
        """Parse one file into a ParsedFile payload."""

    def extract_symbols(self, path: str, source: str) -> list[Symbol]:
        """Extract symbols for one file payload."""

    def get_supported_languages(self) -> Iterable[str]:
        """Return language identifiers supported by this adapter."""


class CoverageImporterAdapter(Protocol):
    """Contract for coverage import adapters."""

    adapter_id: str

    def import_contexts(self, source_path: str) -> dict[str, dict[str, list[int]]]:
        """Return normalized test-context line mappings."""


class MetadataBackend(Protocol):
    """Read-only metadata backend contract used by search/guidance."""

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        """Fetch one symbol by id."""

    def filter_symbols(
        self,
        kinds: list[str] | None = None,
        file_path: str | None = None,
        language: str | None = None,
    ) -> list[Symbol]:
        """Filter symbols with backend-native query semantics."""

    def list_symbols(self) -> list[Symbol]:
        """Return all symbols."""


class VectorBackend(Protocol):
    """Vector backend contract used by index/search flows."""

    def upsert_vectors(self, symbols: Iterable[Symbol | SymbolChunk]) -> None:
        """Insert/replace vectors for a symbol batch."""

    def remove_ids(self, symbol_ids: Iterable[str]) -> None:
        """Remove vectors by symbol id."""

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]:
        """Return nearest-neighbor symbol ids with backend distance metric."""

    def save(self) -> None:
        """Persist backend state."""

    def clear(self) -> None:
        """Clear backend state."""

    def list_symbol_ids(self) -> list[str]:
        """Return vector symbol ids when deterministic listing is supported."""


class RuntimeHost(Protocol):
    """Runtime host abstraction for watch/bootstrap orchestration."""

    host_id: str

    def build_watch_service(
        self,
        *,
        config,
        embedding_provider: EmbeddingProvider | None,
        cache,
        vector_store: VectorBackend,
        parser_registry,
    ):
        """Create a watch service instance for the active runtime host."""


class AdapterDescriptor(Protocol):
    """Descriptor metadata exposed by adapter registries."""

    adapter_id: str
    source: str
    callable_path: str
    aliases: Sequence[str]
