from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from gloggur.models import Symbol


@dataclass
class ParsedFile:
    """Parsed file data: path, language, source, extracted symbols."""

    path: str
    language: str | None
    source: str
    symbols: list[Symbol]


class Parser(ABC):
    """Abstract interface for language parsers."""

    @abstractmethod
    def parse_file(self, path: str, source: str) -> ParsedFile:
        """Parse source into a ParsedFile with symbols."""
        raise NotImplementedError

    @abstractmethod
    def extract_symbols(self, path: str, source: str) -> list[Symbol]:
        """Extract symbols from source without building a ParsedFile."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_languages(self) -> Iterable[str]:
        """Return the set of supported language identifiers."""
        raise NotImplementedError
