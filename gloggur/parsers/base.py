from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional

from gloggur.models import Symbol


@dataclass
class ParsedFile:
    path: str
    language: Optional[str]
    source: str
    symbols: List[Symbol]


class Parser(ABC):
    @abstractmethod
    def parse_file(self, path: str, source: str) -> ParsedFile:
        raise NotImplementedError

    @abstractmethod
    def extract_symbols(self, path: str, source: str) -> List[Symbol]:
        raise NotImplementedError

    @abstractmethod
    def get_supported_languages(self) -> Iterable[str]:
        raise NotImplementedError
