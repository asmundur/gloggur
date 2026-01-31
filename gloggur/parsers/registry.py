from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from gloggur.parsers.treesitter_parser import TreeSitterParser


@dataclass(frozen=True)
class ParserEntry:
    parser: TreeSitterParser
    language: str


_EXTENSION_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
}


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: Dict[str, TreeSitterParser] = {}

    def get_parser_for_path(self, path: str) -> Optional[ParserEntry]:
        for ext, language in _EXTENSION_MAP.items():
            if path.endswith(ext):
                if language not in self._parsers:
                    self._parsers[language] = TreeSitterParser(language)
                return ParserEntry(parser=self._parsers[language], language=language)
        return None

    def supported_extensions(self) -> Dict[str, str]:
        return dict(_EXTENSION_MAP)


EXTENSION_LANGUAGE = dict(_EXTENSION_MAP)


def get_parser_for_file(path: str) -> Optional[TreeSitterParser]:
    entry = ParserRegistry().get_parser_for_path(path)
    return entry.parser if entry else None
