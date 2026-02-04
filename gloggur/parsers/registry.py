from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from gloggur.parsers.treesitter_parser import TreeSitterParser


@dataclass(frozen=True)
class ParserEntry:
    """Pairing of parser and language label."""
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
    """Registry for mapping file extensions to parsers."""
    def __init__(self) -> None:
        """Initialize an empty parser cache."""
        self._parsers: Dict[str, TreeSitterParser] = {}

    def get_parser_for_path(self, path: str) -> Optional[ParserEntry]:
        """Return a parser entry based on the file extension."""
        for ext, language in _EXTENSION_MAP.items():
            if path.endswith(ext):
                if language not in self._parsers:
                    self._parsers[language] = TreeSitterParser(language)
                return ParserEntry(parser=self._parsers[language], language=language)
        return None

    def supported_extensions(self) -> Dict[str, str]:
        """Return the extension-to-language mapping."""
        return dict(_EXTENSION_MAP)


EXTENSION_LANGUAGE = dict(_EXTENSION_MAP)


def get_parser_for_file(path: str) -> Optional[TreeSitterParser]:
    """Return a parser instance for a file path, if supported."""
    entry = ParserRegistry().get_parser_for_path(path)
    return entry.parser if entry else None
