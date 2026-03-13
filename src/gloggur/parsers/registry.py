from __future__ import annotations

from dataclasses import dataclass

from gloggur.adapters.registry import AdapterRegistry, adapter_module_override, instantiate_adapter
from gloggur.parsers.base import Parser
from gloggur.parsers.treesitter_parser import TreeSitterParser


@dataclass(frozen=True)
class ParserEntry:
    """Parser entry: parser adapter and language label."""

    parser: Parser
    language: str


_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
}

_PARSER_ADAPTERS = AdapterRegistry[Parser]("gloggur.parsers")
for _language in sorted(set(_EXTENSION_MAP.values())):
    _PARSER_ADAPTERS.register_builtin(
        _language,
        (lambda language: (lambda: TreeSitterParser(language)))(_language),
    )


class ParserRegistry:
    """Registry that maps file extensions to parser adapters."""

    def __init__(
        self,
        *,
        extension_map: dict[str, str] | None = None,
        adapter_overrides: dict[str, object] | None = None,
    ) -> None:
        """Initialize parser cache, extension mapping, and optional overrides."""
        self._parsers: dict[str, Parser] = {}
        self._extension_map = dict(_EXTENSION_MAP)
        if extension_map:
            self._extension_map.update(extension_map)
        self._adapter_overrides = adapter_overrides or {}

    def get_parser_for_path(self, path: str) -> ParserEntry | None:
        """Return a parser entry for a file path extension."""
        for ext, adapter_id in self._extension_map.items():
            if not path.endswith(ext):
                continue
            parser = self._parsers.get(adapter_id)
            if parser is None:
                module_override = adapter_module_override(
                    self._adapter_overrides,
                    category="parsers",
                    adapter_id=adapter_id,
                )
                factory = _PARSER_ADAPTERS.resolve_factory(
                    adapter_id,
                    module_path_override=module_override,
                )
                parser = instantiate_adapter(factory)
                if not all(
                    hasattr(parser, attr)
                    for attr in ("parse_file", "extract_symbols", "get_supported_languages")
                ):
                    raise TypeError(
                        f"Parser adapter '{adapter_id}' returned invalid parser "
                        f"type: {type(parser).__name__}"
                    )
                self._parsers[adapter_id] = parser
            return ParserEntry(parser=parser, language=adapter_id)
        return None

    def supported_extensions(self) -> dict[str, str]:
        """Return the extension-to-language mapping."""
        return dict(self._extension_map)

    @staticmethod
    def available_adapters() -> list[dict[str, object]]:
        """Return discoverable parser adapter descriptors."""
        return _PARSER_ADAPTERS.available()


EXTENSION_LANGUAGE = dict(_EXTENSION_MAP)


def get_parser_for_file(path: str) -> Parser | None:
    """Return a parser instance for a file path, if supported."""
    entry = ParserRegistry().get_parser_for_path(path)
    return entry.parser if entry else None
