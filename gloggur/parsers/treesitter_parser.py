from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from tree_sitter import Node, Tree
from tree_sitter_language_pack import get_parser

from gloggur.models import Symbol
from gloggur.parsers.base import ParsedFile, Parser


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    function_nodes: List[str]
    class_nodes: List[str]
    interface_nodes: List[str]


_LANGUAGE_SPECS: Dict[str, LanguageSpec] = {
    "python": LanguageSpec("python", ["function_definition"], ["class_definition"], []),
    "javascript": LanguageSpec(
        "javascript",
        ["function_declaration", "method_definition", "generator_function_declaration"],
        ["class_declaration"],
        [],
    ),
    "typescript": LanguageSpec(
        "typescript",
        ["function_declaration", "method_definition"],
        ["class_declaration"],
        ["interface_declaration"],
    ),
    "tsx": LanguageSpec(
        "tsx",
        ["function_declaration", "method_definition"],
        ["class_declaration"],
        ["interface_declaration"],
    ),
    "rust": LanguageSpec(
        "rust",
        ["function_item"],
        ["struct_item", "enum_item"],
        ["trait_item"],
    ),
    "go": LanguageSpec(
        "go",
        ["function_declaration", "method_declaration"],
        [],
        ["interface_type"],
    ),
    "java": LanguageSpec(
        "java",
        ["method_declaration", "constructor_declaration"],
        ["class_declaration"],
        ["interface_declaration"],
    ),
}


class TreeSitterParser(Parser):
    def __init__(self, language: str) -> None:
        if language not in _LANGUAGE_SPECS:
            raise ValueError(f"Unsupported language: {language}")
        self.language = language
        self.spec = _LANGUAGE_SPECS[language]
        self.parser = get_parser(language)

    def parse_file(self, path: str, source: str) -> ParsedFile:
        symbols = self.extract_symbols(path, source)
        return ParsedFile(path=path, language=self.language, source=source, symbols=symbols)

    def parse_with_edit(self, old_tree: Optional[Tree], new_source: str) -> Tree:
        return self.parser.parse(bytes(new_source, "utf8"), old_tree)

    def extract_symbols(self, path: str, source: str) -> List[Symbol]:
        tree = self.parser.parse(bytes(source, "utf8"))
        symbols: List[Symbol] = []
        for node in self._walk(tree.root_node):
            if not node.is_named:
                continue
            symbol = self._symbol_from_node(node, path, source)
            if symbol:
                symbols.append(symbol)
        return symbols

    def get_supported_languages(self) -> Iterable[str]:
        return [self.language]

    def _symbol_from_node(self, node: Node, path: str, source: str) -> Optional[Symbol]:
        kind = self._symbol_kind(node)
        if not kind:
            return None
        name = self._extract_name(node, source)
        if not name:
            return None
        signature = self._extract_signature(node, source)
        docstring = self._extract_docstring(node, source)
        body = source[node.start_byte : node.end_byte]
        body_hash = hashlib.sha256(body.encode("utf8")).hexdigest()
        symbol_id = f"{path}:{node.start_point[0]}:{name}"
        return Symbol(
            id=symbol_id,
            name=name,
            kind=kind,
            file_path=path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            signature=signature,
            docstring=docstring,
            body_hash=body_hash,
            language=self.language,
        )

    def _symbol_kind(self, node: Node) -> Optional[str]:
        if node.type in self.spec.function_nodes:
            return "function"
        if node.type in self.spec.class_nodes:
            return "class"
        if node.type in self.spec.interface_nodes:
            return "interface"
        return None

    def _extract_name(self, node: Node, source: str) -> Optional[str]:
        for child in node.children:
            if child.type in {"identifier", "type_identifier", "name", "property_identifier"}:
                return source[child.start_byte : child.end_byte]
        return None

    def _extract_signature(self, node: Node, source: str) -> Optional[str]:
        text = source[node.start_byte : node.end_byte]
        first_line = text.splitlines()[0] if text else ""
        return first_line.strip() or None

    def _extract_docstring(self, node: Node, source: str) -> Optional[str]:
        if self.language == "python":
            if node.child_count:
                body = node.children[-1]
                if body.type == "block" and body.child_count:
                    first = body.children[0]
                    if first.type == "expression_statement" and first.child_count:
                        expr = first.children[0]
                        if expr.type == "string":
                            return self._strip_quotes(source[expr.start_byte : expr.end_byte])
        prev = node.prev_named_sibling
        if prev and "comment" in prev.type:
            return source[prev.start_byte : prev.end_byte].strip()
        return None

    def _walk(self, node: Node) -> Iterable[Node]:
        yield node
        for child in node.children:
            yield from self._walk(child)

    @staticmethod
    def _strip_quotes(value: str) -> str:
        if value.startswith(("'''", '"""')) and value.endswith(("'''", '"""')):
            return value[3:-3].strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            return value[1:-1].strip()
        return value.strip()
