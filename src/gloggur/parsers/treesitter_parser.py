from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from tree_sitter import Node, Tree
from tree_sitter_language_pack import get_parser

from gloggur.models import Symbol
from gloggur.parsers.base import ParsedFile, Parser


@dataclass(frozen=True)
class LanguageSpec:
    """Tree-sitter node type configuration for functions/classes/interfaces."""

    name: str
    function_nodes: list[str]
    class_nodes: list[str]
    interface_nodes: list[str]


_LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        "python",
        ["function_definition", "decorated_definition"],
        ["class_definition"],
        [],
    ),
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
    """Tree-sitter parser that extracts Symbol objects from source code."""

    def __init__(self, language: str) -> None:
        """Initialize a tree-sitter parser for the given language."""
        if language not in _LANGUAGE_SPECS:
            raise ValueError(f"Unsupported language: {language}")
        self.language = language
        self.spec = _LANGUAGE_SPECS[language]
        self.parser = get_parser(language)

    def parse_file(self, path: str, source: str) -> ParsedFile:
        """Parse a file and return ParsedFile with extracted symbols."""
        symbols = self.extract_symbols(path, source)
        return ParsedFile(path=path, language=self.language, source=source, symbols=symbols)

    def parse_with_edit(self, old_tree: Tree | None, new_source: str) -> Tree:
        """Parse source with optional incremental edit support."""
        return self.parser.parse(bytes(new_source, "utf8"), old_tree)

    def extract_symbols(self, path: str, source: str) -> list[Symbol]:
        """Extract function/class/interface symbols from the AST."""
        tree = self.parser.parse(bytes(source, "utf8"))
        symbols: list[Symbol] = []
        for node in self._walk(tree.root_node):
            if not node.is_named:
                continue
            symbol = self._symbol_from_node(node, path, source)
            if symbol:
                symbols.append(symbol)
        return symbols

    def get_supported_languages(self) -> Iterable[str]:
        """Return the single language supported by this parser."""
        return [self.language]

    def _symbol_from_node(self, node: Node, path: str, source: str) -> Symbol | None:
        """Convert a matching AST node into a Symbol with metadata."""
        kind = self._symbol_kind(node, source)
        if not kind:
            return None
        name = self._extract_name(node, source)
        if not name:
            return None
        signature = self._extract_signature(node, source)
        docstring = self._extract_docstring(node, source)
        invariants = self._extract_invariants(node, source)
        calls = self._extract_call_graph(node, source)
        is_serialization_boundary = self._detect_serialization(node, name, source)

        implicit_contract = None
        if kind == "function" and name.startswith("test_"):
            implicit_contract = name[5:].replace("_", " ").strip()

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
            invariants=invariants,
            calls=calls,
            is_serialization_boundary=is_serialization_boundary,
            implicit_contract=implicit_contract,
        )

    def _symbol_kind(self, node: Node, source: str) -> str | None:
        """Classify a node as function, class, or interface."""
        if node.type in self.spec.function_nodes:
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "decorator":
                        text = source[child.start_byte : child.end_byte]
                        if "fixture" in text:
                            return "fixture"
            return "function"
        if node.type in self.spec.class_nodes:
            return "class"
        if node.type in self.spec.interface_nodes:
            return "interface"
        return None

    def _extract_name(self, node: Node, source: str) -> str | None:
        """Extract the symbol name from a node."""
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in self.spec.function_nodes and child.type != "decorated_definition":
                    return self._extract_name(child, source)
                if child.type in self.spec.class_nodes:
                    return self._extract_name(child, source)

        for child in node.children:
            if child.type in {"identifier", "type_identifier", "name", "property_identifier"}:
                return source[child.start_byte : child.end_byte]
        return None

    def _extract_signature(self, node: Node, source: str) -> str | None:
        """Extract the first-line signature for a symbol."""
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in self.spec.function_nodes and child.type != "decorated_definition":
                    return self._extract_signature(child, source)
                if child.type in self.spec.class_nodes:
                    return self._extract_signature(child, source)

        text = source[node.start_byte : node.end_byte]
        first_line = text.splitlines()[0] if text else ""
        return first_line.strip() or None

    def _extract_docstring(self, node: Node, source: str) -> str | None:
        """Extract a docstring or nearby comment for a symbol."""
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in self.spec.function_nodes and child.type != "decorated_definition":
                    return self._extract_docstring(child, source)
                if child.type in self.spec.class_nodes:
                    return self._extract_docstring(child, source)

        if self.language == "python":
            body = None
            for child in node.named_children:
                if child.type == "block":
                    body = child
                    break
            if body and body.named_child_count:
                first = body.named_children[0]
                if first.type in {"string", "string_literal"}:
                    return self._strip_quotes(source[first.start_byte : first.end_byte])
                if first.type == "expression_statement" and first.named_child_count:
                    expr = first.named_children[0]
                    if expr.type in {"string", "string_literal"}:
                        return self._strip_quotes(source[expr.start_byte : expr.end_byte])
        prev = node.prev_named_sibling
        if prev and "comment" in prev.type:
            return source[prev.start_byte : prev.end_byte].strip()
        return None

    def _walk(self, node: Node) -> Iterable[Node]:
        """Yield a depth-first traversal of the syntax tree."""
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _extract_invariants(self, node: Node, source: str) -> list[str]:
        """Extract invariant comparisons from assert statements."""
        invariants = []
        for child in self._walk(node):
            if child.type == "assert_statement":
                text = source[child.start_byte : child.end_byte].strip()
                if text.startswith("assert "):
                    text = text[7:].strip()
                invariants.append(text)
        return invariants

    def _extract_call_graph(self, node: Node, source: str) -> list[str]:
        """Extract static function calls made within this node's body."""
        calls: list[str] = []
        for child in self._walk(node):
            if child.type in {"call", "call_expression"}:
                if child.named_child_count > 0:
                    func_node = child.named_children[0]
                    text = source[func_node.start_byte : func_node.end_byte]
                    calls.append(text)
        # Preserve order but remove duplicates
        return list(dict.fromkeys(calls))

    def _detect_serialization(self, node: Node, name: str, source: str) -> bool:
        """Heuristically identify if this symbol acts as a serialization boundary."""
        name_lower = name.lower()
        keywords = [
            "serialize",
            "deserialize",
            "to_dict",
            "from_dict",
            "to_json",
            "from_json",
            "parse",
        ]
        if any(k in name_lower for k in keywords):
            return True

        for child in self._walk(node):
            if child.type == "call" or child.type == "call_expression":
                # Find the function being called, typically the first child
                if child.named_child_count > 0:
                    func_node = child.named_children[0]
                    text = source[func_node.start_byte : func_node.end_byte]
                    if "json.dump" in text or "json.load" in text:
                        return True
        return False

    @staticmethod
    def _strip_quotes(value: str) -> str:
        """Strip surrounding quotes from a string literal."""
        if value.startswith(("'''", '"""')) and value.endswith(("'''", '"""')):
            return value[3:-3].strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            return value[1:-1].strip()
        return value.strip()
