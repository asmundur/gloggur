from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from tree_sitter import Node, Tree
from tree_sitter_language_pack import get_parser

from gloggur.models import Symbol
from gloggur.parsers.base import ParsedFile, Parser
from gloggur.parsers.signal_processors import (
    ParserSignalProcessor,
    SignalProcessingOutcome,
    default_signal_processors,
    project_legacy_fields,
)


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
        ["function_declaration", "generator_function_declaration"],
        ["class_declaration"],
        [],
    ),
    "typescript": LanguageSpec(
        "typescript",
        ["function_declaration"],
        ["class_declaration"],
        ["interface_declaration"],
    ),
    "tsx": LanguageSpec(
        "tsx",
        ["function_declaration"],
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


@dataclass(frozen=True)
class _ContainerContext:
    """Container stack frame for fqname/container metadata."""

    symbol_id: str
    fqname: str


class TreeSitterParser(Parser):
    """Tree-sitter parser that extracts Symbol objects from source code."""

    def __init__(
        self,
        language: str,
        signal_processors: list[ParserSignalProcessor] | None = None,
    ) -> None:
        """Initialize a tree-sitter parser for the given language."""
        if language not in _LANGUAGE_SPECS:
            raise ValueError(f"Unsupported language: {language}")
        self.language = language
        self.spec = _LANGUAGE_SPECS[language]
        self.parser = get_parser(language)
        self.signal_processors = signal_processors or default_signal_processors()

    def parse_file(self, path: str, source: str) -> ParsedFile:
        """Parse a file and return ParsedFile with extracted symbols."""
        symbols = self.extract_symbols(path, source)
        return ParsedFile(path=path, language=self.language, source=source, symbols=symbols)

    def parse_with_edit(self, old_tree: Tree | None, new_source: str) -> Tree:
        """Parse source with optional incremental edit support."""
        return self.parser.parse(bytes(new_source, "utf8"), old_tree)

    def extract_symbols(self, path: str, source: str) -> list[Symbol]:
        """Extract symbol records from the syntax tree with container metadata."""
        tree = self.parser.parse(bytes(source, "utf8"))
        repo_id = self._repo_id(path)
        symbols: list[Symbol] = []
        self._collect_symbols(
            node=tree.root_node,
            path=path,
            source=source,
            repo_id=repo_id,
            containers=[],
            out=symbols,
        )
        symbols.sort(key=lambda item: (item.file_path, item.start_line, item.end_line, item.id))
        return symbols

    def get_supported_languages(self) -> Iterable[str]:
        """Return the single language supported by this parser."""
        return [self.language]

    def _collect_symbols(
        self,
        *,
        node: Node,
        path: str,
        source: str,
        repo_id: str,
        containers: list[_ContainerContext],
        out: list[Symbol],
    ) -> None:
        """Depth-first symbol extraction with container stack propagation."""
        if not node.is_named:
            return

        current_container = containers[-1] if containers else None
        symbol = self._symbol_from_node(
            node,
            path=path,
            source=source,
            repo_id=repo_id,
            container=current_container,
        )
        pushed = False
        if symbol is not None:
            out.append(symbol)
            fqname = symbol.fqname or symbol.name
            containers.append(_ContainerContext(symbol_id=symbol.id, fqname=fqname))
            pushed = True

        for child in node.children:
            self._collect_symbols(
                node=child,
                path=path,
                source=source,
                repo_id=repo_id,
                containers=containers,
                out=out,
            )

        if pushed:
            containers.pop()

    def _symbol_from_node(
        self,
        node: Node,
        *,
        path: str,
        source: str,
        repo_id: str,
        container: _ContainerContext | None,
    ) -> Symbol | None:
        """Convert one AST node into a Symbol with fqname/container metadata."""
        kind = self._symbol_kind(node)
        if not kind:
            return None
        name = self._extract_name(node, source)
        if not name:
            return None

        signals = []
        attributes: dict[str, object] = {}
        final_kind = kind
        for processor in self.signal_processors:
            outcome = processor.process(
                language=self.language,
                node=node,
                name=name,
                kind=kind,
                source=source,
            )
            if not isinstance(outcome, SignalProcessingOutcome):
                continue
            if outcome.kind_override:
                final_kind = outcome.kind_override
            if outcome.signals:
                signals.extend(outcome.signals)
            if outcome.attributes:
                attributes.update(outcome.attributes)

        (
            invariants,
            calls,
            is_serialization_boundary,
            implicit_contract,
        ) = project_legacy_fields(signals)

        signature = self._extract_signature(node, source)
        docstring = self._extract_docstring(node, source)
        body = source[node.start_byte : node.end_byte]
        body_hash = hashlib.sha256(body.encode("utf8")).hexdigest()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        fqname = name if container is None else f"{container.fqname}.{name}"
        visibility, exported = self._infer_visibility_and_exported(
            node=node,
            name=name,
            signature=signature,
            source=source,
            container=container,
        )
        symbol_id = self._hash_symbol_id(
            repo_id=repo_id,
            path=path,
            language=self.language,
            kind=final_kind,
            fqname=fqname,
            start_line=start_line,
            end_line=end_line,
        )
        return Symbol(
            id=symbol_id,
            name=name,
            kind=final_kind,
            fqname=fqname,
            file_path=path,
            start_line=start_line,
            end_line=end_line,
            container_id=container.symbol_id if container else None,
            container_fqname=container.fqname if container else None,
            signature=signature,
            docstring=docstring,
            body_hash=body_hash,
            language=self.language,
            repo_id=repo_id,
            commit=None,
            visibility=visibility,
            exported=exported,
            tokens_estimate=self._estimate_tokens(body),
            invariants=invariants,
            calls=calls,
            is_serialization_boundary=is_serialization_boundary,
            implicit_contract=implicit_contract,
            signals=signals,
            attributes=attributes,
        )

    def _symbol_kind(self, node: Node) -> str | None:
        """Classify a node as function/method/class/interface/trait/enum/type."""
        if self._is_decorated_inner_symbol(node):
            return None

        if self.language == "python":
            if node.type == "function_definition":
                parent = node.parent
                if parent and parent.type == "class_definition":
                    return "method"
                return "function"
            if node.type == "decorated_definition":
                child_kind = self._decorated_child_kind(node)
                return child_kind
            if node.type == "class_definition":
                return "class"
            return None

        if self.language in {"javascript", "typescript", "tsx"}:
            if node.type == "method_definition":
                return "method"
            if node.type in self.spec.function_nodes:
                return "function"
            if node.type in self.spec.class_nodes:
                return "class"
            if node.type in self.spec.interface_nodes:
                return "interface"
            return None

        if self.language == "go":
            if node.type == "method_declaration":
                return "method"
            if node.type == "function_declaration":
                return "function"
            if node.type == "interface_type":
                return "interface"
            return None

        if self.language == "rust":
            if node.type == "function_item":
                parent = node.parent
                if parent and parent.type in {"impl_item", "trait_item"}:
                    return "method"
                return "function"
            if node.type == "struct_item":
                return "type"
            if node.type == "enum_item":
                return "enum"
            if node.type == "trait_item":
                return "trait"
            return None

        if self.language == "java":
            if node.type in {"method_declaration", "constructor_declaration"}:
                return "method"
            if node.type == "class_declaration":
                return "class"
            if node.type == "interface_declaration":
                return "interface"
            return None

        if node.type in self.spec.function_nodes:
            return "function"
        if node.type in self.spec.class_nodes:
            return "class"
        if node.type in self.spec.interface_nodes:
            return "interface"
        return None

    def _is_decorated_inner_symbol(self, node: Node) -> bool:
        """Return True when node is the inner declaration under decorated_definition."""
        if self.language != "python":
            return False
        parent = node.parent
        if not parent or parent.type != "decorated_definition":
            return False
        return node.type in {"function_definition", "class_definition"}

    def _decorated_child_kind(self, node: Node) -> str | None:
        """Resolve symbol kind for Python decorated definitions."""
        for child in node.children:
            if child.type == "function_definition":
                parent = child.parent
                if parent and parent.parent and parent.parent.type == "class_definition":
                    return "method"
                return "function"
            if child.type == "class_definition":
                return "class"
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
            if child.type in {
                "identifier",
                "type_identifier",
                "name",
                "property_identifier",
                "field_identifier",
            }:
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

    def _infer_visibility_and_exported(
        self,
        *,
        node: Node,
        name: str,
        signature: str | None,
        source: str,
        container: _ContainerContext | None,
    ) -> tuple[str | None, bool | None]:
        """Infer visibility/export status using language-specific heuristics."""
        _ = container
        sig = (signature or "").strip()
        full_text = source[node.start_byte : min(node.end_byte, node.start_byte + 256)]

        if self.language == "python":
            private = name.startswith("_")
            return ("private" if private else "public", not private)

        if self.language == "go":
            exported = bool(name) and name[0].isupper()
            return ("public" if exported else "private", exported)

        if self.language == "rust":
            exported = bool(re.search(r"\bpub\b", sig))
            return ("public" if exported else "private", exported)

        if self.language in {"javascript", "typescript", "tsx", "java"}:
            lowered = f"{sig} {full_text}".lower()
            if " private " in f" {lowered} ":
                return "private", False
            if " protected " in f" {lowered} ":
                return "protected", False
            if " public " in f" {lowered} ":
                return "public", True
            if " export " in f" {lowered} ":
                return "public", True
            # Best effort fallback: class/module scoped declarations default to public.
            return "public", True

        return None, None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for sizing/ranking constraints."""
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, len(stripped.split()))

    @staticmethod
    def _repo_id(path: str) -> str:
        """Return deterministic repo id hash from nearest .git root or file parent."""
        absolute = os.path.abspath(path)
        current = absolute if os.path.isdir(absolute) else os.path.dirname(absolute)
        while True:
            git_path = os.path.join(current, ".git")
            if os.path.exists(git_path):
                return hashlib.sha256(current.encode("utf8")).hexdigest()
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        fallback = os.path.dirname(absolute) if os.path.isfile(absolute) else absolute
        return hashlib.sha256(fallback.encode("utf8")).hexdigest()

    @staticmethod
    def _hash_symbol_id(
        *,
        repo_id: str,
        path: str,
        language: str,
        kind: str,
        fqname: str,
        start_line: int,
        end_line: int,
    ) -> str:
        """Build stable symbol id from the Spec 1 hash contract."""
        payload = "|".join(
            [
                repo_id,
                os.path.normpath(path),
                language,
                kind,
                fqname,
                str(start_line),
                str(end_line),
            ]
        )
        return hashlib.sha256(payload.encode("utf8")).hexdigest()

    @staticmethod
    def _strip_quotes(value: str) -> str:
        """Strip surrounding quotes from a string literal."""
        if value.startswith(("'''", '"""')) and value.endswith(("'''", '"""')):
            return value[3:-3].strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            return value[1:-1].strip()
        return value.strip()
