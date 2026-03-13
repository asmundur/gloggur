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
    "c": LanguageSpec(
        "c",
        ["function_definition", "declaration"],
        ["struct_specifier", "union_specifier", "enum_specifier"],
        [],
    ),
    "cpp": LanguageSpec(
        "cpp",
        ["function_definition", "declaration", "field_declaration"],
        ["class_specifier", "struct_specifier", "enum_specifier"],
        [],
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

_IDENTIFIER_PROPERTY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class _ContainerContext:
    """Container stack frame for fqname/container metadata."""

    symbol_id: str | None
    fqname: str


@dataclass(frozen=True)
class _SyntheticSymbolSpec:
    """Resolved synthetic symbol metadata for JS-family assignment patterns."""

    kind: str
    name: str
    container: _ContainerContext | None
    attributes: dict[str, object]
    fqname: str | None = None


@dataclass(frozen=True)
class _AssignmentTarget:
    """Resolved assignment target metadata for one synthetic symbol."""

    name: str
    container: _ContainerContext | None
    binding_style: str
    fqname: str | None = None


@dataclass(frozen=True)
class _CFamilyDeclaratorInfo:
    """Resolved C-family declarator metadata for symbol extraction/classification."""

    symbol_name: str | None
    qualified_parts: tuple[str, ...]
    is_callable_declaration: bool


@dataclass(frozen=True)
class _CppMacroPattern:
    """Recoverable C++ macro expansion pattern for method/function declarations."""

    macro_name: str
    kind: str
    name_index: int
    owner_index: int | None
    parameter_count: int


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
        self._js_object_owner_aliases: dict[str, tuple[str, ...]] = {}
        self._cpp_macro_patterns: dict[str, _CppMacroPattern] = {}

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
        if self.language in {"javascript", "typescript", "tsx"}:
            self._js_object_owner_aliases = self._collect_js_object_owner_aliases(
                tree.root_node,
                source,
            )
        else:
            self._js_object_owner_aliases = {}
        if self.language == "cpp":
            self._cpp_macro_patterns = self._collect_cpp_macro_patterns(
                tree.root_node,
                source,
            )
        else:
            self._cpp_macro_patterns = {}
        symbols: list[Symbol] = []
        try:
            self._collect_symbols(
                node=tree.root_node,
                path=path,
                source=source,
                repo_id=repo_id,
                containers=[],
                out=symbols,
            )
        finally:
            self._js_object_owner_aliases = {}
            self._cpp_macro_patterns = {}
        if self.language == "cpp":
            symbols = self._drop_cpp_macro_duplicates(symbols)
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
        symbols_for_node = self._synthetic_symbols_from_node(
            node,
            path=path,
            source=source,
            repo_id=repo_id,
            current_container=current_container,
        )
        if not symbols_for_node:
            base_symbol = self._symbol_from_node(
                node,
                path=path,
                source=source,
                repo_id=repo_id,
                container=current_container,
            )
            symbols_for_node = [] if base_symbol is None else [base_symbol]
        pushed = False
        pushed_container: _ContainerContext | None = None
        if symbols_for_node:
            out.extend(symbols_for_node)
            if len(symbols_for_node) == 1:
                symbol = symbols_for_node[0]
                fqname = symbol.fqname or symbol.name
                pushed_container = _ContainerContext(symbol_id=symbol.id, fqname=fqname)
                pushed = True
        elif self.language == "cpp":
            pushed_container = self._cpp_namespace_container(
                node=node,
                source=source,
                current_container=current_container,
            )
            pushed = pushed_container is not None

        if pushed_container is not None:
            containers.append(pushed_container)

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
        kind = self._symbol_kind(node, source)
        if not kind:
            return None
        name = self._extract_name(node, source)
        if not name:
            return None
        effective_container = container
        if self.language == "cpp" and kind == "method":
            qualified_container = self._cpp_method_container_from_qualified_declarator(node, source)
            if qualified_container is not None:
                effective_container = qualified_container

        return self._build_symbol(
            node=node,
            path=path,
            source=source,
            repo_id=repo_id,
            kind=kind,
            name=name,
            container=effective_container,
        )

    def _synthetic_symbols_from_node(
        self,
        node: Node,
        *,
        path: str,
        source: str,
        repo_id: str,
        current_container: _ContainerContext | None,
    ) -> list[Symbol]:
        """Create JS-family synthetic symbols for assignment-bound declarations."""
        specs = self._synthetic_symbol_specs(node, source, current_container)
        return [
            self._build_symbol(
                node=node,
                path=path,
                source=source,
                repo_id=repo_id,
                kind=spec.kind,
                name=spec.name,
                container=spec.container,
                extra_attributes=spec.attributes,
                fqname_override=spec.fqname,
            )
            for spec in specs
        ]

    def _build_symbol(
        self,
        *,
        node: Node,
        path: str,
        source: str,
        repo_id: str,
        kind: str,
        name: str,
        container: _ContainerContext | None,
        extra_attributes: dict[str, object] | None = None,
        fqname_override: str | None = None,
    ) -> Symbol:
        """Build one symbol object with shared metadata/signals plumbing."""

        signals = []
        attributes: dict[str, object] = dict(extra_attributes or {})
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
        fqname = (
            fqname_override
            if fqname_override is not None
            else (name if container is None else f"{container.fqname}.{name}")
        )
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

    def _synthetic_symbol_specs(
        self,
        node: Node,
        source: str,
        current_container: _ContainerContext | None,
    ) -> list[_SyntheticSymbolSpec]:
        """Return synthetic symbol specs for supported JS-family binding patterns."""
        if self.language == "cpp":
            return self._cpp_macro_symbol_specs(
                node=node,
                source=source,
                current_container=current_container,
            )
        if self.language not in {"javascript", "typescript", "tsx"}:
            return []

        if node.type == "method_definition":
            if node.parent is None or node.parent.type != "object":
                return []
            owners = self._bound_object_owners(node.parent, source)
            if not owners:
                return []
            property_name = self._property_name(node.child_by_field_name("name"), source)
            if not property_name:
                return []
            attributes: dict[str, object] = {"binding_style": "object_binding_property"}
            accessor = self._method_definition_accessor(node, source)
            if accessor is not None:
                attributes["accessor"] = accessor
            return [
                _SyntheticSymbolSpec(
                    kind="method",
                    name=property_name,
                    container=owner,
                    attributes=dict(attributes),
                )
                for owner in owners
            ]

        if node.type not in {"function_expression", "arrow_function", "class_expression"}:
            return []

        parent = node.parent
        if parent is None:
            return []

        if parent.type == "pair" and parent.child_by_field_name("value") == node:
            define_property_specs = self._define_property_symbol_specs(
                pair=parent,
                value_node=node,
                source=source,
            )
            if define_property_specs:
                return define_property_specs
            owners = self._bound_object_owners(parent.parent, source)
            if not owners:
                return []
            property_name = self._property_name(parent.child_by_field_name("key"), source)
            if not property_name:
                return []
            if node.type == "class_expression":
                return []
            return [
                _SyntheticSymbolSpec(
                    kind="method",
                    name=property_name,
                    container=owner,
                    attributes={"binding_style": "object_binding_property"},
                )
                for owner in owners
            ]

        targets = self._assignment_symbol_targets(
            node,
            source,
            current_container=current_container,
        )
        return [
            _SyntheticSymbolSpec(
                kind=self._synthetic_assignment_kind(node=node, binding_style=target.binding_style),
                name=target.name,
                container=target.container,
                attributes={"binding_style": target.binding_style},
                fqname=target.fqname,
            )
            for target in targets
        ]

    def _assignment_symbol_targets(
        self,
        value_node: Node,
        source: str,
        *,
        current_container: _ContainerContext | None,
    ) -> list[_AssignmentTarget]:
        """Resolve all recoverable binding targets in an assignment/declarator chain."""
        intrinsic_name = self._extract_name(value_node, source)
        targets: list[_AssignmentTarget] = []
        seen: set[_AssignmentTarget] = set()
        for target_node in self._binding_target_nodes_for_value(value_node):
            for target in self._assignment_symbol_targets_for_node(
                target_node,
                source,
                current_container=current_container,
                intrinsic_name=intrinsic_name,
            ):
                if target in seen:
                    continue
                seen.add(target)
                targets.append(target)
        return targets

    def _binding_target_nodes_for_value(self, value_node: Node) -> list[Node]:
        """Return outer-to-inner binding targets for a value inside assignment chains."""
        targets: list[Node] = []
        current = value_node
        while True:
            parent = current.parent
            if parent is None:
                break
            if (
                parent.type == "assignment_expression"
                and parent.child_by_field_name("right") == current
            ):
                target = parent.child_by_field_name("left")
                if target is not None:
                    targets.append(target)
                current = parent
                continue
            if (
                parent.type == "variable_declarator"
                and parent.child_by_field_name("value") == current
            ):
                target = parent.child_by_field_name("name")
                if target is not None:
                    targets.append(target)
                current = parent
                continue
            break
        targets.reverse()
        return targets

    def _assignment_symbol_target(
        self,
        node: Node | None,
        source: str,
        *,
        current_container: _ContainerContext | None,
        intrinsic_name: str | None = None,
    ) -> _AssignmentTarget | None:
        """Resolve an identifier/member binding target into synthetic symbol metadata."""
        if node is None:
            return None

        if node.type == "identifier":
            name = source[node.start_byte : node.end_byte]
            if not name:
                return None
            return _AssignmentTarget(
                name=name,
                container=current_container,
                binding_style="variable_assignment",
            )

        path = self._member_expression_path(node, source)
        if not path:
            return None

        if path in {"exports", "module.exports"}:
            return _AssignmentTarget(
                name=intrinsic_name or path,
                container=None,
                binding_style="export_root_assignment",
                fqname=path,
            )

        parts = path.split(".")
        if len(parts) < 2:
            return None
        owner_path = ".".join(parts[:-1])
        binding_style = self._binding_style_for_owner_path(owner_path)
        return _AssignmentTarget(
            name=parts[-1],
            container=_ContainerContext(symbol_id=None, fqname=owner_path),
            binding_style=binding_style,
        )

    def _assignment_symbol_targets_for_node(
        self,
        node: Node | None,
        source: str,
        *,
        current_container: _ContainerContext | None,
        intrinsic_name: str | None = None,
    ) -> list[_AssignmentTarget]:
        """Resolve one target node into all alias-expanded synthetic targets."""
        target = self._assignment_symbol_target(
            node,
            source,
            current_container=current_container,
            intrinsic_name=intrinsic_name,
        )
        if target is None:
            return []
        if node is None or node.type == "identifier" or target.container is None:
            return [target]

        path = self._member_expression_path(node, source)
        if not path:
            return [target]
        parts = path.split(".")
        if len(parts) < 2:
            return [target]

        owner_path = ".".join(parts[:-1])
        property_name = parts[-1]
        alias_paths = self._owner_alias_paths(owner_path)
        if len(alias_paths) <= 1:
            return [target]

        return [
            _AssignmentTarget(
                name=property_name,
                container=_ContainerContext(symbol_id=None, fqname=alias_owner),
                binding_style=self._binding_style_for_owner_path(alias_owner),
            )
            for alias_owner in alias_paths
        ]

    def _bound_object_owners(self, node: Node | None, source: str) -> list[_ContainerContext]:
        """Resolve all recoverable owners for an object literal used as a named target."""
        if node is None or node.type != "object":
            return []
        owners: list[_ContainerContext] = []
        seen: set[_ContainerContext] = set()
        for target_node in self._binding_target_nodes_for_value(node):
            owner = self._binding_target_container(target_node, source)
            if owner is None or owner in seen:
                continue
            seen.add(owner)
            owners.append(owner)
        return owners

    def _collect_js_object_owner_aliases(
        self,
        node: Node,
        source: str,
    ) -> dict[str, tuple[str, ...]]:
        """Collect recoverable alias groups created by object-literal root bindings."""
        aliases: dict[str, tuple[str, ...]] = {}

        def walk(current: Node) -> None:
            if not current.is_named:
                return
            if current.type == "object":
                owner_paths = list(
                    dict.fromkeys(
                        owner.fqname for owner in self._bound_object_owners(current, source)
                    )
                )
                if len(owner_paths) > 1:
                    merged = tuple(owner_paths)
                    for owner_path in owner_paths:
                        existing = aliases.get(owner_path)
                        if existing is not None:
                            merged = tuple(dict.fromkeys([*existing, *merged]))
                    for owner_path in merged:
                        aliases[owner_path] = merged
            for child in current.children:
                walk(child)

        walk(node)
        return aliases

    def _binding_target_container(self, node: Node | None, source: str) -> _ContainerContext | None:
        """Resolve the full target path used as an object-binding owner context."""
        if node is None:
            return None
        if node.type == "identifier":
            name = source[node.start_byte : node.end_byte]
            if not name:
                return None
            return _ContainerContext(symbol_id=None, fqname=name)
        path = self._member_expression_path(node, source)
        if not path:
            return None
        return _ContainerContext(symbol_id=None, fqname=path)

    def _synthetic_assignment_kind(self, *, node: Node, binding_style: str) -> str:
        """Return the stable symbol kind for an assignment-based JS-family symbol."""
        if node.type == "class_expression":
            return "class"
        if binding_style in {
            "variable_assignment",
            "export_assignment",
            "export_root_assignment",
        }:
            return "function"
        return "method"

    def _member_expression_path(self, node: Node, source: str) -> str | None:
        """Return a dotted path for supported identifier/member/subscript chains."""
        if node.type == "identifier":
            return source[node.start_byte : node.end_byte]

        if node.type == "subscript_expression":
            object_node = node.child_by_field_name("object")
            index_node = node.child_by_field_name("index")
            if object_node is None or index_node is None:
                return None
            object_path = self._member_expression_path(object_node, source)
            property_name = self._identifier_string_property_name(index_node, source)
            if not object_path or not property_name:
                return None
            return f"{object_path}.{property_name}"

        if node.type != "member_expression":
            return None

        object_node = node.child_by_field_name("object")
        property_node = node.child_by_field_name("property")
        if object_node is None or property_node is None:
            return None
        object_path = self._member_expression_path(object_node, source)
        property_name = self._property_name(property_node, source)
        if not object_path or not property_name:
            return None
        return f"{object_path}.{property_name}"

    def _define_property_symbol_specs(
        self,
        *,
        pair: Node,
        value_node: Node,
        source: str,
    ) -> list[_SyntheticSymbolSpec]:
        """Resolve exact Object.defineProperty descriptor symbols from a pair value node."""
        key_name = self._property_name(pair.child_by_field_name("key"), source)
        if key_name not in {"value", "get", "set"}:
            return []

        descriptor = pair.parent
        if descriptor is None or descriptor.type != "object":
            return []
        arguments = descriptor.parent
        if arguments is None or arguments.type != "arguments":
            return []
        call = arguments.parent
        if call is None or call.type != "call_expression":
            return []

        callee = call.child_by_field_name("function")
        if (
            callee is None
            or self._member_expression_path(callee, source) != "Object.defineProperty"
        ):
            return []

        args = [child for child in arguments.named_children]
        if len(args) < 3 or args[2] != descriptor:
            return []

        owner_path = self._member_expression_path(args[0], source)
        property_name = self._identifier_string_property_name(args[1], source)
        if not owner_path or not property_name:
            return []

        attributes: dict[str, object] = {"binding_style": "define_property_descriptor"}
        if key_name in {"get", "set"}:
            if value_node.type == "class_expression":
                return []
            attributes["accessor"] = key_name

        return [
            _SyntheticSymbolSpec(
                kind=self._synthetic_assignment_kind(
                    node=value_node,
                    binding_style=self._binding_style_for_owner_path(owner_path),
                ),
                name=property_name,
                container=_ContainerContext(symbol_id=None, fqname=owner_path),
                attributes=attributes,
            )
        ]

    def _method_definition_accessor(self, node: Node, source: str) -> str | None:
        """Return get/set for object-literal accessors when syntactically explicit."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        prefix = source[node.start_byte : name_node.start_byte].strip()
        if prefix == "get":
            return "get"
        if prefix == "set":
            return "set"
        return None

    @staticmethod
    def _binding_style_for_owner_path(owner_path: str) -> str:
        """Resolve the stable binding style for a member-like owner path."""
        if owner_path in {"exports", "module.exports"}:
            return "export_assignment"
        if owner_path.endswith(".prototype"):
            return "prototype_assignment"
        return "member_assignment"

    def _owner_alias_paths(self, owner_path: str) -> tuple[str, ...]:
        """Return alias-equivalent owner paths discovered from object-literal roots."""
        aliases = self._js_object_owner_aliases.get(owner_path)
        if aliases:
            return aliases
        return (owner_path,)

    def _identifier_string_property_name(self, node: Node | None, source: str) -> str | None:
        """Return identifier-shaped property name from a quoted string literal."""
        if node is None or node.type != "string":
            return None
        value = self._strip_quotes(source[node.start_byte : node.end_byte])
        if not value or _IDENTIFIER_PROPERTY_RE.fullmatch(value) is None:
            return None
        return value

    @staticmethod
    def _property_name(node: Node | None, source: str) -> str | None:
        """Extract a recoverable property/binding name from an identifier-like node."""
        if node is None:
            return None
        if node.type not in {
            "identifier",
            "property_identifier",
            "type_identifier",
            "field_identifier",
            "name",
        }:
            return None
        return source[node.start_byte : node.end_byte] or None

    def _cpp_namespace_container(
        self,
        *,
        node: Node,
        source: str,
        current_container: _ContainerContext | None,
    ) -> _ContainerContext | None:
        """Return namespace container metadata for C++ namespace body traversal."""
        if node.type != "namespace_definition":
            return None
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        namespace_parts = self._split_cpp_qualified_text(
            source[name_node.start_byte : name_node.end_byte],
        )
        if not namespace_parts:
            return None
        namespace_fqname = ".".join(namespace_parts)
        if current_container is not None:
            namespace_fqname = f"{current_container.fqname}.{namespace_fqname}"
        return _ContainerContext(symbol_id=None, fqname=namespace_fqname)

    @staticmethod
    def _split_cpp_qualified_text(text: str) -> tuple[str, ...]:
        """Split `A::B::name` text into stable non-empty parts."""
        parts = [part.strip() for part in text.split("::") if part.strip()]
        return tuple(parts)

    def _collect_cpp_macro_patterns(self, node: Node, source: str) -> dict[str, _CppMacroPattern]:
        """Collect same-file macro definitions that expand to recoverable symbols."""
        patterns: dict[str, _CppMacroPattern] = {}

        def walk(current: Node) -> None:
            if not current.is_named:
                return
            if current.type == "preproc_function_def":
                pattern = self._cpp_macro_pattern_from_definition(current, source)
                if pattern is not None:
                    patterns[pattern.macro_name] = pattern
            for child in current.children:
                walk(child)

        walk(node)
        return patterns

    def _cpp_macro_pattern_from_definition(
        self,
        node: Node,
        source: str,
    ) -> _CppMacroPattern | None:
        """Build one recoverable macro pattern from a macro function definition."""
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        value_node = node.child_by_field_name("value")
        if name_node is None or params_node is None or value_node is None:
            return None
        macro_name = source[name_node.start_byte : name_node.end_byte]
        if not macro_name:
            return None

        parameter_names = [
            source[param.start_byte : param.end_byte]
            for param in params_node.named_children
            if param.type == "identifier"
        ]
        if not parameter_names:
            return None
        parameter_indexes = {name: index for index, name in enumerate(parameter_names)}
        value_text = source[value_node.start_byte : value_node.end_byte]
        header_text = value_text.split("{", 1)[0]

        for owner_name, owner_index in parameter_indexes.items():
            for name_name, name_index in parameter_indexes.items():
                if owner_name == name_name:
                    continue
                method_pattern = re.compile(
                    rf"\b{re.escape(owner_name)}\s*::\s*{re.escape(name_name)}\s*\(",
                )
                if method_pattern.search(header_text):
                    return _CppMacroPattern(
                        macro_name=macro_name,
                        kind="method",
                        name_index=name_index,
                        owner_index=owner_index,
                        parameter_count=len(parameter_names),
                    )

        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", header_text):
            candidate = match.group(1)
            if candidate not in parameter_indexes:
                continue
            prefix = header_text[max(0, match.start(1) - 2) : match.start(1)]
            if prefix == "::":
                continue
            return _CppMacroPattern(
                macro_name=macro_name,
                kind="function",
                name_index=parameter_indexes[candidate],
                owner_index=None,
                parameter_count=len(parameter_names),
            )
        return None

    def _cpp_macro_symbol_specs(
        self,
        *,
        node: Node,
        source: str,
        current_container: _ContainerContext | None,
    ) -> list[_SyntheticSymbolSpec]:
        """Emit synthetic C++ symbols recovered from strict macro placeholder patterns."""
        if node.type != "call_expression":
            return []
        callee = node.child_by_field_name("function")
        if callee is None or callee.type != "identifier":
            return []
        macro_name = source[callee.start_byte : callee.end_byte]
        if not macro_name:
            return []
        pattern = self._cpp_macro_patterns.get(macro_name)
        if pattern is None:
            return []

        arguments_node = node.child_by_field_name("arguments")
        if arguments_node is None:
            return []
        arguments = [child for child in arguments_node.named_children]
        if len(arguments) != pattern.parameter_count:
            return []
        symbol_name = self._cpp_macro_argument_name(arguments[pattern.name_index], source)
        if symbol_name is None:
            return []

        container = current_container
        if pattern.kind == "method":
            owner_index = pattern.owner_index
            if owner_index is None:
                return []
            owner_name = self._cpp_macro_argument_name(arguments[owner_index], source)
            if owner_name is None:
                return []
            owner_fqname = owner_name
            if current_container is not None:
                owner_fqname = f"{current_container.fqname}.{owner_name}"
            container = _ContainerContext(symbol_id=None, fqname=owner_fqname)

        return [
            _SyntheticSymbolSpec(
                kind=pattern.kind,
                name=symbol_name,
                container=container,
                attributes={
                    "binding_style": "macro_generated",
                    "macro_generated": True,
                    "macro_name": pattern.macro_name,
                },
            )
        ]

    @staticmethod
    def _cpp_macro_argument_name(node: Node, source: str) -> str | None:
        """Return identifier-like macro argument text for strict recoverable patterns."""
        if node.type not in {
            "identifier",
            "type_identifier",
            "field_identifier",
            "namespace_identifier",
        }:
            return None
        value = source[node.start_byte : node.end_byte]
        if not value or _IDENTIFIER_PROPERTY_RE.fullmatch(value) is None:
            return None
        return value

    @staticmethod
    def _drop_cpp_macro_duplicates(symbols: list[Symbol]) -> list[Symbol]:
        """Drop macro-generated symbols when explicit symbols already exist for same fqname/kind."""
        explicit_keys = {
            (symbol.kind, symbol.fqname or symbol.name)
            for symbol in symbols
            if not bool(symbol.attributes.get("macro_generated"))
        }
        deduped: list[Symbol] = []
        for symbol in symbols:
            is_macro = bool(symbol.attributes.get("macro_generated"))
            symbol_key = (symbol.kind, symbol.fqname or symbol.name)
            if is_macro and symbol_key in explicit_keys:
                continue
            deduped.append(symbol)
        return deduped

    def _symbol_kind(self, node: Node, source: str) -> str | None:
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
                if node.parent is None or node.parent.type != "class_body":
                    return None
                return "method"
            if node.type in self.spec.function_nodes:
                return "function"
            if node.type in self.spec.class_nodes:
                return "class"
            if node.type in self.spec.interface_nodes:
                return "interface"
            return None

        if self.language == "c":
            if node.type in {"struct_specifier", "union_specifier"}:
                return "type"
            if node.type == "enum_specifier":
                return "enum"
            if node.type == "function_definition":
                return "function"
            if node.type == "declaration" and self._c_family_is_callable_declaration(node, source):
                return "function"
            return None

        if self.language == "cpp":
            if node.type in {"class_specifier", "struct_specifier"}:
                return "class"
            if node.type == "enum_specifier":
                return "enum"
            if node.type == "function_definition":
                if self._cpp_is_class_body_method_definition(node):
                    return "method"
                if self._cpp_has_qualified_method_declarator(node, source):
                    return "method"
                return "function"
            if node.type == "field_declaration" and self._c_family_is_callable_declaration(
                node,
                source,
            ):
                return "method"
            if node.type == "declaration" and self._c_family_is_callable_declaration(node, source):
                return "function"
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

    def _c_family_is_callable_declaration(self, node: Node, source: str) -> bool:
        """Return True when declaration-like node declares a callable symbol."""
        info = self._c_family_declarator_info(node.child_by_field_name("declarator"), source)
        return info.is_callable_declaration

    def _c_family_declarator_info(
        self,
        declarator: Node | None,
        source: str,
    ) -> _CFamilyDeclaratorInfo:
        """Resolve C-family declarator metadata from nested declarator chains."""
        path = self._c_family_declarator_path(declarator)
        if not path:
            return _CFamilyDeclaratorInfo(
                symbol_name=None,
                qualified_parts=(),
                is_callable_declaration=False,
            )

        terminal = path[-1]
        symbol_name: str | None = None
        qualified_parts: tuple[str, ...] = ()
        if terminal.type == "qualified_identifier":
            qualified_parts = self._c_family_qualified_parts(terminal, source)
            if qualified_parts:
                symbol_name = qualified_parts[-1]
        elif terminal.type in {
            "identifier",
            "field_identifier",
            "type_identifier",
            "namespace_identifier",
            "operator_name",
            "destructor_name",
        }:
            symbol_name = source[terminal.start_byte : terminal.end_byte] or None

        is_callable = False
        function_indexes = [
            index
            for index, current in enumerate(path[:-1])
            if current.type == "function_declarator"
        ]
        if function_indexes:
            innermost_index = function_indexes[-1]
            path_tail = path[innermost_index + 1 : -1]
            is_callable = not any(
                current.type in {"pointer_declarator", "reference_declarator", "array_declarator"}
                for current in path_tail
            )

        return _CFamilyDeclaratorInfo(
            symbol_name=symbol_name,
            qualified_parts=qualified_parts,
            is_callable_declaration=is_callable and symbol_name is not None,
        )

    def _c_family_declarator_path(self, declarator: Node | None) -> list[Node]:
        """Return terminal declarator walk used for C-family name/callability analysis."""
        if declarator is None:
            return []
        path: list[Node] = []
        current: Node | None = declarator
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            path.append(current)
            visited.add(id(current))
            if current.type in {
                "identifier",
                "field_identifier",
                "type_identifier",
                "qualified_identifier",
                "namespace_identifier",
                "operator_name",
                "destructor_name",
            }:
                break
            nested = current.child_by_field_name("declarator")
            if nested is None and current.type == "parenthesized_declarator":
                for child in current.named_children:
                    nested = child
                    break
            current = nested
        return path

    def _c_family_qualified_parts(self, node: Node, source: str) -> tuple[str, ...]:
        """Resolve nested qualified identifier/name nodes into ordered parts."""
        if node.type == "qualified_identifier":
            scope = node.child_by_field_name("scope")
            name = node.child_by_field_name("name")
            parts: list[str] = []
            if scope is not None:
                parts.extend(self._c_family_qualified_parts(scope, source))
            if name is not None:
                parts.extend(self._c_family_qualified_parts(name, source))
            if parts:
                return tuple(parts)
            text = source[node.start_byte : node.end_byte]
            return self._split_cpp_qualified_text(text)
        text = source[node.start_byte : node.end_byte]
        if not text:
            return ()
        return (text,)

    @staticmethod
    def _cpp_is_class_body_method_definition(node: Node) -> bool:
        """Return True for C++ method definitions defined inside class/struct bodies."""
        parent = node.parent
        if parent is None or parent.type != "field_declaration_list":
            return False
        owner = parent.parent
        return bool(owner and owner.type in {"class_specifier", "struct_specifier"})

    def _cpp_has_qualified_method_declarator(self, node: Node, source: str) -> bool:
        """Return True when C++ definition declarator is type-qualified (`Type::name`)."""
        info = self._c_family_declarator_info(node.child_by_field_name("declarator"), source)
        return info.is_callable_declaration and len(info.qualified_parts) >= 2

    def _cpp_method_container_from_qualified_declarator(
        self,
        node: Node,
        source: str,
    ) -> _ContainerContext | None:
        """Build method container context from `Type::method` declarators."""
        info = self._c_family_declarator_info(node.child_by_field_name("declarator"), source)
        if len(info.qualified_parts) < 2:
            return None
        owner_parts = info.qualified_parts[:-1]
        return _ContainerContext(symbol_id=None, fqname=".".join(owner_parts))

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

        if self.language in {"c", "cpp"}:
            if node.type in {
                "function_definition",
                "declaration",
                "field_declaration",
            }:
                info = self._c_family_declarator_info(
                    node.child_by_field_name("declarator"),
                    source,
                )
                name = info.symbol_name
                if name:
                    return name
            if node.type in {
                "class_specifier",
                "struct_specifier",
                "union_specifier",
                "enum_specifier",
            }:
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    return source[name_node.start_byte : name_node.end_byte] or None

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
