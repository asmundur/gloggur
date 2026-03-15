from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from gloggur.models import EdgeRecord, Symbol

_REFERENCE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PYTHON_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)$")
_PYTHON_IMPORT_RE = re.compile(r"^\s*import\s+(.+)$")
_JS_IMPORT_FROM_RE = re.compile(r"^\s*import\s+.+\s+from\s+['\"]([^'\"]+)['\"]")
_JS_IMPORT_SIDE_EFFECT_RE = re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]")
_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_\.\\*]+)\s*;")
_RUST_USE_RE = re.compile(r"^\s*use\s+([^;]+);")
_GO_SINGLE_IMPORT_RE = re.compile(r'^\s*import\s+"([^"]+)"')
_GO_BLOCK_IMPORT_OPEN_RE = re.compile(r"^\s*import\s*\(\s*$")
_GO_BLOCK_IMPORT_ENTRY_RE = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)?\"([^\"]+)\"")
_REFERENCE_KEYWORDS = {
    "and",
    "as",
    "assert",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "def",
    "default",
    "elif",
    "else",
    "enum",
    "except",
    "export",
    "false",
    "finally",
    "fn",
    "for",
    "from",
    "function",
    "if",
    "impl",
    "import",
    "in",
    "interface",
    "let",
    "match",
    "new",
    "null",
    "package",
    "private",
    "protected",
    "pub",
    "public",
    "return",
    "self",
    "static",
    "struct",
    "super",
    "switch",
    "this",
    "throw",
    "trait",
    "true",
    "try",
    "type",
    "undefined",
    "use",
    "var",
    "while",
}


@dataclass(frozen=True)
class _ResolvedTarget:
    """Resolved graph target with deterministic identity/kind/confidence."""

    target_id: str
    target_kind: str
    confidence: float


@dataclass(frozen=True)
class CandidateSymbolIndex:
    """Reusable candidate-symbol lookup tables for deterministic edge extraction."""

    symbols: tuple[Symbol, ...]
    by_id: dict[str, Symbol]
    by_fqname: dict[str, Symbol]
    by_name: dict[str, tuple[Symbol, ...]]

    @staticmethod
    def _symbol_sort_key(symbol: Symbol) -> tuple[str, int, int, str]:
        return (symbol.file_path, symbol.start_line, symbol.end_line, symbol.id)

    @staticmethod
    def _name_sort_key(symbol: Symbol) -> tuple[str, int, str]:
        return (symbol.file_path, symbol.start_line, symbol.id)

    @classmethod
    def from_symbols(cls, symbols: Sequence[Symbol]) -> CandidateSymbolIndex:
        """Build one reusable candidate-symbol index with stable merge precedence by id."""
        merged: dict[str, Symbol] = {}
        for symbol in symbols:
            merged[symbol.id] = symbol
        ordered = sorted(merged.values(), key=cls._symbol_sort_key)
        by_id = {symbol.id: symbol for symbol in ordered}
        by_fqname: dict[str, Symbol] = {}
        by_name_lists: dict[str, list[Symbol]] = {}
        for symbol in ordered:
            if symbol.fqname:
                by_fqname[symbol.fqname] = symbol
            by_name_lists.setdefault(symbol.name, []).append(symbol)
        by_name = {
            name: tuple(sorted(items, key=cls._name_sort_key))
            for name, items in by_name_lists.items()
        }
        return cls(
            symbols=tuple(ordered),
            by_id=by_id,
            by_fqname=by_fqname,
            by_name=by_name,
        )

    def with_local_symbols(self, local_symbols: Sequence[Symbol]) -> CandidateSymbolIndex:
        """Overlay local symbols only when the prebuilt candidate index does not already contain them."""
        if not local_symbols:
            return self
        if all(self.by_id.get(symbol.id) is symbol for symbol in local_symbols):
            return self
        return CandidateSymbolIndex.from_symbols([*self.symbols, *local_symbols])


class GraphEdgeExtractor:
    """Extract deterministic edge records from symbols plus source text."""

    def __init__(self, language: str) -> None:
        self.language = language

    def extract_edges(
        self,
        *,
        path: str,
        source: str,
        symbols: list[Symbol],
        candidate_symbols: list[Symbol] | None = None,
        candidate_symbol_index: CandidateSymbolIndex | None = None,
        repo_id: str,
        commit: str,
        include_text: bool = True,
    ) -> list[EdgeRecord]:
        """Extract DEFINES/CONTAINS/IMPORTS/CALLS/REFERENCES/TESTS edges for one file."""
        if candidate_symbol_index is None:
            candidate_symbol_index = CandidateSymbolIndex.from_symbols(candidate_symbols or [])
        combined_index = candidate_symbol_index.with_local_symbols(symbols)
        combined = combined_index.symbols
        by_id = combined_index.by_id
        by_fqname = combined_index.by_fqname
        by_name = combined_index.by_name

        file_node_id = self._file_node_id(repo_id=repo_id, path=path)
        edges: list[EdgeRecord] = []
        seen: set[tuple[str, str, str, str, int]] = set()

        def add_edge(
            *,
            edge_type: str,
            from_id: str,
            to_id: str,
            from_kind: str,
            to_kind: str,
            line: int,
            confidence: float,
            text: str | None,
        ) -> None:
            key = (edge_type, from_id, to_id, path, line)
            if key in seen:
                return
            seen.add(key)
            edge_id = self._edge_id(
                repo_id=repo_id,
                edge_type=edge_type,
                from_id=from_id,
                to_id=to_id,
                file_path=path,
                line=line,
            )
            edges.append(
                EdgeRecord(
                    edge_id=edge_id,
                    edge_type=edge_type,
                    from_id=from_id,
                    to_id=to_id,
                    from_kind=from_kind,
                    to_kind=to_kind,
                    file_path=path,
                    line=line,
                    confidence=confidence,
                    repo_id=repo_id,
                    commit=commit,
                    text=text,
                )
            )

        lines = source.splitlines()
        for symbol in symbols:
            add_edge(
                edge_type="DEFINES",
                from_id=file_node_id,
                to_id=symbol.id,
                from_kind="file",
                to_kind=symbol.kind,
                line=symbol.start_line,
                confidence=1.0,
                text=(
                    self._edge_text(
                        edge_type="DEFINES",
                        from_id=file_node_id,
                        to_id=symbol.id,
                        file_path=path,
                        line=symbol.start_line,
                        confidence=1.0,
                    )
                    if include_text
                    else None
                ),
            )
            if symbol.container_id:
                add_edge(
                    edge_type="CONTAINS",
                    from_id=symbol.container_id,
                    to_id=symbol.id,
                    from_kind="container",
                    to_kind=symbol.kind,
                    line=symbol.start_line,
                    confidence=1.0,
                    text=(
                        self._edge_text(
                            edge_type="CONTAINS",
                            from_id=symbol.container_id,
                            to_id=symbol.id,
                            file_path=path,
                            line=symbol.start_line,
                            confidence=1.0,
                        )
                        if include_text
                        else None
                    ),
                )

            for call_target in symbol.calls:
                resolved = self._resolve_target(
                    call_target,
                    current_file=path,
                    by_fqname=by_fqname,
                    by_name=by_name,
                    container_fqname=symbol.container_fqname,
                )
                add_edge(
                    edge_type="CALLS",
                    from_id=symbol.id,
                    to_id=resolved.target_id,
                    from_kind=symbol.kind,
                    to_kind=resolved.target_kind,
                    line=symbol.start_line,
                    confidence=resolved.confidence,
                    text=(
                        self._edge_text(
                            edge_type="CALLS",
                            from_id=symbol.id,
                            to_id=resolved.target_id,
                            file_path=path,
                            line=symbol.start_line,
                            confidence=resolved.confidence,
                        )
                        if include_text
                        else None
                    ),
                )

            for reference in self._symbol_reference_tokens(symbol=symbol, source_lines=lines):
                resolved_reference = self._resolve_target(
                    reference,
                    current_file=path,
                    by_fqname=by_fqname,
                    by_name=by_name,
                    container_fqname=symbol.container_fqname,
                )
                if resolved_reference.target_id == symbol.id:
                    continue
                reference_confidence = min(resolved_reference.confidence, 0.45)
                add_edge(
                    edge_type="REFERENCES",
                    from_id=symbol.id,
                    to_id=resolved_reference.target_id,
                    from_kind=symbol.kind,
                    to_kind=resolved_reference.target_kind,
                    line=symbol.start_line,
                    confidence=reference_confidence,
                    text=(
                        self._edge_text(
                            edge_type="REFERENCES",
                            from_id=symbol.id,
                            to_id=resolved_reference.target_id,
                            file_path=path,
                            line=symbol.start_line,
                            confidence=reference_confidence,
                        )
                        if include_text
                        else None
                    ),
                )

            for test_ref in sorted(set(symbol.covered_by)):
                resolved_test = self._resolve_test_reference(
                    test_ref,
                    by_id=by_id,
                    by_fqname=by_fqname,
                    by_name=by_name,
                    current_file=path,
                )
                add_edge(
                    edge_type="TESTS",
                    from_id=resolved_test.target_id,
                    to_id=symbol.id,
                    from_kind=resolved_test.target_kind,
                    to_kind=symbol.kind,
                    line=symbol.start_line,
                    confidence=resolved_test.confidence,
                    text=(
                        self._edge_text(
                            edge_type="TESTS",
                            from_id=resolved_test.target_id,
                            to_id=symbol.id,
                            file_path=path,
                            line=symbol.start_line,
                            confidence=resolved_test.confidence,
                        )
                        if include_text
                        else None
                    ),
                )

        for import_line, import_target in self._extract_import_targets(source):
            resolved_import = self._resolve_target(
                import_target,
                current_file=path,
                by_fqname=by_fqname,
                by_name=by_name,
                container_fqname=None,
                unresolved_confidence=0.2,
            )
            add_edge(
                edge_type="IMPORTS",
                from_id=file_node_id,
                to_id=resolved_import.target_id,
                from_kind="file",
                to_kind=resolved_import.target_kind,
                line=import_line,
                confidence=max(0.2, resolved_import.confidence),
                text=(
                    self._edge_text(
                        edge_type="IMPORTS",
                        from_id=file_node_id,
                        to_id=resolved_import.target_id,
                        file_path=path,
                        line=import_line,
                        confidence=max(0.2, resolved_import.confidence),
                    )
                    if include_text
                    else None
                ),
            )

        edges.sort(key=lambda item: (item.file_path, item.line, item.edge_type, item.edge_id))
        return edges

    @staticmethod
    def _file_node_id(*, repo_id: str, path: str) -> str:
        """Return deterministic file-node id used for DEFINES/IMPORTS edges."""
        payload = f"{repo_id}|{os.path.normpath(path)}"
        return f"file:{hashlib.sha256(payload.encode('utf8')).hexdigest()}"

    @staticmethod
    def _edge_id(
        *,
        repo_id: str,
        edge_type: str,
        from_id: str,
        to_id: str,
        file_path: str,
        line: int,
    ) -> str:
        """Build deterministic edge id from endpoint and location metadata."""
        payload = "|".join(
            [repo_id, edge_type, from_id, to_id, os.path.normpath(file_path), str(line)]
        )
        return hashlib.sha256(payload.encode("utf8")).hexdigest()

    @staticmethod
    def _edge_text(
        *,
        edge_type: str,
        from_id: str,
        to_id: str,
        file_path: str,
        line: int,
        confidence: float,
    ) -> str:
        """Build deterministic edge fact text used for semantic edge search."""
        return (
            f"EDGE_TYPE: {edge_type}\n"
            f"FROM: {from_id}\n"
            f"TO: {to_id}\n"
            f"FILE: {file_path}\n"
            f"LINE: {line}\n"
            f"CONFIDENCE: {confidence:.3f}"
        )

    def _extract_import_targets(self, source: str) -> list[tuple[int, str]]:
        """Extract import targets with source line anchors per language."""
        results: list[tuple[int, str]] = []
        go_block = False
        for index, raw_line in enumerate(source.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if self.language == "python":
                match = _PYTHON_FROM_IMPORT_RE.match(line)
                if match:
                    module = match.group(1).strip()
                    imports = match.group(2).strip()
                    for name in imports.replace("(", "").replace(")", "").split(","):
                        clean = name.strip().split(" as ", 1)[0].strip()
                        if clean:
                            results.append((index, f"{module}.{clean}"))
                    continue
                match = _PYTHON_IMPORT_RE.match(line)
                if match:
                    for name in match.group(1).split(","):
                        clean = name.strip().split(" as ", 1)[0].strip()
                        if clean:
                            results.append((index, clean))
                    continue
            elif self.language in {"javascript", "typescript", "tsx"}:
                match = _JS_IMPORT_FROM_RE.match(line)
                if match:
                    results.append((index, match.group(1).strip()))
                    continue
                match = _JS_IMPORT_SIDE_EFFECT_RE.match(line)
                if match:
                    results.append((index, match.group(1).strip()))
                    continue
            elif self.language == "rust":
                match = _RUST_USE_RE.match(line)
                if match:
                    results.append((index, match.group(1).strip()))
                    continue
            elif self.language == "java":
                match = _JAVA_IMPORT_RE.match(line)
                if match:
                    results.append((index, match.group(1).strip()))
                    continue
            elif self.language == "go":
                if go_block:
                    if line.startswith(")"):
                        go_block = False
                        continue
                    block_match = _GO_BLOCK_IMPORT_ENTRY_RE.match(line)
                    if block_match:
                        results.append((index, block_match.group(1).strip()))
                    continue
                if _GO_BLOCK_IMPORT_OPEN_RE.match(line):
                    go_block = True
                    continue
                match = _GO_SINGLE_IMPORT_RE.match(line)
                if match:
                    results.append((index, match.group(1).strip()))
                    continue
        results.sort(key=lambda item: (item[0], item[1]))
        return results

    def _symbol_reference_tokens(self, *, symbol: Symbol, source_lines: list[str]) -> list[str]:
        """Extract best-effort identifier references from a symbol body slice."""
        start = max(0, symbol.start_line - 1)
        end = max(start, symbol.end_line)
        body = "\n".join(source_lines[start:end])
        tokens = sorted(set(_REFERENCE_TOKEN_RE.findall(body)))
        references: list[str] = []
        for token in tokens:
            lowered = token.lower()
            if lowered in _REFERENCE_KEYWORDS:
                continue
            if token == symbol.name:
                continue
            if len(token) < 3:
                continue
            references.append(token)
            if len(references) >= 25:
                break
        return references

    @staticmethod
    def _resolve_target(
        target: str,
        *,
        current_file: str,
        by_fqname: Mapping[str, Symbol],
        by_name: Mapping[str, Sequence[Symbol]],
        container_fqname: str | None,
        unresolved_confidence: float = 0.15,
    ) -> _ResolvedTarget:
        """Resolve a call/import/reference target to symbol id when possible."""
        normalized = target.strip()
        if not normalized:
            return _ResolvedTarget(
                target_id="unresolved:<empty>",
                target_kind="unresolved",
                confidence=unresolved_confidence,
            )

        fqname_symbol = by_fqname.get(normalized)
        if fqname_symbol is not None:
            return _ResolvedTarget(
                target_id=fqname_symbol.id,
                target_kind=fqname_symbol.kind,
                confidence=0.95,
            )

        if container_fqname:
            container_candidate = f"{container_fqname}.{normalized}"
            container_symbol = by_fqname.get(container_candidate)
            if container_symbol is not None:
                return _ResolvedTarget(
                    target_id=container_symbol.id,
                    target_kind=container_symbol.kind,
                    confidence=0.9,
                )

        name = normalized.split(".")[-1]
        candidates = by_name.get(name, [])
        if len(candidates) == 1:
            only = candidates[0]
            confidence = 0.85 if only.file_path == current_file else 0.75
            return _ResolvedTarget(
                target_id=only.id,
                target_kind=only.kind,
                confidence=confidence,
            )
        if len(candidates) > 1:
            same_file = [
                candidate for candidate in candidates if candidate.file_path == current_file
            ]
            picked = same_file[0] if same_file else candidates[0]
            confidence = 0.65 if same_file else 0.45
            return _ResolvedTarget(
                target_id=picked.id,
                target_kind=picked.kind,
                confidence=confidence,
            )

        return _ResolvedTarget(
            target_id=f"unresolved:{normalized}",
            target_kind="unresolved",
            confidence=unresolved_confidence,
        )

    @staticmethod
    def _resolve_test_reference(
        reference: str,
        *,
        by_id: Mapping[str, Symbol],
        by_fqname: Mapping[str, Symbol],
        by_name: Mapping[str, Sequence[Symbol]],
        current_file: str,
    ) -> _ResolvedTarget:
        """Resolve covered_by values into test-symbol endpoints when available."""
        normalized = reference.strip()
        if not normalized:
            return _ResolvedTarget("unresolved:<empty>", "test", 0.2)
        by_id_symbol = by_id.get(normalized)
        if by_id_symbol is not None:
            return _ResolvedTarget(by_id_symbol.id, by_id_symbol.kind, 0.95)
        by_fqname_symbol = by_fqname.get(normalized)
        if by_fqname_symbol is not None:
            return _ResolvedTarget(by_fqname_symbol.id, by_fqname_symbol.kind, 0.9)
        name = normalized.split(":")[-1].split(".")[-1]
        resolved = GraphEdgeExtractor._resolve_target(
            name,
            current_file=current_file,
            by_fqname=by_fqname,
            by_name=by_name,
            container_fqname=None,
            unresolved_confidence=0.25,
        )
        if resolved.target_kind == "unresolved":
            return _ResolvedTarget(f"unresolved:{normalized}", "test", 0.25)
        return _ResolvedTarget(
            resolved.target_id,
            resolved.target_kind,
            max(resolved.confidence, 0.6),
        )
