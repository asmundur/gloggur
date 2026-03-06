from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from gloggur.byte_spans import LineByteSpanIndex
from gloggur.config import GloggurConfig
from gloggur.indexer.shared import ParsedFileSnapshot
from gloggur.models import Symbol
from gloggur.parsers.registry import ParserRegistry
from gloggur.symbol_index.models import IndexedFile, SymbolIndexResult, SymbolOccurrence
from gloggur.symbol_index.store import SymbolIndexStore, SymbolIndexStoreConfig

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
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


class SymbolIndexer:
    """Incremental builder for .gloggur/index/symbols.db."""

    def __init__(
        self,
        *,
        repo_root: Path,
        config: GloggurConfig,
        parser_registry: ParserRegistry | None = None,
        store: SymbolIndexStore | None = None,
    ) -> None:
        self.repo_root = Path(os.path.abspath(str(repo_root)))
        self.config = config
        self.parser_registry = parser_registry or ParserRegistry(
            extension_map=config.parser_extension_map,
            adapter_overrides=config.adapters if isinstance(config.adapters, dict) else None,
        )
        self.store = store or SymbolIndexStore(
            SymbolIndexStoreConfig(repo_root=self.repo_root),
            create_if_missing=True,
        )

    def index_path(self, path: str) -> SymbolIndexResult:
        return self.index_path_with_prefetched(path)

    def index_path_with_prefetched(
        self,
        path: str,
        *,
        prefetched_files: list[ParsedFileSnapshot] | None = None,
        file_paths: list[str] | None = None,
    ) -> SymbolIndexResult:
        target = Path(os.path.abspath(path))
        started = time.perf_counter()
        result = SymbolIndexResult(db_path=self.store.db_path)
        prefetched_by_path = {
            os.path.abspath(snapshot.path): snapshot for snapshot in prefetched_files or []
        }

        files = (
            sorted(os.path.abspath(file_path) for file_path in file_paths)
            if file_paths is not None
            else (
                list(self._iter_source_files(target))
                if target.is_dir()
                else (
                    [str(target)]
                    if self._is_supported_file(str(target)) and not self._is_excluded(str(target))
                    else []
                )
            )
        )
        seen_paths: set[str] = set()
        for file_path in files:
            seen_paths.add(file_path)
            result.files_considered += 1
            prefetched = prefetched_by_path.get(file_path)
            mtime_ns = prefetched.mtime_ns if prefetched is not None else None
            if mtime_ns is None:
                try:
                    stat = os.stat(file_path)
                    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
                except OSError as exc:
                    result.add_failure(
                        "symbol_index_stat_error",
                        f"{file_path}: {type(exc).__name__}: {exc}",
                    )
                    continue
            previous = self.store.get_file(file_path)
            if prefetched is None and previous is not None and previous.mtime_ns == mtime_ns:
                result.files_unchanged += 1
                continue

            if prefetched is not None:
                source = prefetched.source
                content_hash = prefetched.content_hash
                span_index = prefetched.span_index
                language = prefetched.language
                symbols = prefetched.symbols
                if previous is not None and previous.content_hash == content_hash:
                    self.store.upsert_file(
                        IndexedFile(
                            path=file_path,
                            content_hash=content_hash,
                            mtime_ns=mtime_ns,
                            language=language,
                            last_indexed=datetime.now(timezone.utc),
                        )
                    )
                    result.files_unchanged += 1
                    continue
            else:
                try:
                    with open(file_path, "rb") as handle:
                        raw_source = handle.read()
                except OSError as exc:
                    result.add_failure(
                        "symbol_index_read_error",
                        f"{file_path}: {type(exc).__name__}: {exc}",
                    )
                    continue
                try:
                    source = raw_source.decode("utf8")
                except UnicodeDecodeError as exc:
                    result.add_failure(
                        "symbol_index_read_error",
                        f"{file_path}: {type(exc).__name__}: {exc}",
                    )
                    continue

                content_hash = self._hash_content(source)
                span_index = LineByteSpanIndex.from_bytes(raw_source)
                parser_entry = self.parser_registry.get_parser_for_path(file_path)
                language = parser_entry.language if parser_entry else None
                if previous is not None and previous.content_hash == content_hash:
                    self.store.upsert_file(
                        IndexedFile(
                            path=file_path,
                            content_hash=content_hash,
                            mtime_ns=mtime_ns,
                            language=language,
                            last_indexed=datetime.now(timezone.utc),
                        )
                    )
                    result.files_unchanged += 1
                    continue

                symbols = []
                if parser_entry is not None:
                    try:
                        symbols = parser_entry.parser.extract_symbols(file_path, source)
                    except Exception as exc:
                        result.add_failure(
                            "symbol_index_parse_error",
                            f"{file_path}: {type(exc).__name__}: {exc}",
                        )
                        continue

            definitions: list[SymbolOccurrence] = []
            for symbol in symbols:
                start_byte, end_byte = span_index.span_for_lines(
                    symbol.start_line,
                    symbol.end_line,
                )
                definitions.append(
                    SymbolOccurrence(
                        symbol=symbol.name,
                        kind="def",
                        path=file_path,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        start_byte=start_byte,
                        end_byte=end_byte,
                        language=symbol.language or language,
                        container=symbol.container_fqname,
                        signature=symbol.signature,
                    )
                )

            references = self._extract_references(
                source=source,
                path=file_path,
                language=language,
                definitions=symbols,
                span_index=span_index,
            )
            occurrences = definitions + references
            try:
                self.store.replace_file_occurrences(
                    indexed_file=IndexedFile(
                        path=file_path,
                        content_hash=content_hash,
                        mtime_ns=mtime_ns,
                        language=language,
                        last_indexed=datetime.now(timezone.utc),
                    ),
                    occurrences=occurrences,
                )
            except Exception as exc:
                result.add_failure(
                    "symbol_index_store_error",
                    f"{file_path}: {type(exc).__name__}: {exc}",
                )
                continue

            result.files_changed += 1
            result.defs_indexed += len(definitions)
            result.refs_indexed += len(references)

        if target.is_dir():
            try:
                result.files_removed = self.store.prune_missing_files(
                    seen_paths=seen_paths,
                    scope_prefix=str(target),
                )
            except Exception as exc:
                result.add_failure(
                    "symbol_index_prune_error",
                    f"{target}: {type(exc).__name__}: {exc}",
                )
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    def _iter_source_files(self, root: Path) -> list[str]:
        files: list[str] = []
        excludes = set(self.config.excluded_dirs)
        for current_root, dirs, filenames in os.walk(str(root)):
            dirs[:] = [name for name in dirs if name not in excludes]
            for filename in filenames:
                full_path = os.path.join(current_root, filename)
                if self._is_supported_file(full_path) and not self._is_excluded(full_path):
                    files.append(full_path)
        files.sort()
        return files

    def _is_supported_file(self, path: str) -> bool:
        return any(path.endswith(ext) for ext in self.config.supported_extensions)

    def _is_excluded(self, path: str) -> bool:
        normalized = set(os.path.normpath(path).split(os.sep))
        return any(excluded in normalized for excluded in self.config.excluded_dirs)

    @staticmethod
    def _hash_content(source: str) -> str:
        return hashlib.sha256(source.encode("utf8")).hexdigest()

    @staticmethod
    def _extract_references(
        *,
        source: str,
        path: str,
        language: str | None,
        definitions: list[Symbol],
        span_index: LineByteSpanIndex,
    ) -> list[SymbolOccurrence]:
        definition_lines: dict[int, set[str]] = {}
        for symbol in definitions:
            line_defs = definition_lines.setdefault(symbol.start_line, set())
            line_defs.add(symbol.name)

        refs: list[SymbolOccurrence] = []
        seen: set[tuple[str, str, int]] = set()
        for line_number, raw_line in enumerate(source.splitlines(), start=1):
            for token in _IDENTIFIER_RE.findall(raw_line):
                lowered = token.lower()
                if lowered in _REFERENCE_KEYWORDS:
                    continue
                if len(token) < 3:
                    continue
                if token in definition_lines.get(line_number, set()):
                    continue
                key = (token, path, line_number)
                if key in seen:
                    continue
                seen.add(key)
                start_byte, end_byte = span_index.span_for_lines(line_number, line_number)
                refs.append(
                    SymbolOccurrence(
                        symbol=token,
                        kind="ref",
                        path=path,
                        start_line=line_number,
                        end_line=line_number,
                        start_byte=start_byte,
                        end_byte=end_byte,
                        language=language,
                        container=None,
                        signature=None,
                    )
                )
        return refs
