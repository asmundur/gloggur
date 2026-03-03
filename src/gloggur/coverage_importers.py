from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from gloggur.adapters.registry import AdapterRegistry, adapter_module_override, instantiate_adapter
from gloggur.config import GloggurConfig


class CoverageImportError(RuntimeError):
    """Structured coverage import failure with stable CLI-facing error code."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class CoverageImporter:
    """Coverage importer interface for mapping source formats into context payloads."""

    adapter_id: str

    def import_contexts(self, source_path: str) -> dict[str, dict[str, list[int]]]:
        """Parse input source and return normalized context mappings."""
        raise NotImplementedError


@dataclass
class PythonCoverageSQLiteImporter(CoverageImporter):
    """Import contexts from coverage.py sqlite databases."""

    adapter_id: str = "python"
    normalize_pytest_contexts: bool = True

    def import_contexts(self, source_path: str) -> dict[str, dict[str, list[int]]]:
        contexts: dict[str, dict[str, list[int]]] = {}
        try:
            conn = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise CoverageImportError(
                f"SQLite error reading coverage db: {exc}",
                error_code="coverage_sqlite_invalid",
            ) from exc
        try:
            context_map: dict[int, str] = {}
            for row in conn.execute("SELECT id, context FROM context"):
                if row[1]:
                    context_map[int(row[0])] = str(row[1])

            file_map: dict[int, str] = {}
            for row in conn.execute("SELECT id, path FROM file"):
                file_map[int(row[0])] = str(row[1])

            try:
                rows = conn.execute("SELECT file_id, context_id, line FROM line_data")
            except sqlite3.OperationalError as exc:
                raise CoverageImportError(
                    "Missing 'line_data' table. Coverage file might not be "
                    "from coverage.py version 7+.",
                    error_code="coverage_sqlite_invalid",
                ) from exc

            for file_id, context_id, line in rows:
                raw_ctx = context_map.get(int(context_id))
                if not raw_ctx:
                    continue
                test_context = self._normalize_context(raw_ctx)
                file_path = file_map.get(int(file_id))
                if not file_path:
                    continue
                if test_context not in contexts:
                    contexts[test_context] = {}
                if file_path not in contexts[test_context]:
                    contexts[test_context][file_path] = []
                contexts[test_context][file_path].append(int(line))
        except sqlite3.Error as exc:
            raise CoverageImportError(
                f"SQLite error reading coverage db: {exc}",
                error_code="coverage_sqlite_invalid",
            ) from exc
        finally:
            conn.close()
        return contexts

    def _normalize_context(self, raw_context: str) -> str:
        """Normalize coverage context labels in adapter-specific fashion."""
        if not self.normalize_pytest_contexts:
            return raw_context
        parts = raw_context.split("::")
        return parts[-1] if len(parts) > 1 else raw_context


@dataclass
class JsonCoverageContextImporter(CoverageImporter):
    """Import contexts from generic JSON map format."""

    adapter_id: str = "json"

    def import_contexts(self, source_path: str) -> dict[str, dict[str, list[int]]]:
        try:
            with open(source_path, encoding="utf8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise CoverageImportError(
                f"Failed reading coverage file: {exc}",
                error_code="coverage_file_missing",
            ) from exc
        except json.JSONDecodeError as exc:
            raise CoverageImportError(
                f"Invalid JSON in coverage file: {exc}",
                error_code="coverage_file_invalid",
            ) from exc

        if not isinstance(payload, dict):
            raise CoverageImportError(
                "Coverage JSON payload must be an object mapping contexts to file/line maps.",
                error_code="coverage_file_invalid",
            )
        normalized: dict[str, dict[str, list[int]]] = {}
        for raw_context, raw_file_map in payload.items():
            context = str(raw_context)
            if not isinstance(raw_file_map, dict):
                raise CoverageImportError(
                    f"Coverage context '{context}' must map to an object.",
                    error_code="coverage_file_invalid",
                )
            file_map: dict[str, list[int]] = {}
            for raw_path, raw_lines in raw_file_map.items():
                path = str(raw_path)
                if not isinstance(raw_lines, list) or not all(
                    isinstance(line, int) for line in raw_lines
                ):
                    raise CoverageImportError(
                        f"Coverage lines for '{context}:{path}' must be integer lists.",
                        error_code="coverage_file_invalid",
                    )
                file_map[path] = [int(line) for line in raw_lines]
            normalized[context] = file_map
        return normalized


_COVERAGE_IMPORTERS = AdapterRegistry[CoverageImporter]("gloggur.coverage_importers")
_COVERAGE_IMPORTERS.register_builtin("python", lambda: PythonCoverageSQLiteImporter())
_COVERAGE_IMPORTERS.register_builtin(
    "python-coverage-sqlite",
    lambda: PythonCoverageSQLiteImporter(),
)
_COVERAGE_IMPORTERS.register_builtin("json", lambda: JsonCoverageContextImporter())


def create_coverage_importer(config: GloggurConfig, importer_id: str) -> CoverageImporter:
    """Create one coverage importer from builtins/entrypoints/module overrides."""
    module_override = adapter_module_override(
        config.adapters if isinstance(config.adapters, dict) else None,
        category="coverage_importers",
        adapter_id=importer_id,
    )
    factory = _COVERAGE_IMPORTERS.resolve_factory(
        importer_id,
        module_path_override=module_override,
    )
    importer = instantiate_adapter(factory)
    if not isinstance(importer, CoverageImporter):
        if not hasattr(importer, "import_contexts"):
            raise CoverageImportError(
                f"Coverage importer '{importer_id}' is invalid ({type(importer).__name__}).",
                error_code="coverage_file_invalid",
            )
    return importer


def list_coverage_importers() -> list[dict[str, object]]:
    """Return discoverable coverage importer adapter descriptors."""
    return _COVERAGE_IMPORTERS.available()
