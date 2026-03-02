from __future__ import annotations

import json
from dataclasses import dataclass

from gloggur.io_failures import wrap_io_error
from gloggur.storage.metadata_store import MetadataStore


@dataclass
class CoverageData:
    """Represents a generic mapping of test_symbol_id to executed lines."""

    # test_symbol_id -> { file_path: list_of_lines }
    contexts: dict[str, dict[str, list[int]]]


class CoverageIngester:
    """Ingests generic JSON coverage data and updates symbols with `covered_by`."""

    def __init__(self, metadata_store: MetadataStore) -> None:
        self._metadata = metadata_store

    def ingest_json(self, file_path: str) -> dict[str, int]:
        """Ingests a gloggur-coverage.json file and returns ingestion statistics."""
        try:
            with open(file_path, encoding="utf-8") as f:
                raw_data = json.load(f)
        except OSError as exc:
            raise wrap_io_error(exc, operation="read JSON coverage file", path=file_path) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in coverage file {file_path}: {exc}") from exc

        # Build CoverageData object from generic schema
        coverage_data = CoverageData(contexts={})
        for test_symbol_id, file_mapping in raw_data.items():
            if not isinstance(file_mapping, dict):
                raise ValueError(
                    f"Expected dict for mapping of test {test_symbol_id}, got {type(file_mapping)}"
                )

            normalized_mapping: dict[str, list[int]] = {}
            for path, lines in file_mapping.items():
                if not isinstance(lines, list) or not all(isinstance(line, int) for line in lines):
                    raise ValueError(
                        f"Expected list of integers for lines in {test_symbol_id}:{path}"
                    )
                normalized_mapping[path] = lines

            coverage_data.contexts[test_symbol_id] = normalized_mapping

        return self._apply_coverage(coverage_data)

    def _apply_coverage(self, data: CoverageData) -> dict[str, int]:
        """Applies dynamic contexts to the Symbol models in the metadata store."""
        # Find all files involved
        all_files: set[str] = set()
        for mapping in data.contexts.values():
            all_files.update(mapping.keys())

        # For efficiency, pull all symbols into memory for affected files
        file_to_symbols: dict[str, list[object]] = {}
        for path in all_files:
            file_to_symbols[path] = self._metadata.filter_symbols(file_path=path)

        # Apply coverage mappings
        # symbol_id -> set of test_symbol_ids that cover it
        symbol_updates: dict[str, set[str]] = {}

        for test_symbol_id, mapping in data.contexts.items():
            for path, lines in mapping.items():
                symbols = file_to_symbols.get(path, [])
                if not symbols:
                    continue
                # For each executed line, check which symbols contain that line
                for line in lines:
                    for s in symbols:
                        if s.start_line <= line <= s.end_line:
                            if s.id not in symbol_updates:
                                symbol_updates[s.id] = set(s.covered_by)
                            symbol_updates[s.id].add(test_symbol_id)

        # Construct batch update data for the cache interface
        from gloggur.models import Symbol

        symbols_to_update: list[Symbol] = []
        files_affected: set[str] = set()
        for _path, symbols in file_to_symbols.items():
            needs_update = False
            for s in symbols:
                new_covered_by = list(symbol_updates.get(s.id, set()))
                new_covered_by.sort()

                # Update if changed
                if new_covered_by != s.covered_by:
                    s.covered_by = new_covered_by
                    needs_update = True

            if needs_update:
                symbols_to_update.extend(symbols)
                files_affected.add(_path)

        # NOTE: MetadataStore doesn't expose upsert_symbols as it's read-only.
        # We need CacheManager here.
        # But for architectural separation, we'll return the symbols to be updated,
        # and the CLI will dispatch.

        return {
            "tests_processed": len(data.contexts),
            "files_affected": len(files_affected),
            "symbols_to_update": symbols_to_update,
        }
