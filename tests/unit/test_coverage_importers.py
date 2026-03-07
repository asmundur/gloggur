from __future__ import annotations

import json
import sqlite3

import pytest

import gloggur.coverage_importers as coverage_importers_module
from gloggur.config import GloggurConfig
from gloggur.coverage_importers import (
    CoverageImportError,
    JsonCoverageContextImporter,
    PythonCoverageSQLiteImporter,
    create_coverage_importer,
)


def test_json_coverage_importer_rejects_invalid_shapes(tmp_path) -> None:
    importer = JsonCoverageContextImporter()
    not_mapping = tmp_path / "invalid-root.json"
    invalid_lines = tmp_path / "invalid-lines.json"
    not_mapping.write_text(json.dumps(["bad"]), encoding="utf8")
    invalid_lines.write_text(json.dumps({"test_ok": {"file.py": ["1"]}}), encoding="utf8")

    with pytest.raises(CoverageImportError) as root_error:
        importer.import_contexts(str(not_mapping))
    with pytest.raises(CoverageImportError) as lines_error:
        importer.import_contexts(str(invalid_lines))

    assert root_error.value.error_code == "coverage_file_invalid"
    assert lines_error.value.error_code == "coverage_file_invalid"


def test_python_coverage_sqlite_importer_normalizes_contexts_and_skips_unmapped_rows(
    tmp_path,
) -> None:
    db_path = tmp_path / "coverage.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE context (id INTEGER PRIMARY KEY, context TEXT);
            CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT);
            CREATE TABLE line_data (file_id INTEGER, context_id INTEGER, line INTEGER);
            """)
        conn.execute("INSERT INTO context (id, context) VALUES (1, ?)", ("suite::test_add",))
        conn.execute("INSERT INTO context (id, context) VALUES (2, ?)", ("plain_context",))
        conn.execute("INSERT INTO file (id, path) VALUES (1, ?)", ("src/service.py",))
        conn.execute("INSERT INTO line_data (file_id, context_id, line) VALUES (1, 1, 12)")
        conn.execute("INSERT INTO line_data (file_id, context_id, line) VALUES (1, 2, 14)")
        conn.execute("INSERT INTO line_data (file_id, context_id, line) VALUES (99, 1, 20)")
        conn.execute("INSERT INTO line_data (file_id, context_id, line) VALUES (1, 99, 30)")
        conn.commit()
    finally:
        conn.close()

    importer = PythonCoverageSQLiteImporter()
    normalized = importer.import_contexts(str(db_path))
    raw = PythonCoverageSQLiteImporter(normalize_pytest_contexts=False).import_contexts(
        str(db_path)
    )

    assert normalized == {
        "test_add": {"src/service.py": [12]},
        "plain_context": {"src/service.py": [14]},
    }
    assert raw["suite::test_add"]["src/service.py"] == [12]


def test_create_coverage_importer_accepts_duck_typed_importer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DuckImporter:
        def import_contexts(self, source_path: str) -> dict[str, dict[str, list[int]]]:
            return {"ok": {source_path: [1]}}

    monkeypatch.setattr(
        coverage_importers_module._COVERAGE_IMPORTERS,
        "resolve_factory",
        lambda importer_id, module_path_override=None: (lambda: DuckImporter()),
    )

    importer = create_coverage_importer(GloggurConfig(cache_dir="/tmp/cache"), "duck")

    assert importer.import_contexts("source.json") == {"ok": {"source.json": [1]}}


def test_create_coverage_importer_rejects_invalid_factory_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coverage_importers_module._COVERAGE_IMPORTERS,
        "resolve_factory",
        lambda importer_id, module_path_override=None: (lambda: object()),
    )

    with pytest.raises(CoverageImportError) as error:
        create_coverage_importer(GloggurConfig(cache_dir="/tmp/cache"), "invalid")

    assert error.value.error_code == "coverage_file_invalid"


def test_create_coverage_importer_forwards_module_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_resolve_factory(importer_id: str, module_path_override: str | None = None):
        seen["importer_id"] = importer_id
        seen["module_path_override"] = module_path_override
        return lambda: JsonCoverageContextImporter()

    monkeypatch.setattr(
        coverage_importers_module._COVERAGE_IMPORTERS,
        "resolve_factory",
        fake_resolve_factory,
    )

    importer = create_coverage_importer(
        GloggurConfig(
            cache_dir="/tmp/cache",
            adapters={"coverage_importers": {"json": "custom.module:factory"}},
        ),
        "json",
    )

    assert isinstance(importer, JsonCoverageContextImporter)
    assert seen == {
        "importer_id": "json",
        "module_path_override": "custom.module:factory",
    }
