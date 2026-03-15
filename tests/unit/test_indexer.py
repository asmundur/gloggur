from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.errors import EmbeddingProviderError
from gloggur.indexer.cache import (
    BUILD_FILE_CHECKPOINT_STATE_EMBEDDED_COMPLETE,
    BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE,
    CacheConfig,
    CacheManager,
)
from gloggur.indexer.indexer import (
    EXTRACT_SYMBOLS_TIMEOUT_REASON,
    Indexer,
    VerboseLineMetricsAccumulator,
)
from gloggur.models import EdgeRecord, SymbolChunk
from gloggur.parsers.registry import ParserRegistry
from scripts.verification.fixtures import TestFixtures


def _write_timeout_parser_module(tmp_path: Path, *, sleep_seconds: float = 0.2) -> str:
    module_name = f"gloggur_test_timeout_parser_{next(tempfile._get_candidate_names())}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        textwrap.dedent(
            f"""
            from __future__ import annotations

            import hashlib
            import os
            import time

            from gloggur.models import Symbol
            from gloggur.parsers.base import ParsedFile, Parser


            class TimeoutParser(Parser):
                def parse_file(self, path: str, source: str) -> ParsedFile:
                    return ParsedFile(
                        path=path,
                        language="python",
                        source=source,
                        symbols=self.extract_symbols(path, source),
                    )

                def extract_symbols(self, path: str, source: str) -> list[Symbol]:
                    if os.path.basename(path) == "b_hang.py":
                        time.sleep({sleep_seconds})
                    body_hash = hashlib.sha256(source.encode("utf8")).hexdigest()
                    return [
                        Symbol(
                            id=f"{{path}}::sample",
                            name="sample",
                            kind="function",
                            fqname="sample",
                            file_path=path,
                            start_line=1,
                            end_line=2,
                            body_hash=body_hash,
                            language="python",
                        )
                    ]

                def get_supported_languages(self):
                    return ["python"]


            def create_parser() -> TimeoutParser:
                return TimeoutParser()
            """
        ),
        encoding="utf8",
    )
    return module_name


def test_indexer_indexes_repo_and_sets_metadata() -> None:
    """Indexer should index repo and set metadata."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        result = indexer.index_repository(str(repo))

        assert result.indexed_files == 1
        assert result.files_changed == 1
        assert result.files_removed == 0
        assert result.symbols_added > 0
        assert result.symbols_updated == 0
        assert result.symbols_removed == 0
        assert result.indexed_symbols > 0
        assert result.skipped_files == 0
        payload = result.as_payload()
        for field in (
            "files_scanned",
            "files_changed",
            "files_removed",
            "symbols_added",
            "symbols_updated",
            "symbols_removed",
        ):
            assert field in payload
        index_stats = payload["index_stats"]
        assert index_stats["symbol_count"] == len(cache.list_symbols())
        assert index_stats["chunk_count"] == len(cache.list_chunks())
        assert index_stats["graph_edge_count"] == len(cache.list_edges())
        assert index_stats["embedded_edge_vectors"] == 0
        metadata = cache.get_index_metadata()
        assert metadata is not None
        assert metadata.indexed_files == 1
        assert metadata.total_symbols == len(cache.list_symbols())
        search_integrity = cache.get_search_integrity()
        assert isinstance(search_integrity, dict)
        assert search_integrity["vector_cache"]["status"] == "missing"
        assert search_integrity["chunk_span"]["status"] == "passed"


def test_verbose_line_metrics_accumulator_tracks_duplicates_and_edge_additions() -> None:
    """Verbose line accounting should distinguish duplicated and unique coverage."""
    accumulator = VerboseLineMetricsAccumulator()
    accumulator.add_source_lines(6)
    accumulator.add_symbol_chunks(
        [
            SymbolChunk(
                chunk_id="chunk-1",
                symbol_id="sym-1",
                chunk_part_index=1,
                chunk_part_total=1,
                text="alpha",
                file_path="sample.py",
                start_line=1,
                end_line=3,
                embedding_vector=[0.1, 0.2],
            ),
            SymbolChunk(
                chunk_id="chunk-2",
                symbol_id="sym-2",
                chunk_part_index=1,
                chunk_part_total=1,
                text="beta",
                file_path="sample.py",
                start_line=3,
                end_line=4,
                embedding_vector=[0.3, 0.4],
            ),
            SymbolChunk(
                chunk_id="chunk-ignored",
                symbol_id="sym-3",
                chunk_part_index=1,
                chunk_part_total=1,
                text="ignored",
                file_path="sample.py",
                start_line=5,
                end_line=5,
                embedding_vector=None,
            ),
        ]
    )
    accumulator.add_graph_edges(
        [
            EdgeRecord(
                edge_id="edge-1",
                edge_type="CALLS",
                from_id="sym-1",
                to_id="sym-2",
                from_kind="function",
                to_kind="function",
                file_path="sample.py",
                line=3,
                confidence=1.0,
                embedding_vector=[0.5, 0.6],
            ),
            EdgeRecord(
                edge_id="edge-2",
                edge_type="CALLS",
                from_id="sym-1",
                to_id="sym-4",
                from_kind="function",
                to_kind="function",
                file_path="sample.py",
                line=6,
                confidence=1.0,
                embedding_vector=[0.7, 0.8],
            ),
            EdgeRecord(
                edge_id="edge-ignored",
                edge_type="CALLS",
                from_id="sym-1",
                to_id="sym-5",
                from_kind="function",
                to_kind="function",
                file_path="sample.py",
                line=2,
                confidence=1.0,
                embedding_vector=None,
            ),
        ]
    )

    metrics = accumulator.build()

    assert metrics.source_total == 6
    assert metrics.embedded_total == 7
    assert metrics.embedded_unique == 5
    assert metrics.embedded_duplicate == 2
    assert metrics.vector_count == 4
    assert metrics.symbol_chunks.line_total == 5
    assert metrics.symbol_chunks.line_unique == 4
    assert metrics.graph_edges.line_total == 2
    assert metrics.graph_edges.line_unique == 2


def test_indexer_verbose_line_metrics_count_physical_lines_across_line_endings() -> None:
    """Verbose source totals should follow the shared byte-span line model."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({})
        (repo / "lf.py").write_text("def alpha():\n    return 1\n", encoding="utf8")
        (repo / "nonewline.py").write_text("def beta():\n    return 2", encoding="utf8")
        (repo / "empty.py").write_text("", encoding="utf8")
        (repo / "crlf.py").write_bytes(b"def gamma():\r\n    return 3\r\n")
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=cache,
            parser_registry=ParserRegistry(),
        )

        result = indexer.index_repository(str(repo), capture_verbose_metrics=True)

        assert result.failed == 0
        assert result.verbose_lines is not None
        assert result.verbose_lines.source_total == 6


def test_indexer_verbose_line_metrics_track_duplicates_and_cache_reuse() -> None:
    """Verbose metrics should be stable across unchanged reruns."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    source = (
        "class Example:\n"
        "    def first(self) -> int:\n"
        "        return 1\n"
        "    def second(self) -> int:\n"
        "        return self.first()\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
        )

        first = indexer.index_repository(str(repo), capture_verbose_metrics=True)
        second = indexer.index_repository(str(repo), capture_verbose_metrics=True)

        assert first.failed == 0
        assert first.verbose_lines is not None
        assert first.verbose_lines.source_total == 5
        assert first.verbose_lines.embedded_total == 9
        assert first.verbose_lines.embedded_unique == 5
        assert first.verbose_lines.embedded_duplicate == 4
        assert first.verbose_lines.vector_count == 3
        assert first.verbose_lines.graph_edges.vector_count == 0
        assert second.failed == 0
        assert second.unchanged == 1
        assert second.verbose_lines == first.verbose_lines


def test_indexer_verbose_line_metrics_include_edge_vectors_when_enabled() -> None:
    """Verbose metrics should count optional embedded graph edges separately."""

    class RecordingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    source = (
        "def helper(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def caller(value: int) -> int:\n"
        "    return helper(value)\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir, embed_graph_edges=True),
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=RecordingProvider(),
        )

        result = indexer.index_repository(str(repo), capture_verbose_metrics=True)

        assert result.failed == 0
        assert result.verbose_lines is not None
        assert result.verbose_lines.symbol_chunks.vector_count > 0
        assert result.verbose_lines.graph_edges.vector_count > 0
        assert result.verbose_lines.graph_edges.line_total >= result.verbose_lines.graph_edges.line_unique
        assert (
            result.verbose_lines.embedded_total
            == result.verbose_lines.symbol_chunks.line_total
            + result.verbose_lines.graph_edges.line_total
        )


def test_indexer_verbose_line_metrics_count_source_lines_for_failed_files() -> None:
    """Failed files should still contribute source lines but no embedded coverage."""

    class ExplodingParser:
        def extract_symbols(self, _path: str, _source: str) -> list[object]:
            raise RuntimeError("parse exploded")

    class FakeParserEntry:
        language = "python"
        parser = ExplodingParser()

    class FakeParserRegistry:
        def get_parser_for_path(self, _path: str) -> FakeParserEntry:
            return FakeParserEntry()

    source = "def broken():\n    return 1\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=cache,
            parser_registry=FakeParserRegistry(),
        )

        result = indexer.index_repository(str(repo), capture_verbose_metrics=True)

        assert result.failed == 1
        assert result.verbose_lines is not None
        assert result.verbose_lines.source_total == 2
        assert result.verbose_lines.embedded_total == 0
        assert result.verbose_lines.embedded_unique == 0
        assert result.verbose_lines.vector_count == 0


def test_indexer_verbose_line_metrics_single_file_matches_repo_scope() -> None:
    """Single-file detailed indexing should emit the same verbose metrics as repo indexing."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    source = (
        "class Example:\n"
        "    def first(self) -> int:\n"
        "        return 1\n"
        "    def second(self) -> int:\n"
        "        return self.first()\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        repo_cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        single_cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        repo_indexer = Indexer(
            config=GloggurConfig(cache_dir=repo_cache_dir),
            cache=CacheManager(CacheConfig(repo_cache_dir)),
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        single_indexer = Indexer(
            config=GloggurConfig(cache_dir=single_cache_dir),
            cache=CacheManager(CacheConfig(single_cache_dir)),
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
        )

        repo_result = repo_indexer.index_repository(str(repo), capture_verbose_metrics=True)
        execution = single_indexer.index_file_with_details(
            str(repo / "sample.py"),
            capture_verbose_metrics=True,
        )

        assert repo_result.failed == 0
        assert execution.outcome.status == "indexed"
        assert repo_result.verbose_lines is not None
        assert execution.verbose_lines == repo_result.verbose_lines


def test_indexer_indexes_mixed_c_cpp_files_without_parser_unavailable_failures() -> None:
    """Indexer should parse default C/C++ source and header extensions without parser gaps."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "main.c": "int add(int a, int b) { return a + b; }\n",
                "include/math.h": "int declared(int value);\n",
                "include/callbacks.h": "int (*make_cb(void))(int);\nint (*fp)(int);\n",
                "include/greeter.hpp": (
                    "namespace core {\n"
                    "class Greeter {\n"
                    "public:\n"
                    "  static int ping();\n"
                    "};\n"
                    "}\n"
                ),
                "src/greeter.cpp": (
                    "int core::Greeter::ping() { return 2; }\n"
                    "#define DECL_METHOD(Type, Name) int Type::Name() { return 0; }\n"
                    "DECL_METHOD(Greeter, macro_ping)\n"
                ),
                "src/operators.cpp": (
                    "struct Vec { int operator[](int idx) const; };\n"
                    "int Vec::operator[](int idx) const { return idx; }\n"
                ),
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        result = indexer.index_repository(str(repo))

        assert result.files_considered == 6
        assert result.failed == 0
        assert result.indexed_symbols > 0
        assert "parser_unavailable" not in result.failed_reasons


def test_indexer_skips_minified_js_by_default_and_can_include_with_config() -> None:
    """Repository scan should exclude `.min.js` unless explicitly opted in."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "app.js": "function appMain() { return 1; }\n",
                "vendor/jquery.min.js": "function minifiedVendor(){return 1};",
            }
        )

        default_cache = tempfile.mkdtemp(prefix="gloggur-cache-")
        default_indexer = Indexer(
            config=GloggurConfig(cache_dir=default_cache),
            cache=CacheManager(CacheConfig(default_cache)),
            parser_registry=ParserRegistry(),
        )
        default_result = default_indexer.index_repository(str(repo))

        assert default_result.failed == 0
        assert default_result.files_considered == 1
        assert all(not path.endswith("jquery.min.js") for path in default_result.source_files)

        include_cache = tempfile.mkdtemp(prefix="gloggur-cache-")
        include_indexer = Indexer(
            config=GloggurConfig(cache_dir=include_cache, include_minified_js=True),
            cache=CacheManager(CacheConfig(include_cache)),
            parser_registry=ParserRegistry(),
        )
        include_result = include_indexer.index_repository(str(repo))

        assert include_result.failed == 0
        assert include_result.files_considered == 2
        assert any(path.endswith("jquery.min.js") for path in include_result.source_files)


def test_indexer_skips_django_vendor_minified_paths_by_default() -> None:
    """Django-style vendored minified paths should be treated as out-of-scope by default."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "django/contrib/admin/static/admin/js/vendor/jquery/jquery.min.js": (
                    "function jqueryLike(){return 1};"
                ),
                "django/contrib/admin/static/admin/js/vendor/select2/select2.full.min.js": (
                    "function select2Like(){return 2};"
                ),
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=CacheManager(CacheConfig(cache_dir)),
            parser_registry=ParserRegistry(),
        )

        result = indexer.index_repository(str(repo))

        assert result.failed == 0
        assert result.files_considered == 0
        assert "chunk_span_integrity_error" not in result.failed_reasons


def test_indexer_skips_structural_virtualenv_roots_and_keeps_sibling_project_code() -> None:
    """Repository scan should skip arbitrary-name virtualenv roots but keep sibling sources."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "src/app.py": "def app() -> int:\n    return 1\n",
                "tools/helper.py": "def helper() -> int:\n    return 2\n",
                "_venv/Lib/site-packages/vendor_win.py": "def vendor_win() -> int:\n    return 3\n",
                "python-runtime/lib/python3.13/site-packages/vendor.py": (
                    "def vendor() -> int:\n    return 4\n"
                ),
            }
        )
        (repo / "_venv" / "Scripts").mkdir(parents=True)
        (repo / "_venv" / "Scripts" / "python.exe").write_text("", encoding="utf8")
        (repo / "python-runtime" / "bin").mkdir(parents=True)
        (repo / "python-runtime" / "bin" / "python").write_text("", encoding="utf8")

        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=CacheManager(CacheConfig(cache_dir)),
            parser_registry=ParserRegistry(),
        )

        result = indexer.index_repository(str(repo))

        assert result.failed == 0
        assert result.files_considered == 2
        assert sorted(result.source_files) == [
            str(repo / "src" / "app.py"),
            str(repo / "tools" / "helper.py"),
        ]


def test_indexer_fails_closed_on_chunk_span_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed chunk spans should fail the index run with an explicit integrity code."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        original_build_chunks = indexer._build_symbol_chunks

        def _bad_chunks(*args: object, **kwargs: object) -> list[SymbolChunk]:
            chunks = original_build_chunks(*args, **kwargs)
            assert chunks
            broken = chunks[0].model_copy(deep=True)
            broken.end_line = broken.end_line + 100
            return [broken, *chunks[1:]]

        monkeypatch.setattr(indexer, "_build_symbol_chunks", _bad_chunks)

        result = indexer.index_repository(str(repo))

        assert result.failed == 1
        assert result.failed_reasons == {"chunk_span_integrity_error": 1}
        assert any("escapes symbol" in sample for sample in result.failed_samples)
        search_integrity = cache.get_search_integrity()
        assert isinstance(search_integrity, dict)
        assert search_integrity["chunk_span"]["status"] == "failed"


def test_indexer_skips_unchanged_files() -> None:
    """Indexer should skip unchanged files on reindex."""
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        first = indexer.index_file(str(repo / "sample.py"))
        second = indexer.index_file(str(repo / "sample.py"))

        assert first is not None
        assert second is None
        metadata = cache.get_file_metadata(str(repo / "sample.py"))
        assert metadata is not None
        assert metadata.content_hash


def test_indexer_replaces_changed_file_symbols_and_updates_counts() -> None:
    """Reindexing a changed file should replace stale symbol rows and refresh totals."""
    initial_source = "def alpha(value: int) -> int:\n" "    return value + 1\n"
    updated_source = "def beta(value: int) -> int:\n" "    return value + 2\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": initial_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        first = indexer.index_repository(str(repo))
        assert first.failed == 0
        first_symbols = cache.list_symbols_for_file(str(repo / "sample.py"))
        assert [symbol.name for symbol in first_symbols] == ["alpha"]

        (repo / "sample.py").write_text(updated_source, encoding="utf8")
        second = indexer.index_repository(str(repo))

        assert second.failed == 0
        assert second.indexed_files == 1
        assert second.symbols_added == 1
        assert second.symbols_removed == 1
        assert second.symbols_updated == 0
        refreshed_symbols = cache.list_symbols_for_file(str(repo / "sample.py"))
        assert [symbol.name for symbol in refreshed_symbols] == ["beta"]
        assert cache.count_symbols() == 1
        metadata = cache.get_index_metadata()
        assert metadata is not None
        assert metadata.total_symbols == 1


def test_indexer_prunes_deleted_files_and_reports_symbol_removals() -> None:
    """Reindex should remove stale file symbols when files are deleted from disk."""
    keep_source = "def keep_me(value: int) -> int:\n" "    return value + 1\n"
    drop_source = "def drop_me(value: int) -> int:\n" "    return value + 2\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": keep_source, "drop.py": drop_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        first = indexer.index_repository(str(repo))
        assert first.indexed_files == 2
        assert first.files_removed == 0

        drop_path = repo / "drop.py"
        drop_metadata = cache.get_file_metadata(str(drop_path))
        assert drop_metadata is not None
        removed_symbol_count = len(drop_metadata.symbols)
        assert removed_symbol_count > 0

        drop_path.unlink()
        second = indexer.index_repository(str(repo))

        assert second.indexed_files == 0
        assert second.skipped_files == 1
        assert second.files_removed == 1
        assert second.symbols_removed == removed_symbol_count
        assert cache.get_file_metadata(str(drop_path)) is None
        assert cache.list_symbols_for_file(str(drop_path)) == []


def test_indexer_surfaces_stale_cleanup_failures_with_deterministic_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup failure for stale files should not report success and should include remediation."""
    keep_source = "def keep_me(value: int) -> int:\n" "    return value + 1\n"
    drop_source = "def drop_me(value: int) -> int:\n" "    return value + 2\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": keep_source, "drop.py": drop_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        first = indexer.index_repository(str(repo))
        assert first.failed == 0

        stale_path = str(repo / "drop.py")
        (repo / "drop.py").unlink()
        original_delete = cache.delete_file_metadata

        def _raise_on_stale_delete(path: str) -> None:
            if path == stale_path:
                raise OSError("simulated stale cleanup failure")
            original_delete(path)

        monkeypatch.setattr(cache, "delete_file_metadata", _raise_on_stale_delete)
        second = indexer.index_repository(str(repo))

        assert second.failed == 1
        assert second.failed_reasons == {"stale_cleanup_error": 1}
        assert any(stale_path in sample for sample in second.failed_samples)
        payload = second.as_payload()
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert payload["failure_codes"] == ["stale_cleanup_error"]
        assert "stale_cleanup_error" in guidance
        assert isinstance(guidance["stale_cleanup_error"], list)
        assert guidance["stale_cleanup_error"]


def test_indexer_persists_graph_edges_without_edge_embeddings_by_default() -> None:
    """Default indexing should keep graph edges structural-only."""

    class RecordingProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self.batch_texts: list[list[str]] = []

        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            self.batch_texts.append(list(texts))
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    source = (
        "import math\n\n"
        "def helper(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def target(value: int) -> int:\n"
        "    computed = helper(value)\n"
        "    return unknown_api(computed)\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        embedding_provider = RecordingProvider()
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=embedding_provider,
        )

        result = indexer.index_repository(str(repo))

        assert result.failed == 0
        edges = cache.list_edges_for_file(str(repo / "sample.py"))
        edge_types = {edge.edge_type for edge in edges}
        assert "DEFINES" in edge_types
        assert "CALLS" in edge_types
        assert "IMPORTS" in edge_types
        unresolved_calls = [
            edge for edge in edges if edge.edge_type == "CALLS" and "unresolved:" in edge.to_id
        ]
        assert unresolved_calls
        assert all(edge.text is None for edge in edges)
        assert all(edge.embedding_vector is None for edge in edges)
        assert not any(
            text.startswith("EDGE_TYPE:")
            for batch in embedding_provider.batch_texts
            for text in batch
        )
        index_stats = cache.get_index_stats()
        assert index_stats["graph_edge_count"] == len(edges)
        assert index_stats["embedded_edge_vectors"] == 0
        assert index_stats["embedded_vector_count"] == index_stats["embedded_symbol_vectors"]


def test_indexer_opt_in_restores_edge_text_and_embeddings() -> None:
    """Enabling edge embeddings should restore edge text serialization and vectors."""

    class RecordingProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self.batch_texts: list[list[str]] = []

        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            self.batch_texts.append(list(texts))
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    source = (
        "import math\n\n"
        "def helper(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def target(value: int) -> int:\n"
        "    computed = helper(value)\n"
        "    return unknown_api(computed)\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir, embed_graph_edges=True)
        cache = CacheManager(CacheConfig(cache_dir))
        embedding_provider = RecordingProvider()
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=embedding_provider,
        )

        result = indexer.index_repository(str(repo))

        assert result.failed == 0
        edges = cache.list_edges_for_file(str(repo / "sample.py"))
        assert edges
        assert all(
            isinstance(edge.text, str) and edge.text.startswith("EDGE_TYPE:") for edge in edges
        )
        assert all(edge.embedding_vector is not None for edge in edges)
        assert any(
            text.startswith("EDGE_TYPE:")
            for batch in embedding_provider.batch_texts
            for text in batch
        )
        index_stats = cache.get_index_stats()
        assert index_stats["graph_edge_count"] == len(edges)
        assert index_stats["embedded_edge_vectors"] == len(edges)
        assert index_stats["embedded_vector_count"] > index_stats["embedded_symbol_vectors"]


def test_indexer_removes_edges_for_deleted_files() -> None:
    """Reindexing after file deletion should garbage-collect file-local edges."""
    source = "def keep(value: int) -> int:\n" "    return value + 1\n"
    drop_source = "def drop(value: int) -> int:\n" "    return value + 2\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": source, "drop.py": drop_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        first = indexer.index_repository(str(repo))
        assert first.failed == 0
        drop_path = str(repo / "drop.py")
        assert cache.list_edges_for_file(drop_path)

        os.remove(drop_path)
        second = indexer.index_repository(str(repo))
        assert second.failed == 0
        assert cache.list_edges_for_file(drop_path) == []


def test_indexer_rolls_back_file_state_after_transient_vector_upsert_failure() -> None:
    """Transient vector persistence failures should not leave sticky cache/vector divergence."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    class FlakyVectorStore:
        def __init__(self) -> None:
            self._ids: set[str] = set()
            self._fail_once = True

        def remove_ids(self, symbol_ids: list[str]) -> None:
            for symbol_id in symbol_ids:
                self._ids.discard(symbol_id)

        def upsert_vectors(self, symbols: list[object]) -> None:
            if self._fail_once:
                self._fail_once = False
                raise RuntimeError("simulated transient vector upsert failure")
            for symbol in symbols:
                symbol_id = getattr(symbol, "chunk_id", None)
                if isinstance(symbol_id, str):
                    self._ids.add(symbol_id)

        def list_symbol_ids(self) -> list[str]:
            return sorted(self._ids)

        def save(self) -> None:
            return

    source = "def alpha(value: int) -> int:\n" "    return value + 1\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        vector_store = FlakyVectorStore()
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=vector_store,
        )

        first = indexer.index_repository(str(repo))

        assert first.failed == 1
        assert first.failed_reasons == {"storage_error": 1}
        assert first.indexed_symbols == 0
        sample_path = str(repo / "sample.py")
        checkpoint = cache.get_build_file_checkpoint(sample_path)
        assert checkpoint is not None
        assert checkpoint.state == BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE
        assert cache.get_file_metadata(sample_path) is not None
        assert cache.count_symbols() == 1
        assert all(chunk.embedding_vector is None for chunk in cache.list_chunks_for_file(sample_path))
        assert vector_store.list_symbol_ids() == []

        second = indexer.index_repository(str(repo))

        assert second.failed == 0
        assert second.indexed_files == 1
        assert second.indexed_symbols == 1
        metadata = cache.get_file_metadata(sample_path)
        assert metadata is not None
        assert cache.count_symbols() == 1
        expected_chunk_ids = [chunk.chunk_id for chunk in cache.list_chunks_for_file(sample_path)]
        assert vector_store.list_symbol_ids() == expected_chunk_ids


def test_indexer_detects_vector_metadata_mismatch_under_symbol_removal() -> None:
    """Vector/cache divergence should fail closed with a stable mismatch reason code."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    class FaultyVectorStore:
        def __init__(self) -> None:
            self._ids: set[str] = set()

        def remove_ids(self, _symbol_ids: list[str]) -> None:
            # Intentional no-op to emulate stale vectors surviving symbol removals.
            return

        def upsert_vectors(self, symbols: list[object]) -> None:
            for symbol in symbols:
                symbol_id = getattr(symbol, "chunk_id", None)
                if isinstance(symbol_id, str):
                    self._ids.add(symbol_id)

        def list_symbol_ids(self) -> list[str]:
            return sorted(self._ids)

        def save(self) -> None:
            return

    source_with_two = (
        "def alpha(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def beta(value: int) -> int:\n"
        "    return value + 2\n"
    )
    source_with_one = "def alpha(value: int) -> int:\n" "    return value + 1\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source_with_two})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        vector_store = FaultyVectorStore()
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=vector_store,
        )

        first = indexer.index_repository(str(repo))
        assert first.failed == 0

        (repo / "sample.py").write_text(source_with_one, encoding="utf8")
        second = indexer.index_repository(str(repo))

        assert second.failed == 1
        assert second.failed_reasons == {"vector_metadata_mismatch": 1}
        assert any("vector/cache mismatch" in sample for sample in second.failed_samples)
        payload = second.as_payload()
        guidance = payload["failure_guidance"]
        assert payload["failure_codes"] == ["vector_metadata_mismatch"]
        assert isinstance(guidance, dict)
        assert "vector_metadata_mismatch" in guidance
        assert guidance["vector_metadata_mismatch"]


def test_indexer_fails_closed_when_vector_consistency_is_unverifiable() -> None:
    """Indexing should not report success when vector/cache consistency cannot be verified."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            size = float(len(text))
            return [size, size + 1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def get_dimension(self) -> int:
            return 2

    class OpaqueVectorStore:
        def remove_ids(self, _symbol_ids: list[str]) -> None:
            return

        def upsert_vectors(self, _symbols: list[object]) -> None:
            return

        def save(self) -> None:
            return

    source = "def alpha(value: int) -> int:\n" "    return value + 1\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=OpaqueVectorStore(),
        )

        result = indexer.index_repository(str(repo))

        assert result.failed == 1
        assert result.failed_reasons == {"vector_consistency_unverifiable": 1}
        assert any("list_symbol_ids" in sample for sample in result.failed_samples)
        payload = result.as_payload()
        assert payload["failure_codes"] == ["vector_consistency_unverifiable"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "vector_consistency_unverifiable" in guidance
        assert isinstance(guidance["vector_consistency_unverifiable"], list)
        assert guidance["vector_consistency_unverifiable"]


def test_indexer_scan_callback_reports_done_and_total_counts() -> None:
    """Repository scan callback should receive deterministic done/total progress values."""
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "a.py": "def a() -> int:\n    return 1\n",
                "b.py": "def b() -> int:\n    return 2\n",
                "c.py": "def c() -> int:\n    return 3\n",
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())
        scan_calls: list[tuple[int, int, str]] = []

        def _scan(done: int, total: int, status: str) -> None:
            scan_calls.append((done, total, status))

        indexer._scan_callback = _scan

        result = indexer.index_repository(str(repo))

    assert result.files_considered == 3
    assert len(scan_calls) == 3
    assert [done for done, _, _ in scan_calls] == [1, 2, 3]
    assert all(total == 3 for _, total, _ in scan_calls)
    assert scan_calls[-1][0] == scan_calls[-1][1]
    assert all(status == "prepared" for _, _, status in scan_calls)


def _make_embedding_test_chunks(total: int) -> list[SymbolChunk]:
    """Build deterministic symbol chunks for _apply_embeddings unit tests."""
    return [
        SymbolChunk(
            chunk_id=f"chunk-{index}",
            symbol_id=f"sym-{index}",
            chunk_part_index=1,
            chunk_part_total=1,
            text=f"def fn_{index}():\n    return {index}",
            file_path="f.py",
            start_line=1,
            end_line=2,
        )
        for index in range(total)
    ]


@pytest.mark.parametrize("provider_name", ["local", "openai", "gemini"])
def test_apply_embeddings_uses_same_batching_for_local_and_remote_provider_configs(
    provider_name: str,
) -> None:
    """Batching shape should be identical regardless of provider id in config."""

    class RecordingProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self._chunk_size = 4
            self.batch_sizes: list[int] = []

        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.1, 0.2]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            payload = list(texts)
            self.batch_sizes.append(len(payload))
            return [[0.1, 0.2] for _ in payload]

        def get_dimension(self) -> int:
            return 2

    chunks = _make_embedding_test_chunks(9)
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(cache_dir=cache_dir, embedding_provider=provider_name)
    cache = CacheManager(CacheConfig(cache_dir))
    provider = RecordingProvider()
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=provider,
    )

    result = indexer._apply_embeddings(chunks)

    assert len(result) == 9
    assert provider.batch_sizes == [4, 4, 1]
    assert all(chunk.embedding_vector == [0.1, 0.2] for chunk in result)


def test_apply_embeddings_defaults_to_chunk_size_50_when_provider_has_no_chunk_size() -> None:
    """Providers without _chunk_size should use indexer default batch size of 50."""

    class NoChunkSizeProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.5, 0.5]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            payload = list(texts)
            self.batch_sizes.append(len(payload))
            return [[0.5, 0.5] for _ in payload]

        def get_dimension(self) -> int:
            return 2

    chunks = _make_embedding_test_chunks(60)
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(cache_dir=cache_dir, embedding_provider="openai")
    cache = CacheManager(CacheConfig(cache_dir))
    provider = NoChunkSizeProvider()
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=provider,
    )

    result = indexer._apply_embeddings(chunks)

    assert len(result) == 60
    assert provider.batch_sizes == [50, 10]
    assert all(chunk.embedding_vector == [0.5, 0.5] for chunk in result)


def test_apply_embeddings_submits_batches_serially_with_single_in_flight_call() -> None:
    """Embedding batch submission should stay serial (no concurrent in-flight embed_batch calls)."""

    class SerialGuardProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self._chunk_size = 2
            self.in_flight = 0
            self.max_in_flight = 0
            self.batch_sizes: list[int] = []

        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.3, 0.7]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            payload = list(texts)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                if self.in_flight != 1:
                    raise AssertionError("embed_batch called concurrently")
                return [[0.3, 0.7] for _ in payload]
            finally:
                self.batch_sizes.append(len(payload))
                self.in_flight -= 1

        def get_dimension(self) -> int:
            return 2

    chunks = _make_embedding_test_chunks(5)
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(cache_dir=cache_dir, embedding_provider="gemini")
    cache = CacheManager(CacheConfig(cache_dir))
    provider = SerialGuardProvider()
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=provider,
    )

    result = indexer._apply_embeddings(chunks)

    assert len(result) == 5
    assert provider.batch_sizes == [2, 2, 1]
    assert provider.max_in_flight == 1
    assert all(chunk.embedding_vector == [0.3, 0.7] for chunk in result)


def test_apply_embeddings_calls_progress_callback() -> None:
    """_apply_embeddings fires progress_callback after each chunk with correct counts."""

    class FakeProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self._chunk_size = 2

        def embed_text(self, text: str) -> list[float]:
            return [0.1, 0.2]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]] * len(list(texts))

        def get_dimension(self) -> int:
            return 2

    source = (
        "def alpha() -> None:\n    pass\n"
        "def beta() -> None:\n    pass\n"
        "def gamma() -> None:\n    pass\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(
            config=config,
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeProvider(),
        )
        result = indexer.index_repository(str(repo))

    assert result.indexed_symbols > 0

    # Now test _apply_embeddings directly with progress_callback.
    from gloggur.models import SymbolChunk

    def _make_chunk(name: str) -> SymbolChunk:
        return SymbolChunk(
            chunk_id=f"chunk-{name}",
            symbol_id=f"sym-{name}",
            chunk_part_index=1,
            chunk_part_total=1,
            text=f"def {name}():\n    pass",
            file_path="f.py",
            start_line=1,
            end_line=3,
        )

    chunks = [_make_chunk(f"fn{i}") for i in range(5)]
    progress_calls: list[tuple[int, int]] = []

    fake_provider = FakeProvider()
    cache_dir2 = tempfile.mkdtemp(prefix="gloggur-cache-")
    config2 = GloggurConfig(cache_dir=cache_dir2)
    cache2 = CacheManager(CacheConfig(cache_dir2))
    indexer2 = Indexer(
        config=config2,
        cache=cache2,
        parser_registry=ParserRegistry(),
        embedding_provider=fake_provider,
    )

    result_chunks = indexer2._apply_embeddings(
        chunks,
        progress_callback=lambda done, total: progress_calls.append((done, total)),
    )

    total = len(chunks)
    assert len(result_chunks) == total
    assert all(chunk.embedding_vector == [0.1, 0.2] for chunk in result_chunks)
    assert len(progress_calls) >= 1
    # All totals should be the total symbol count
    assert all(t == total for _, t in progress_calls)
    # Final call should be (total, total)
    assert progress_calls[-1] == (total, total)
    # Calls should be monotonically non-decreasing
    dones = [d for d, _ in progress_calls]
    assert dones == sorted(dones)


def test_apply_embeddings_no_progress_callback_works() -> None:
    """_apply_embeddings works normally when no progress_callback is provided."""

    class FakeProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self._chunk_size = 2

        def embed_text(self, text: str) -> list[float]:
            return [0.5, 0.5]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.5, 0.5]] * len(list(texts))

        def get_dimension(self) -> int:
            return 2

    from gloggur.models import SymbolChunk

    def _make_chunk(name: str) -> SymbolChunk:
        return SymbolChunk(
            chunk_id=f"chunk-{name}",
            symbol_id=f"sym-{name}",
            chunk_part_index=1,
            chunk_part_total=1,
            text=f"def {name}():\n    pass",
            file_path="f.py",
            start_line=1,
            end_line=2,
        )

    chunks = [_make_chunk(f"s{i}") for i in range(3)]
    fake_provider = FakeProvider()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(cache_dir=cache_dir)
    cache = CacheManager(CacheConfig(cache_dir))
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=fake_provider,
    )

    result = indexer._apply_embeddings(chunks)
    assert len(result) == 3
    assert all(chunk.embedding_vector == [0.5, 0.5] for chunk in result)


def test_apply_embeddings_fails_closed_on_provider_vector_count_mismatch() -> None:
    """Indexer must fail loud when a provider returns fewer vectors than requested."""

    class FakeProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self._chunk_size = 10

        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.1, 0.2]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            _ = texts
            return [[0.1, 0.2]]

        def get_dimension(self) -> int:
            return 2

    from gloggur.models import SymbolChunk

    chunks = [
        SymbolChunk(
            chunk_id=f"chunk-s{i}",
            symbol_id=f"sym-s{i}",
            chunk_part_index=1,
            chunk_part_total=1,
            text=f"def s{i}():\n    pass",
            file_path="f.py",
            start_line=1,
            end_line=2,
        )
        for i in range(2)
    ]

    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(cache_dir=cache_dir, embedding_provider="openai")
    cache = CacheManager(CacheConfig(cache_dir))
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=FakeProvider(),
    )

    with pytest.raises(EmbeddingProviderError, match="returned 1 vectors for 2 chunks"):
        indexer._apply_embeddings(chunks)


def test_index_file_with_outcome_classifies_embedding_provider_failures() -> None:
    """Single-file indexing should preserve provider failure taxonomy for CLI contracts."""

    class FakeProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.1, 0.2]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            _ = texts
            raise RuntimeError("provider vector API failed")

        def get_dimension(self) -> int:
            return 2

    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    repo_dir = tempfile.mkdtemp(prefix="gloggur-repo-")
    file_path = os.path.join(repo_dir, "sample.py")
    with open(file_path, "w", encoding="utf8") as handle:
        handle.write("def add(a, b):\n    return a + b\n")

    config = GloggurConfig(
        cache_dir=cache_dir,
        embedding_provider="openai",
    )
    cache = CacheManager(CacheConfig(cache_dir))
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=FakeProvider(),
    )

    outcome = indexer.index_file_with_outcome(file_path)

    assert outcome.status == "failed"
    assert outcome.reason == "embedding_provider_error"
    assert outcome.detail is not None
    assert "Embedding provider failure [openai]" in outcome.detail


def test_indexer_extract_timeout_fails_fast_and_reports_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repository indexing should fail fast on extract timeouts when partial failures are disabled."""
    module_name = _write_timeout_parser_module(tmp_path, sleep_seconds=0.6)
    monkeypatch.syspath_prepend(str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    a_path = repo / "a_ok.py"
    b_path = repo / "b_hang.py"
    c_path = repo / "c_ok.py"
    for path in (a_path, b_path, c_path):
        path.write_text("def sample() -> int:\n    return 1\n", encoding="utf8")

    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(
        cache_dir=cache_dir,
        extract_symbols_timeout_seconds=0.2,
        adapters={"parsers": {"python": f"{module_name}:create_parser"}},
    )
    cache = CacheManager(CacheConfig(cache_dir))
    indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())
    indexer._iter_source_files = lambda _root: [str(a_path), str(b_path), str(c_path)]  # type: ignore[method-assign]
    progress_updates: list[dict[str, object] | None] = []
    indexer._extract_progress_callback = lambda progress: progress_updates.append(progress)

    result = indexer.index_repository(str(repo))

    assert result.failed == 1
    assert result.failed_reasons == {EXTRACT_SYMBOLS_TIMEOUT_REASON: 1}
    assert result.indexed == 0
    assert result.files_considered == 2
    assert cache.count_files() == 0
    assert any(
        timing.path == str(b_path) and timing.reason == EXTRACT_SYMBOLS_TIMEOUT_REASON
        for timing in result.file_timings
    )
    assert any(
        isinstance(progress, dict)
        and progress["current_file"] == str(b_path)
        and progress["subphase"] == "prepare_file"
        for progress in progress_updates
    )


def test_indexer_extract_timeout_restarts_worker_when_partial_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial indexing should restart the extract worker and continue after a timed-out file."""
    module_name = _write_timeout_parser_module(tmp_path, sleep_seconds=0.6)
    monkeypatch.syspath_prepend(str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    a_path = repo / "a_ok.py"
    b_path = repo / "b_hang.py"
    c_path = repo / "c_ok.py"
    for path in (a_path, b_path, c_path):
        path.write_text("def sample() -> int:\n    return 1\n", encoding="utf8")

    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(
        cache_dir=cache_dir,
        extract_symbols_timeout_seconds=0.2,
        adapters={"parsers": {"python": f"{module_name}:create_parser"}},
    )
    cache = CacheManager(CacheConfig(cache_dir))
    indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())
    indexer._iter_source_files = lambda _root: [str(a_path), str(b_path), str(c_path)]  # type: ignore[method-assign]
    indexer._allow_partial_failures = True

    result = indexer.index_repository(str(repo))

    assert result.failed == 1
    assert result.failed_reasons == {EXTRACT_SYMBOLS_TIMEOUT_REASON: 1}
    assert result.indexed == 2
    assert cache.count_files() == 2
    assert cache.get_file_metadata(str(a_path)) is not None
    assert cache.get_file_metadata(str(c_path)) is not None
    assert cache.get_file_metadata(str(b_path)) is None


def test_index_file_with_details_reports_extract_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-file execution should classify extract timeouts with a stable reason code."""
    module_name = _write_timeout_parser_module(tmp_path, sleep_seconds=0.6)
    monkeypatch.syspath_prepend(str(tmp_path))
    file_path = tmp_path / "b_hang.py"
    file_path.write_text("def sample() -> int:\n    return 1\n", encoding="utf8")

    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    config = GloggurConfig(
        cache_dir=cache_dir,
        extract_symbols_timeout_seconds=0.2,
        adapters={"parsers": {"python": f"{module_name}:create_parser"}},
    )
    cache = CacheManager(CacheConfig(cache_dir))
    indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

    execution = indexer.index_file_with_details(str(file_path))

    assert execution.outcome.status == "failed"
    assert execution.outcome.reason == EXTRACT_SYMBOLS_TIMEOUT_REASON
    assert execution.timing is not None
    assert execution.timing.reason == EXTRACT_SYMBOLS_TIMEOUT_REASON


def test_indexer_resume_skips_prepare_for_stat_matched_extract_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume should skip prepare_file for files already checkpointed during extract."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.1, 0.2]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

        def get_dimension(self) -> int:
            return 2

    source = "def sample(value: int) -> int:\n    return value + 1\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"a.py": source, "b.py": source})
        ordered_paths = [str(repo / "a.py"), str(repo / "b.py")]
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        cache = CacheManager(CacheConfig(cache_dir))

        first_indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        first_indexer._iter_source_files = lambda _root: ordered_paths  # type: ignore[method-assign]
        original_build_edges = first_indexer._build_edges_with_watchdog
        edge_calls = 0

        def _halt_on_second_edge(**kwargs: object) -> tuple[list[EdgeRecord] | None, int, bool, str | None]:
            nonlocal edge_calls
            edge_calls += 1
            if edge_calls == 2:
                return None, 0, True, "simulated timeout"
            return original_build_edges(**kwargs)

        monkeypatch.setattr(first_indexer, "_build_edges_with_watchdog", _halt_on_second_edge)

        first_result = first_indexer.index_repository(str(repo))

        assert first_result.failed == 1
        assert cache.get_build_file_checkpoint(ordered_paths[0]) is not None
        assert cache.get_build_file_checkpoint(ordered_paths[1]) is None
        assert cache.get_build_checkpoint_stats()["pending_embed_files"] == 1

        second_indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        second_indexer._iter_source_files = lambda _root: ordered_paths  # type: ignore[method-assign]
        original_prepare = second_indexer._prepare_file_for_index_with_watchdog
        prepare_calls: list[str] = []

        def _record_prepare(**kwargs: object):
            prepare_calls.append(str(kwargs["path"]))
            return original_prepare(**kwargs)

        monkeypatch.setattr(second_indexer, "_prepare_file_for_index_with_watchdog", _record_prepare)

        second_result = second_indexer.index_repository(str(repo))

        assert second_result.failed == 0
        assert prepare_calls == [ordered_paths[1]]
        assert cache.count_files() == 2
        assert cache.get_build_checkpoint_stats() == {
            "extract_completed_files": 2,
            "embedded_completed_files": 2,
            "pending_embed_files": 0,
        }


def test_indexer_resume_embeds_extract_checkpoint_without_reextracting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embed resume should consume extract checkpoints without rerunning prepare_file."""

    class FailingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.1, 0.2]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            _ = texts
            raise RuntimeError("embed failed")

        def get_dimension(self) -> int:
            return 2

    class RecordingProvider(EmbeddingProvider):
        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.3, 0.4]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.3, 0.4] for _ in texts]

        def get_dimension(self) -> int:
            return 2

    source = "def sample(value: int) -> int:\n    return value + 1\n"
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        file_path = str(repo / "sample.py")
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        cache = CacheManager(CacheConfig(cache_dir))

        first_indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=FailingProvider(),
        )
        first_result = first_indexer.index_repository(str(repo))

        assert first_result.failed == 1
        checkpoint = cache.get_build_file_checkpoint(file_path)
        assert checkpoint is not None
        assert checkpoint.state == BUILD_FILE_CHECKPOINT_STATE_EXTRACT_COMPLETE
        assert cache.get_file_metadata(file_path) is not None
        assert all(
            chunk.embedding_vector is None for chunk in cache.list_chunks_for_file(file_path)
        )

        second_indexer = Indexer(
            config=GloggurConfig(cache_dir=cache_dir),
            cache=cache,
            parser_registry=ParserRegistry(),
            embedding_provider=RecordingProvider(),
        )
        original_prepare = second_indexer._prepare_file_for_index_with_watchdog
        prepare_calls: list[str] = []

        def _record_prepare(**kwargs: object):
            prepare_calls.append(str(kwargs["path"]))
            return original_prepare(**kwargs)

        monkeypatch.setattr(second_indexer, "_prepare_file_for_index_with_watchdog", _record_prepare)

        second_result = second_indexer.index_repository(str(repo))

        assert second_result.failed == 0
        assert prepare_calls == []
        resumed_checkpoint = cache.get_build_file_checkpoint(file_path)
        assert resumed_checkpoint is not None
        assert resumed_checkpoint.state == BUILD_FILE_CHECKPOINT_STATE_EMBEDDED_COMPLETE
        assert cache.get_file_metadata(file_path) is not None


def test_indexer_builds_repo_symbol_catalog_once_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repository indexing should fetch the cached symbol catalog once per run."""
    source = (
        "def alpha(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def beta(value: int) -> int:\n"
        "    return alpha(value)\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"a.py": source, "b.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())
        list_symbols_calls = 0
        original_list_symbols = cache.list_symbols

        def _counted_list_symbols() -> list:
            nonlocal list_symbols_calls
            list_symbols_calls += 1
            return original_list_symbols()

        monkeypatch.setattr(cache, "list_symbols", _counted_list_symbols)

        result = indexer.index_repository(str(repo))

        assert result.failed == 0
        assert list_symbols_calls == 1
