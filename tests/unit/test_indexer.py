from __future__ import annotations

import tempfile

from gloggur.config import GloggurConfig
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.indexer import Indexer
from gloggur.parsers.registry import ParserRegistry
from scripts.validation.fixtures import TestFixtures


def test_indexer_indexes_repo_and_sets_metadata() -> None:
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        config = GloggurConfig(cache_dir=cache_dir)
        cache = CacheManager(CacheConfig(cache_dir))
        indexer = Indexer(config=config, cache=cache, parser_registry=ParserRegistry())

        result = indexer.index_repository(str(repo))

        assert result.indexed_files == 1
        assert result.indexed_symbols > 0
        assert result.skipped_files == 0
        metadata = cache.get_index_metadata()
        assert metadata is not None
        assert metadata.indexed_files == 1
        assert metadata.total_symbols == len(cache.list_symbols())


def test_indexer_skips_unchanged_files() -> None:
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
