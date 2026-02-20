from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.watch.service import WatchService
from scripts.verification.fixtures import TestFixtures


def _write_fallback_marker(cache_dir: str) -> None:
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def test_watch_service_process_batch_changed_unchanged_and_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        sample = repo / "sample.py"
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        config = GloggurConfig(
            cache_dir=cache_dir,
            watch_path=str(repo),
            local_embedding_model="local",
        )
        cache = CacheManager(CacheConfig(cache_dir))
        vector_store = VectorStore(VectorStoreConfig(cache_dir))
        embedding = LocalEmbeddingProvider("local", cache_dir=cache_dir)
        service = WatchService(
            config=config,
            embedding_provider=embedding,
            cache=cache,
            vector_store=vector_store,
        )

        # Seed with one indexed file.
        service.indexer.index_file(str(sample))

        unchanged = service.process_batch(
            changes={(2, str(sample))},
            watch_root=str(repo),
        )
        assert unchanged.skipped_files == 1
        assert unchanged.indexed_files == 0

        sample.write_text(
            source + "\n\ndef extra(value: int) -> int:\n    return value\n",
            encoding="utf8",
        )
        changed = service.process_batch(
            changes={(2, str(sample))},
            watch_root=str(repo),
        )
        assert changed.indexed_files == 1
        assert changed.indexed_symbols > 0

        sample.unlink()
        deleted = service.process_batch(
            changes={(3, str(sample))},
            watch_root=str(repo),
        )
        assert deleted.deleted_files == 1
        assert cache.get_file_metadata(str(sample)) is None
        assert cache.count_files() == 0
