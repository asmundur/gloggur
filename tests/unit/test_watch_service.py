from __future__ import annotations

import signal
import tempfile
from pathlib import Path

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.watch.service import BatchResult, WatchService, load_watch_state
from scripts.verification.fixtures import TestFixtures


def _write_fallback_marker(cache_dir: str) -> None:
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _build_watch_service(
    repo: Path,
    cache_dir: str,
    watch_state_file: str | None = None,
) -> tuple[WatchService, CacheManager]:
    config = GloggurConfig(
        cache_dir=cache_dir,
        watch_path=str(repo),
        watch_state_file=watch_state_file or str(Path(cache_dir) / "watch_state.json"),
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
    return service, cache


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
        service, cache = _build_watch_service(repo, cache_dir)

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


def test_watch_service_unchanged_file_does_not_reindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        sample = repo / "sample.py"
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        service, _cache = _build_watch_service(repo, cache_dir)

        # Seed with one indexed file.
        assert service.indexer.index_file(str(sample)) is not None

        called = {"index_file": 0}

        def fail_index_file(_path: str) -> int:
            called["index_file"] += 1
            raise AssertionError("index_file should not be called for unchanged content")

        monkeypatch.setattr(service.indexer, "index_file", fail_index_file)
        unchanged = service.process_batch(
            changes={(2, str(sample))},
            watch_root=str(repo),
        )

        assert unchanged.skipped_files == 1
        assert unchanged.indexed_files == 0
        assert called["index_file"] == 0


def test_watch_service_watch_changes_requires_watchfiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        service, _cache = _build_watch_service(repo, cache_dir)

        import builtins

        original_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "watchfiles":
                raise ImportError("watchfiles missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(RuntimeError, match="watchfiles is required for watch mode"):
            next(service._watch_changes(str(repo)))


def test_watch_service_run_forever_updates_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        sample = repo / "sample.py"
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        state_file = str(Path(cache_dir) / "watch_state.json")
        service, _cache = _build_watch_service(repo, cache_dir, watch_state_file=state_file)

        def fake_watch_changes(_watch_target: str):  # type: ignore[no-untyped-def]
            yield {(2, str(sample))}

        def fake_process_batch(
            changes,  # noqa: ANN001
            watch_root: str,
            watch_file: str | None = None,
        ) -> BatchResult:
            _ = changes
            assert watch_root == str(repo)
            assert watch_file is None
            return BatchResult(changed_files=1, indexed_files=1, indexed_symbols=2, skipped_files=0)

        def fake_signal(_sig: signal.Signals, _handler):  # type: ignore[no-untyped-def]
            return signal.SIG_DFL

        monkeypatch.setattr(service, "_watch_changes", fake_watch_changes)
        monkeypatch.setattr(service, "process_batch", fake_process_batch)
        monkeypatch.setattr("gloggur.watch.service.signal.signal", fake_signal)

        payload = service.run_forever(str(repo))

        assert payload["indexed_files"] == 1
        assert payload["indexed_symbols"] == 2
        assert payload["error_count"] == 0
        state_payload = load_watch_state(state_file)
        assert state_payload["running"] is False
        assert state_payload["status"] == "stopped"
        assert state_payload["indexed_files"] == 1
        assert state_payload["indexed_symbols"] == 2
        assert state_payload.get("last_batch", {}).get("indexed_files") == 1
