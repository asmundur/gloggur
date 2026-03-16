from __future__ import annotations

import errno
import json
import signal
import tempfile
import threading
import time
from pathlib import Path

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings.test_provider import DeterministicTestEmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.watch.service import (
    BatchResult,
    WatchService,
    _atomic_write_json_file,
    load_watch_state,
)
from scripts.verification.fixtures import TestFixtures


def _write_fallback_marker(cache_dir: str) -> None:
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _build_watch_service(
    repo: Path,
    cache_dir: str,
    watch_state_file: str | None = None,
    include_minified_js: bool = False,
) -> tuple[WatchService, CacheManager]:
    config = GloggurConfig(
        cache_dir=cache_dir,
        watch_path=str(repo),
        watch_state_file=watch_state_file or str(Path(cache_dir) / "watch_state.json"),
        local_embedding_model="local",
        include_minified_js=include_minified_js,
    )
    cache = CacheManager(CacheConfig(cache_dir))
    vector_store = VectorStore(VectorStoreConfig(cache_dir))
    embedding = DeterministicTestEmbeddingProvider()
    service = WatchService(
        config=config,
        embedding_provider=embedding,
        cache=cache,
        vector_store=vector_store,
    )
    return service, cache


def test_atomic_write_json_file_uses_unique_temp_paths_under_concurrent_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "watch_state.json"
    original_replace = Path.replace
    inflight_temp_paths: set[str] = set()
    replace_lock = threading.Lock()

    def _replace_with_collision_guard(self: Path, destination: Path) -> Path:
        key = str(self)
        with replace_lock:
            if key in inflight_temp_paths:
                raise FileNotFoundError(
                    errno.ENOENT,
                    "No such file or directory",
                    str(self),
                    str(destination),
                )
            inflight_temp_paths.add(key)
        try:
            time.sleep(0.02)
            return original_replace(self, destination)
        finally:
            with replace_lock:
                inflight_temp_paths.discard(key)

    monkeypatch.setattr(Path, "replace", _replace_with_collision_guard)

    barrier = threading.Barrier(2)
    failures: list[Exception] = []

    def _writer(value: int) -> None:
        barrier.wait(timeout=2.0)
        try:
            _atomic_write_json_file(str(target), {"value": value})
        except Exception as exc:  # pragma: no cover - collected for assertion
            failures.append(exc)

    left = threading.Thread(target=_writer, args=(1,))
    right = threading.Thread(target=_writer, args=(2,))
    left.start()
    right.start()
    left.join(timeout=2.0)
    right.join(timeout=2.0)

    assert failures == []
    payload = json.loads(target.read_text(encoding="utf8"))
    assert payload["value"] in {1, 2}


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


def test_watch_service_skips_minified_js_by_default_and_can_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch filtering should skip `.min.js` by default and index when opted in."""
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {"vendor/jquery.min.js": "function minifiedVendor(){return 1};"}
        )
        minified = repo / "vendor" / "jquery.min.js"

        skip_cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(skip_cache_dir)
        skip_service, _skip_cache = _build_watch_service(repo, skip_cache_dir)
        skipped = skip_service.process_batch(
            changes={(2, str(minified))},
            watch_root=str(repo),
        )

        assert skipped.files_considered == 0
        assert skipped.indexed_files == 0
        assert skipped.skipped_files == 0
        assert skipped.error_count == 0

        include_cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(include_cache_dir)
        include_service, _include_cache = _build_watch_service(
            repo,
            include_cache_dir,
            include_minified_js=True,
        )
        included = include_service.process_batch(
            changes={(2, str(minified))},
            watch_root=str(repo),
        )

        assert included.files_considered == 1
        assert included.indexed_files == 1
        assert included.error_count == 0


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


def test_watch_service_run_forever_keeps_last_failed_batch_on_noop_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-op watch events should not clear the last failing batch failure contract."""
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
            # Simulate a follow-up watch event outside indexed extensions.
            yield {(2, str(repo / ".gloggur" / "index" / "symbols.db"))}

        batches = [
            BatchResult(
                changed_files=1,
                indexed_files=1,
                indexed_symbols=1,
                error_count=1,
                failed_reasons={"vector_metadata_mismatch": 1},
                failed_samples=[
                    "vector/cache mismatch (missing_vectors=0, stale_vectors=1); "
                    "stale_example=ghost::watch::symbol"
                ],
                last_error="vector/cache mismatch (missing_vectors=0, stale_vectors=1)",
            ),
            BatchResult(),
        ]

        def fake_process_batch(
            changes,  # noqa: ANN001
            watch_root: str,
            watch_file: str | None = None,
        ) -> BatchResult:
            _ = changes
            assert watch_root == str(repo)
            assert watch_file is None
            return batches.pop(0)

        def fake_signal(_sig: signal.Signals, _handler):  # type: ignore[no-untyped-def]
            return signal.SIG_DFL

        monkeypatch.setattr(service, "_watch_changes", fake_watch_changes)
        monkeypatch.setattr(service, "process_batch", fake_process_batch)
        monkeypatch.setattr("gloggur.watch.service.signal.signal", fake_signal)

        service.run_forever(str(repo))

        state_payload = load_watch_state(state_file)
        last_batch = state_payload.get("last_batch", {})
        assert isinstance(last_batch, dict)
        failure_codes = last_batch.get("failure_codes", [])
        assert isinstance(failure_codes, list)
        assert "vector_metadata_mismatch" in failure_codes
        assert (
            state_payload.get("last_error")
            == "vector/cache mismatch (missing_vectors=0, stale_vectors=1)"
        )


def test_noop_last_batch_contract_backfills_partial_existing_failure_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        state_file = str(Path(cache_dir) / "watch_state.json")
        service, _cache = _build_watch_service(repo, cache_dir, watch_state_file=state_file)

        Path(state_file).write_text(
            json.dumps(
                {
                    "last_batch": {
                        "failure_codes": ["vector_metadata_mismatch"],
                    }
                }
            ),
            encoding="utf8",
        )
        service._total_failed_reasons = {"vector_metadata_mismatch": 1}
        service._total_errors = 1

        payload = service._noop_last_batch_contract()

        assert isinstance(payload, dict)
        assert payload["failure_codes"] == ["vector_metadata_mismatch"]
        assert payload["failed_reasons"] == {"vector_metadata_mismatch": 1}
        assert payload["failed"] == 1
        assert payload["error_count"] == 1
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "vector_metadata_mismatch" in guidance
        assert isinstance(guidance["vector_metadata_mismatch"], list)
        assert guidance["vector_metadata_mismatch"]


def test_watch_service_surfaces_stale_cleanup_failure_contract_and_keeps_metadata_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch batches should fail closed with deterministic stale-cleanup codes/remediation."""
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        sample = repo / "sample.py"
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        service, cache = _build_watch_service(repo, cache_dir)

        # Seed index metadata so we can verify failure keeps metadata invalid.
        service.indexer.index_repository(str(repo))
        assert cache.get_index_metadata() is not None

        sample.write_text(
            source + "\n\ndef extra(value: int) -> int:\n    return value + 1\n",
            encoding="utf8",
        )

        monkeypatch.setattr(
            service.indexer,
            "prune_missing_file_entries",
            lambda: {
                "files_removed": 0,
                "symbols_removed": 0,
                "failed": 1,
                "failed_reasons": {"stale_cleanup_error": 1},
                "failed_samples": ["/tmp/stale.py: OSError: simulated stale cleanup failure"],
            },
        )
        monkeypatch.setattr(
            service.indexer,
            "validate_vector_metadata_consistency",
            lambda: {"failed": 0, "failed_reasons": {}, "failed_samples": []},
        )

        batch = service.process_batch(changes=[(2, str(sample))], watch_root=str(repo))

        assert batch.error_count == 1
        assert batch.failed_reasons == {"stale_cleanup_error": 1}
        payload = batch.as_dict()
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["stale_cleanup_error"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "stale_cleanup_error" in guidance
        assert isinstance(guidance["stale_cleanup_error"], list)
        assert guidance["stale_cleanup_error"]
        # Fail closed: metadata remains invalid so status/search can require reindex.
        assert cache.get_index_metadata() is None


def test_watch_service_synthesizes_incremental_inconsistent_failure_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch should fail closed when downstream failure payloads omit reason codes."""
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        sample = repo / "sample.py"
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        service, cache = _build_watch_service(repo, cache_dir)

        service.indexer.index_repository(str(repo))
        assert cache.get_index_metadata() is not None

        sample.write_text(
            source + "\n\ndef extra_again(value: int) -> int:\n    return value + 2\n",
            encoding="utf8",
        )

        # Simulate a contract-drift failure payload that reports failure count but no reason codes.
        monkeypatch.setattr(
            service.indexer,
            "prune_missing_file_entries",
            lambda: {
                "files_removed": 0,
                "symbols_removed": 0,
                "failed": 1,
                "failed_reasons": {},
                "failed_samples": ["/tmp/unknown.py: RuntimeError: inconsistent incremental state"],
            },
        )
        monkeypatch.setattr(
            service.indexer,
            "validate_vector_metadata_consistency",
            lambda: {"failed": 0, "failed_reasons": {}, "failed_samples": []},
        )

        batch = service.process_batch(changes=[(2, str(sample))], watch_root=str(repo))

        assert batch.error_count == 1
        assert batch.failed_reasons == {"watch_incremental_inconsistent": 1}
        payload = batch.as_dict()
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["watch_incremental_inconsistent"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "watch_incremental_inconsistent" in guidance
        assert isinstance(guidance["watch_incremental_inconsistent"], list)
        assert guidance["watch_incremental_inconsistent"]
        assert cache.get_index_metadata() is None
