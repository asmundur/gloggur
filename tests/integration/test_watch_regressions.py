from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner
import pytest

from gloggur.cli.main import cli
from gloggur.config import GloggurConfig
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.watch.service import WatchService
from scripts.verification.fixtures import TestFixtures


def _parse_json_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {output!r}")
    return json.loads(output[start:])


def _invoke_json(runner: CliRunner, args: list[str], env: dict[str, str]) -> dict[str, object]:
    result = runner.invoke(cli, args, env=env)
    assert result.exit_code == 0, result.output
    return _parse_json_output(result.output)


def _assert_search_ready(payload: dict[str, object]) -> None:
    metadata = payload.get("metadata", {})
    assert isinstance(metadata, dict)
    assert metadata.get("needs_reindex") is not True


def _write_fallback_marker(cache_dir: str) -> None:
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _build_watch_service(repo: Path, cache_dir: str) -> WatchService:
    config = GloggurConfig(
        cache_dir=cache_dir,
        watch_path=str(repo),
        local_embedding_model="local",
    )
    cache = CacheManager(CacheConfig(cache_dir))
    vector_store = VectorStore(VectorStoreConfig(cache_dir))
    embedding = LocalEmbeddingProvider("local", cache_dir=cache_dir)
    return WatchService(
        config=config,
        embedding_provider=embedding,
        cache=cache,
        vector_store=vector_store,
    )


def test_watch_delete_removes_stale_search_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    runner = CliRunner()
    source = (
        "def legacy_symbol() -> str:\n"
        '    """legacy sentinel phrase for watch delete regression"""\n'
        '    return "legacy"\n'
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"legacy.py": source})
        target = repo / "legacy.py"
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir, "GLOGGUR_LOCAL_MODEL": "local"}

        _invoke_json(runner, ["index", str(repo), "--json"], env)
        before = _invoke_json(
            runner,
            [
                "search",
                "legacy sentinel phrase for watch delete regression",
                "--json",
                "--top-k",
                "10",
            ],
            env,
        )
        _assert_search_ready(before)
        before_results = before.get("results", [])
        assert isinstance(before_results, list) and before_results
        assert any(item.get("file") == str(target) for item in before_results)

        service = _build_watch_service(repo, cache_dir)
        target.unlink()
        batch = service.process_batch(changes=[(3, str(target))], watch_root=str(repo))
        assert batch.deleted_files == 1

        after = _invoke_json(
            runner,
            [
                "search",
                "legacy sentinel phrase for watch delete regression",
                "--json",
                "--top-k",
                "10",
            ],
            env,
        )
        _assert_search_ready(after)
        after_results = after.get("results", [])
        assert isinstance(after_results, list)
        assert all(item.get("file") != str(target) for item in after_results)


def test_watch_rename_replaces_search_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    runner = CliRunner()
    old_source = (
        "def old_symbol() -> str:\n"
        '    """legacy rename phrase unique old"""\n'
        '    return "old"\n'
    )
    new_source = (
        "def new_symbol() -> str:\n"
        '    """fresh rename phrase unique new"""\n'
        '    return "new"\n'
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"module.py": old_source})
        old_path = repo / "module.py"
        new_path = repo / "module_renamed.py"
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir, "GLOGGUR_LOCAL_MODEL": "local"}

        _invoke_json(runner, ["index", str(repo), "--json"], env)
        before = _invoke_json(
            runner,
            ["search", "legacy rename phrase unique old", "--json", "--top-k", "10"],
            env,
        )
        _assert_search_ready(before)
        before_results = before.get("results", [])
        assert isinstance(before_results, list) and before_results
        assert any(item.get("file") == str(old_path) for item in before_results)

        old_path.unlink()
        new_path.write_text(new_source, encoding="utf8")
        service = _build_watch_service(repo, cache_dir)
        batch = service.process_batch(
            changes=[(3, str(old_path)), (2, str(new_path))],
            watch_root=str(repo),
        )
        assert batch.deleted_files == 1
        assert batch.changed_files == 1

        old_query = _invoke_json(
            runner,
            ["search", "legacy rename phrase unique old", "--json", "--top-k", "10"],
            env,
        )
        _assert_search_ready(old_query)
        old_results = old_query.get("results", [])
        assert isinstance(old_results, list)
        assert all(item.get("file") != str(old_path) for item in old_results)

        new_query = _invoke_json(
            runner,
            ["search", "fresh rename phrase unique new", "--json", "--top-k", "10"],
            env,
        )
        _assert_search_ready(new_query)
        new_results = new_query.get("results", [])
        assert isinstance(new_results, list) and new_results
        assert any(item.get("file") == str(new_path) for item in new_results)


def test_watch_rename_change_only_batch_prunes_ghost_old_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch should not report success while stale old-path rows survive a rename change-only batch."""
    monkeypatch.setattr(VectorStore, "_check_faiss", staticmethod(lambda: False))
    runner = CliRunner()
    old_source = (
        "def old_symbol() -> str:\n"
        '    """watch rename change-only old token"""\n'
        '    return "old"\n'
    )
    new_source = (
        "def new_symbol() -> str:\n"
        '    """watch rename change-only new token"""\n'
        '    return "new"\n'
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"module.py": old_source})
        old_path = repo / "module.py"
        new_path = repo / "module_renamed.py"
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir, "GLOGGUR_LOCAL_MODEL": "local"}

        _invoke_json(runner, ["index", str(repo), "--json"], env)
        before = _invoke_json(
            runner,
            ["search", "watch rename change-only old token", "--json", "--top-k", "10"],
            env,
        )
        _assert_search_ready(before)
        before_results = before.get("results", [])
        assert isinstance(before_results, list) and before_results
        assert any(item.get("file") == str(old_path) for item in before_results)

        old_path.rename(new_path)
        new_path.write_text(new_source, encoding="utf8")
        service = _build_watch_service(repo, cache_dir)
        batch = service.process_batch(
            changes=[(2, str(new_path))],
            watch_root=str(repo),
        )
        assert batch.changed_files == 1
        assert batch.deleted_files == 1
        assert batch.error_count == 0

        old_query = _invoke_json(
            runner,
            ["search", "watch rename change-only old token", "--json", "--top-k", "10"],
            env,
        )
        _assert_search_ready(old_query)
        old_results = old_query.get("results", [])
        assert isinstance(old_results, list)
        assert all(item.get("file") != str(old_path) for item in old_results)

        new_query = _invoke_json(
            runner,
            ["search", "watch rename change-only new token", "--json", "--top-k", "10"],
            env,
        )
        _assert_search_ready(new_query)
        new_results = new_query.get("results", [])
        assert isinstance(new_results, list) and new_results
        assert any(item.get("file") == str(new_path) for item in new_results)
