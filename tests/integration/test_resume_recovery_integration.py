from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from gloggur.cli import main as cli_main
from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.embedding_ledger import EmbeddingLedger
from gloggur.indexer.indexer import Indexer
from gloggur.parsers.registry import ParserRegistry
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from scripts.verification.fixtures import TestFixtures


def _parse_json_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {output!r}")
    return json.loads(output[start:])


def test_index_auto_resumes_interrupted_staged_build_without_reembedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "one.py": "def one(value: int) -> int:\n    return value + 1\n",
                "two.py": "def two(value: int) -> int:\n    return value + 2\n",
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")

        class FailAfterFirstBatchProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 1
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.1, 0.2]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                if len(self.calls) > 1:
                    raise RuntimeError("timeout")
                return [[0.1, 0.2] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        class RecordingProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 1
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.3, 0.4]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                return [[0.3, 0.4] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        first_provider = FailAfterFirstBatchProvider()
        second_provider = RecordingProvider()
        providers = [first_provider, second_provider]

        monkeypatch.setattr(
            cli_main,
            "_create_embedding_provider_for_command",
            lambda *_args, **_kwargs: providers.pop(0),
        )
        monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
        monkeypatch.setattr(
            cli_main,
            "_run_symbol_index",
            lambda **_kwargs: {
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
                "duration_ms": 0,
            },
        )

        env = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "gemini",
        }

        first = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert first.exit_code == 1, first.output

        status = runner.invoke(cli_main.cli, ["status", "--json"], env=env)
        assert status.exit_code == 0, status.output
        status_payload = _parse_json_output(status.output)
        assert status_payload["interrupted_resume_available"] is True
        resume_candidate = status_payload["resume_candidate"]
        assert isinstance(resume_candidate, dict)
        assert resume_candidate["source"] == "manifest"
        assert resume_candidate["compatibility"] == {"compatible": True, "reason_codes": []}
        assert resume_candidate["counts"]["extract_completed_files"] == 2
        assert resume_candidate["counts"]["embedded_completed_files"] == 1
        assert resume_candidate["counts"]["pending_embed_files"] == 1

        second = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert second.exit_code == 0, second.output
        second_payload = _parse_json_output(second.output)
        assert second_payload["resumed_build"] is True
        assert second_payload["resume_source"] == "manifest"
        assert len(second_provider.calls) == 1
        assert len(second_provider.calls[0]) == 1

        final_cache = CacheManager(CacheConfig(cache_dir))
        assert final_cache.get_build_checkpoint_stats() == {
            "extract_completed_files": 0,
            "embedded_completed_files": 0,
            "pending_embed_files": 0,
        }


def test_index_resume_rebuilds_missing_stage_vector_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "one.py": "def one(value: int) -> int:\n    return value + 1\n",
                "two.py": "def two(value: int) -> int:\n    return value + 2\n",
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")

        class FailAfterFirstBatchProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 1
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.1, 0.2]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                if len(self.calls) > 1:
                    raise RuntimeError("timeout")
                return [[0.1, 0.2] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        class RecordingProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 1
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.3, 0.4]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                return [[0.3, 0.4] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        first_provider = FailAfterFirstBatchProvider()
        second_provider = RecordingProvider()
        providers = [first_provider, second_provider]

        monkeypatch.setattr(
            cli_main,
            "_create_embedding_provider_for_command",
            lambda *_args, **_kwargs: providers.pop(0),
        )
        monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
        monkeypatch.setattr(
            cli_main,
            "_run_symbol_index",
            lambda **_kwargs: {
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
                "duration_ms": 0,
            },
        )

        env = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "gemini",
        }

        first = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert first.exit_code == 1, first.output

        cache = CacheManager(CacheConfig(cache_dir))
        build_state = cache.get_build_state()
        assert isinstance(build_state, dict)
        build_id = str(build_state["build_id"])
        stage_dir = Path(cache.build_cache_dir(build_id))
        for artifact_name in ("vectors.index", "vectors.json", "vectors.npy"):
            artifact_path = stage_dir / artifact_name
            if artifact_path.exists():
                artifact_path.unlink()

        second = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert second.exit_code == 0, second.output
        second_payload = _parse_json_output(second.output)
        assert second_payload["resumed_build"] is True
        assert second_payload["resume_source"] == "manifest"
        assert len(second_provider.calls) == 1
        assert len(second_provider.calls[0]) == 1

        final_cache = CacheManager(CacheConfig(cache_dir))
        final_vector_store = VectorStore(VectorStoreConfig(cache_dir))
        assert len(final_vector_store.list_symbol_ids()) == final_cache.count_embedded_chunks()


def test_index_auto_resumes_legacy_cwd_bound_manifest_when_target_path_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "one.py": "def one(value: int) -> int:\n    return value + 1\n",
                "two.py": "def two(value: int) -> int:\n    return value + 2\n",
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")

        class FailAfterFirstBatchProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 1

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.1, 0.2]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                _ = list(texts)
                raise RuntimeError("timeout")

            def get_dimension(self) -> int:
                return 2

        class RecordingProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 1
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.3, 0.4]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                return [[0.3, 0.4] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        first_provider = FailAfterFirstBatchProvider()
        second_provider = RecordingProvider()
        providers = [first_provider, second_provider]

        monkeypatch.setattr(
            cli_main,
            "_create_embedding_provider_for_command",
            lambda *_args, **_kwargs: providers.pop(0),
        )
        monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
        monkeypatch.setattr(
            cli_main,
            "_run_symbol_index",
            lambda **_kwargs: {
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
                "duration_ms": 0,
            },
        )

        env = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "gemini",
        }

        first = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert first.exit_code == 1, first.output

        cache = CacheManager(CacheConfig(cache_dir))
        build_state = cache.get_build_state()
        assert isinstance(build_state, dict)
        build_id = str(build_state["build_id"])
        manifest_path = cache.build_resume_manifest_path(build_id)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest["workspace_path_hash"] = cli_main._current_workspace_path_hash(str(Path.cwd()))
        Path(manifest_path).write_text(json.dumps(manifest), encoding="utf-8")

        second = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert second.exit_code == 0, second.output
        second_payload = _parse_json_output(second.output)
        assert second_payload["resumed_build"] is True
        assert second_payload["resume_source"] == "manifest"
        assert second_provider.calls != []


def test_index_adopts_legacy_interrupted_build_without_reembedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "sample.py": "def sample(value: int) -> int:\n    return value + 1\n",
            }
        )
        legacy_cache_root = tempfile.mkdtemp(prefix="legacy-gloggur-cache-")
        legacy_active_cache = CacheManager(CacheConfig(legacy_cache_root))
        build_id = "legacy-build"
        stage_dir = legacy_active_cache.prepare_staged_build(build_id)
        stage_cache = CacheManager(CacheConfig(stage_dir))
        stage_config = GloggurConfig(cache_dir=stage_dir, embedding_provider="gemini")
        vector_store = VectorStore(VectorStoreConfig(stage_dir), load_existing=False)

        class LegacyProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 10

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.1, 0.2]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2] for _ in texts]

            def get_dimension(self) -> int:
                return 2

        legacy_indexer = Indexer(
            config=stage_config,
            cache=stage_cache,
            parser_registry=ParserRegistry(),
            embedding_provider=LegacyProvider(),
            vector_store=vector_store,
            embedding_ledger=EmbeddingLedger(legacy_cache_root),
        )
        legacy_result = legacy_indexer.index_repository(str(repo))
        assert legacy_result.failed == 0
        stage_cache.delete_index_metadata()

        class RecordingProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 10
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.3, 0.4]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                return [[0.3, 0.4] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        provider = RecordingProvider()
        monkeypatch.setattr(
            cli_main,
            "_create_embedding_provider_for_command",
            lambda *_args, **_kwargs: provider,
        )
        monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
        monkeypatch.setattr(
            cli_main,
            "_run_symbol_index",
            lambda **_kwargs: {
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
                "duration_ms": 0,
            },
        )

        current_cache_root = tempfile.mkdtemp(prefix="gloggur-cache-")
        env = {
            "GLOGGUR_CACHE_DIR": current_cache_root,
            "GLOGGUR_EMBEDDING_PROVIDER": "gemini",
        }

        result = runner.invoke(
            cli_main.cli,
            ["index", str(repo), "--json", "--adopt-interrupted-build", stage_dir],
            env=env,
        )

        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        assert payload["resumed_build"] is True
        assert payload["resume_source"] == "legacy_adopted"
        assert provider.calls == []


def test_index_auto_resumes_orphaned_staged_build_without_persisted_build_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "one.py": "def one(value: int) -> int:\n    return value + 1\n",
                "two.py": "def two(value: int) -> int:\n    return value + 2\n",
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")

        class FailAfterFirstBatchProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 1
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.1, 0.2]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                if len(self.calls) > 1:
                    raise RuntimeError("timeout")
                return [[0.1, 0.2] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        class RecordingProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 1
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.3, 0.4]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                return [[0.3, 0.4] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        first_provider = FailAfterFirstBatchProvider()
        second_provider = RecordingProvider()
        providers = [first_provider, second_provider]

        monkeypatch.setattr(
            cli_main,
            "_create_embedding_provider_for_command",
            lambda *_args, **_kwargs: providers.pop(0),
        )
        monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
        monkeypatch.setattr(
            cli_main,
            "_run_symbol_index",
            lambda **_kwargs: {
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
                "duration_ms": 0,
            },
        )

        env = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "gemini",
        }

        first = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert first.exit_code == 1, first.output

        cache = CacheManager(CacheConfig(cache_dir))
        build_state = cache.get_build_state()
        assert isinstance(build_state, dict)
        original_build_id = str(build_state["build_id"])
        cache.clear_build_state()

        write_build_ids: list[str] = []
        original_write_cache_build_state = cli_main._write_cache_build_state

        def _record_build_state(cache_obj: object, **kwargs: object) -> dict[str, object]:
            build_id = kwargs.get("build_id")
            if build_id is not None:
                write_build_ids.append(str(build_id))
            return original_write_cache_build_state(cache_obj, **kwargs)

        monkeypatch.setattr(cli_main, "_write_cache_build_state", _record_build_state)

        second = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert second.exit_code == 0, second.output
        second_payload = _parse_json_output(second.output)
        assert second_payload["resumed_build"] is True
        assert second_payload["resume_source"] == "manifest"
        assert len(second_provider.calls) == 1
        assert all(build_id == original_build_id for build_id in write_build_ids)


def test_index_blocks_manifestless_staged_build_until_discard_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "sample.py": "def sample(value: int) -> int:\n    return value + 1\n",
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        cache = CacheManager(CacheConfig(cache_dir))
        cache.prepare_staged_build("legacy-build")

        class RecordingProvider(EmbeddingProvider):
            def __init__(self) -> None:
                self._chunk_size = 10
                self.calls: list[list[str]] = []

            def embed_text(self, text: str) -> list[float]:
                _ = text
                return [0.3, 0.4]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                payload = list(texts)
                self.calls.append(payload)
                return [[0.3, 0.4] for _ in payload]

            def get_dimension(self) -> int:
                return 2

        provider = RecordingProvider()
        monkeypatch.setattr(
            cli_main,
            "_create_embedding_provider_for_command",
            lambda *_args, **_kwargs: provider,
        )
        monkeypatch.setattr(cli_main, "_warm_embedding_provider", lambda _embedding: {})
        monkeypatch.setattr(
            cli_main,
            "_run_symbol_index",
            lambda **_kwargs: {
                "failed": 0,
                "failed_reasons": {},
                "failed_samples": [],
                "duration_ms": 0,
            },
        )

        env = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": "gemini",
        }

        blocked = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert blocked.exit_code == 1, blocked.output
        blocked_payload = _parse_json_output(blocked.output)
        assert blocked_payload["resume_blocked"] is True
        assert blocked_payload["error"]["code"] == "resume_blocked"
        blocked_candidates = blocked_payload["resume_block_candidates"]
        assert isinstance(blocked_candidates, list)
        assert blocked_candidates[0]["build_id"] == "legacy-build"
        assert cache.list_staged_build_ids() == ["legacy-build"]
        assert provider.calls == []

        allowed = runner.invoke(
            cli_main.cli,
            ["index", str(repo), "--json", "--discard-interrupted-builds"],
            env=env,
        )
        assert allowed.exit_code == 0, allowed.output
        allowed_payload = _parse_json_output(allowed.output)
        assert allowed_payload["resume_blocked"] is False
        assert allowed_payload["resumed_build"] is False
        assert provider.calls != []
        assert cache.list_staged_build_ids() == []
