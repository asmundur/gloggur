from __future__ import annotations

import json
import tempfile

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

        second = runner.invoke(cli_main.cli, ["index", str(repo), "--json"], env=env)
        assert second.exit_code == 0, second.output
        second_payload = _parse_json_output(second.output)
        assert second_payload["resumed_build"] is True
        assert second_payload["resume_source"] == "manifest"
        assert len(second_provider.calls) == 1
        assert len(second_provider.calls[0]) == 1


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
