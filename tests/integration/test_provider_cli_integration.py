from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from gloggur.cli.main import cli
from scripts.verification.fixtures import TestFixtures


def _parse_json_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    return payload


@pytest.mark.parametrize(
    ("provider", "model_env", "model_value", "factory_attr"),
    [
        ("openai", "GLOGGUR_OPENAI_MODEL", "openai-test-model", "OpenAIEmbeddingProvider"),
        ("gemini", "GLOGGUR_GEMINI_MODEL", "gemini-test-model", "GeminiEmbeddingProvider"),
    ],
)
def test_index_and_search_use_selected_provider_with_mocked_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model_env: str,
    model_value: str,
    factory_attr: str,
) -> None:
    """Index/search should run end-to-end for OpenAI/Gemini paths with deterministic mocks."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix=f"gloggur-{provider}-cache-")
        observed_models: list[str] = []

        class FakeProvider:
            provider = "unknown"

            def __init__(self, model: str, api_key: str | None = None) -> None:
                _ = api_key
                observed_models.append(model)

            def embed_text(self, _text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[float(idx + 1), 0.5, 0.25] for idx, _ in enumerate(texts)]

            def get_dimension(self) -> int:
                return 3

        FakeProvider.provider = provider
        monkeypatch.setattr(f"gloggur.embeddings.factory.{factory_attr}", FakeProvider)
        env = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_EMBEDDING_PROVIDER": provider,
            model_env: model_value,
            "GLOGGUR_GEMINI_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key",
        }

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0, index_result.output
        index_payload = _parse_json_output(index_result.output)
        assert int(index_payload["indexed_files"]) == 1
        assert int(index_payload["indexed_symbols"]) > 0

        search_result = runner.invoke(cli, ["search", "add", "--json", "--top-k", "5"], env=env)
        assert search_result.exit_code == 0, search_result.output
        search_payload = _parse_json_output(search_result.output)
        metadata = search_payload["metadata"]
        assert isinstance(metadata, dict)
        assert int(metadata["total_results"]) > 0
        assert observed_models and observed_models[0] == model_value


@pytest.mark.parametrize(
    ("provider", "detail_hint"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
    ],
)
def test_missing_provider_credentials_fail_without_traceback_and_with_json_payload(
    tmp_path: Path,
    provider: str,
    detail_hint: str,
) -> None:
    """Provider init failures should be actionable and traceback-free for JSON mode."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )
    env = {
        "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
        "GLOGGUR_EMBEDDING_PROVIDER": provider,
    }

    result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)

    assert result.exit_code == 1
    assert "Traceback (most recent call last)" not in result.output
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "embedding_provider_error"
    assert error["provider"] == provider
    assert detail_hint in str(error["detail"])


def test_gemini_profile_not_overwritten_by_different_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indexing with a second provider profile in a separate cache dir does not touch
    the Gemini cache directory."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        gemini_cache = tmp_path / "gemini-cache"
        openai_cache = tmp_path / "openai-cache"

        class FakeGeminiProvider:
            provider = "gemini"

            def __init__(self, model: str, api_key: str | None = None) -> None:
                pass

            def embed_text(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

            def get_dimension(self) -> int:
                return 3

        class FakeOpenAIProvider:
            provider = "openai"

            def __init__(self, model: str, api_key: str | None = None) -> None:
                pass

            def embed_text(self, text: str) -> list[float]:
                return [0.4, 0.5, 0.6]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.4, 0.5, 0.6] for _ in texts]

            def get_dimension(self) -> int:
                return 3

        monkeypatch.setattr("gloggur.embeddings.factory.GeminiEmbeddingProvider", FakeGeminiProvider)
        monkeypatch.setattr("gloggur.embeddings.factory.OpenAIEmbeddingProvider", FakeOpenAIProvider)

        # Step 1: Index with Gemini profile into gemini_cache
        gemini_env = {
            "GLOGGUR_CACHE_DIR": str(gemini_cache),
            "GLOGGUR_EMBEDDING_PROVIDER": "gemini",
            "GLOGGUR_GEMINI_MODEL": "gemini-embedding-001",
            "GLOGGUR_GEMINI_API_KEY": "test-key",
        }
        r1 = runner.invoke(cli, ["index", str(repo), "--json"], env=gemini_env)
        assert r1.exit_code == 0, r1.output

        # Snapshot Gemini cache files
        gemini_files_before = set(gemini_cache.rglob("*"))
        assert gemini_files_before, "Gemini cache should have files after indexing"

        # Step 2: Index with OpenAI profile into openai_cache
        openai_env = {
            "GLOGGUR_CACHE_DIR": str(openai_cache),
            "GLOGGUR_EMBEDDING_PROVIDER": "openai",
            "GLOGGUR_OPENAI_MODEL": "text-embedding-3-small",
            "OPENAI_API_KEY": "test-key",
        }
        r2 = runner.invoke(cli, ["index", str(repo), "--json"], env=openai_env)
        assert r2.exit_code == 0, r2.output

        # Step 3: Assert Gemini cache files are untouched
        gemini_files_after = set(gemini_cache.rglob("*"))
        assert gemini_files_before == gemini_files_after, (
            "Gemini cache files were modified when indexing with OpenAI profile"
        )

        # Assert the two cache dirs are distinct
        assert gemini_cache != openai_cache
        assert openai_cache.exists()
        openai_files = set(openai_cache.rglob("*"))
        assert openai_files, "OpenAI cache should have files after indexing"
