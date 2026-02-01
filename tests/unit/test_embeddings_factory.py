from __future__ import annotations

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings import factory


def test_create_embedding_provider_local(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    class DummyLocal:
        def __init__(self, model_name: str, cache_dir: str | None) -> None:
            created["model_name"] = model_name
            created["cache_dir"] = cache_dir

    monkeypatch.setattr(factory, "LocalEmbeddingProvider", DummyLocal)

    config = GloggurConfig(
        embedding_provider="local",
        local_embedding_model="local-model",
        model_cache_dir="model-cache",
    )
    provider = factory.create_embedding_provider(config)
    assert isinstance(provider, DummyLocal)
    assert created == {"model_name": "local-model", "cache_dir": "model-cache"}


def test_create_embedding_provider_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    class DummyOpenAI:
        def __init__(self, model: str) -> None:
            created["model"] = model

    monkeypatch.setattr(factory, "OpenAIEmbeddingProvider", DummyOpenAI)

    config = GloggurConfig(embedding_provider="openai", openai_embedding_model="openai-model")
    provider = factory.create_embedding_provider(config)
    assert isinstance(provider, DummyOpenAI)
    assert created == {"model": "openai-model"}


def test_create_embedding_provider_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    class DummyGemini:
        def __init__(self, model: str, api_key: str | None) -> None:
            created["model"] = model
            created["api_key"] = api_key

    monkeypatch.setattr(factory, "GeminiEmbeddingProvider", DummyGemini)

    config = GloggurConfig(
        embedding_provider="gemini",
        gemini_embedding_model="gemini-model",
        gemini_api_key="gemini-key",
    )
    provider = factory.create_embedding_provider(config)
    assert isinstance(provider, DummyGemini)
    assert created == {"model": "gemini-model", "api_key": "gemini-key"}


def test_create_embedding_provider_unknown() -> None:
    config = GloggurConfig(embedding_provider="unknown")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        factory.create_embedding_provider(config)
