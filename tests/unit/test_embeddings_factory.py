from __future__ import annotations

import pytest

from gloggur.config import GloggurConfig
from gloggur.embeddings import factory


def test_create_embedding_provider_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory returns local embedding provider with config args."""
    created = {}

    class DummyLocal:
        """Dummy local provider capturing constructor args."""
        def __init__(
            self,
            model_name: str,
            cache_dir: str | None,
            fallback_cache_dir: str | None = None,
        ) -> None:
            """Record constructor arguments."""
            created["model_name"] = model_name
            created["cache_dir"] = cache_dir
            created["fallback_cache_dir"] = fallback_cache_dir

    monkeypatch.setattr(factory, "LocalEmbeddingProvider", DummyLocal)

    config = GloggurConfig(
        embedding_provider="local",
        local_embedding_model="local-model",
        model_cache_dir="model-cache",
    )
    provider = factory.create_embedding_provider(config)
    assert isinstance(provider, DummyLocal)
    assert created == {
        "model_name": "local-model",
        "cache_dir": "model-cache",
        "fallback_cache_dir": ".gloggur-cache",
    }


def test_create_embedding_provider_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory returns OpenAI embedding provider with config args."""
    created = {}

    class DummyOpenAI:
        """Dummy OpenAI provider capturing constructor args."""
        def __init__(
            self,
            model: str,
            *,
            openai_api_key: str | None = None,
            openai_base_url: str | None = None,
            openrouter_api_key: str | None = None,
            openrouter_base_url: str | None = None,
            openrouter_site_url: str | None = None,
            openrouter_app_name: str | None = None,
        ) -> None:
            """Record constructor arguments."""
            created["model"] = model
            created["openai_api_key"] = openai_api_key
            created["openai_base_url"] = openai_base_url
            created["openrouter_api_key"] = openrouter_api_key
            created["openrouter_base_url"] = openrouter_base_url
            created["openrouter_site_url"] = openrouter_site_url
            created["openrouter_app_name"] = openrouter_app_name

    monkeypatch.setattr(factory, "OpenAIEmbeddingProvider", DummyOpenAI)

    config = GloggurConfig(
        embedding_provider="openai",
        openai_embedding_model="openai-model",
        openai_api_key="openai-key",
        openai_base_url="https://api.openai.com/v1",
        openrouter_api_key="openrouter-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_site_url="https://example.com",
        openrouter_app_name="gloggur",
    )
    provider = factory.create_embedding_provider(config)
    assert isinstance(provider, DummyOpenAI)
    assert created == {
        "model": "openai-model",
        "openai_api_key": "openai-key",
        "openai_base_url": "https://api.openai.com/v1",
        "openrouter_api_key": "openrouter-key",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_site_url": "https://example.com",
        "openrouter_app_name": "gloggur",
    }


def test_create_embedding_provider_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory returns Gemini embedding provider with config args."""
    created = {}

    class DummyGemini:
        """Dummy Gemini provider capturing constructor args."""
        def __init__(self, model: str, api_key: str | None) -> None:
            """Record constructor arguments."""
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
    """Factory raises for unknown embedding provider."""
    config = GloggurConfig(embedding_provider="unknown")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        factory.create_embedding_provider(config)


def test_create_embedding_provider_test() -> None:
    """Factory returns deterministic test embedding provider."""
    config = GloggurConfig(embedding_provider="test")
    provider = factory.create_embedding_provider(config)
    assert provider.provider == "test"
