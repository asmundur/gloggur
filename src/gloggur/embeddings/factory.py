from __future__ import annotations

from gloggur.adapters.registry import (
    AdapterRegistry,
    AdapterResolutionError,
    adapter_module_override,
    instantiate_adapter,
)
from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.embeddings.test_provider import DeterministicTestEmbeddingProvider

try:
    from gloggur.embeddings.openai import OpenAIEmbeddingProvider
except ImportError:
    OpenAIEmbeddingProvider = None  # type: ignore[assignment]

try:
    from gloggur.embeddings.gemini import GeminiEmbeddingProvider
except ImportError:
    GeminiEmbeddingProvider = None  # type: ignore[assignment]


_EMBEDDING_ADAPTERS = AdapterRegistry[EmbeddingProvider]("gloggur.embedding_providers")


def _create_local_provider(config: GloggurConfig) -> EmbeddingProvider:
    return LocalEmbeddingProvider(
        model_name=config.local_embedding_model,
        cache_dir=config.model_cache_dir,
        fallback_cache_dir=config.cache_dir,
    )


def _create_openai_provider(config: GloggurConfig) -> EmbeddingProvider:
    provider_cls: type[EmbeddingProvider] | None = OpenAIEmbeddingProvider
    if provider_cls is None:
        try:
            from gloggur.embeddings.openai import OpenAIEmbeddingProvider as provider_cls
        except ImportError as exc:
            raise RuntimeError("Install with pip install gloggur[openai]") from exc
    return provider_cls(model=config.openai_embedding_model)


def _create_gemini_provider(config: GloggurConfig) -> EmbeddingProvider:
    provider_cls = GeminiEmbeddingProvider
    if provider_cls is None:
        try:
            from gloggur.embeddings.gemini import GeminiEmbeddingProvider as provider_cls
        except ImportError as exc:
            raise RuntimeError("Install with pip install gloggur[gemini]") from exc
    return provider_cls(
        model=config.gemini_embedding_model,
        api_key=config.gemini_api_key,
    )


def _create_test_provider(config: GloggurConfig) -> EmbeddingProvider:
    _ = config
    return DeterministicTestEmbeddingProvider()


_EMBEDDING_ADAPTERS.register_builtin("local", _create_local_provider)
_EMBEDDING_ADAPTERS.register_builtin("openai", _create_openai_provider)
_EMBEDDING_ADAPTERS.register_builtin("gemini", _create_gemini_provider)
_EMBEDDING_ADAPTERS.register_builtin("test", _create_test_provider)


def list_embedding_provider_adapters() -> list[dict[str, object]]:
    """Return discoverable embedding-provider adapter descriptors."""
    return _EMBEDDING_ADAPTERS.available()


def create_embedding_provider(config: GloggurConfig) -> EmbeddingProvider:
    """Create an embedding provider based on configuration and adapter registry."""
    provider = config.embedding_provider
    module_override = adapter_module_override(
        config.adapters if isinstance(config.adapters, dict) else None,
        category="embedding_providers",
        adapter_id=provider,
    )
    try:
        factory = _EMBEDDING_ADAPTERS.resolve_factory(
            provider,
            module_path_override=module_override,
        )
    except AdapterResolutionError as exc:
        raise ValueError(f"Unknown embedding provider: {provider}") from exc
    return instantiate_adapter(factory, config)
