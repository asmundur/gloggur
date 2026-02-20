from __future__ import annotations

from typing import Optional, Type

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.local import LocalEmbeddingProvider

try:
    from gloggur.embeddings.openai import OpenAIEmbeddingProvider
except ImportError:
    OpenAIEmbeddingProvider = None  # type: ignore[assignment]

try:
    from gloggur.embeddings.gemini import GeminiEmbeddingProvider
except ImportError:
    GeminiEmbeddingProvider = None  # type: ignore[assignment]


def create_embedding_provider(config: GloggurConfig) -> EmbeddingProvider:
    """Create an embedding provider based on configuration."""
    provider = config.embedding_provider
    if provider == "local":
        return LocalEmbeddingProvider(
            model_name=config.local_embedding_model,
            cache_dir=config.model_cache_dir,
        )
    if provider == "openai":
        provider_cls: Optional[Type[EmbeddingProvider]] = OpenAIEmbeddingProvider
        if provider_cls is None:
            try:
                from gloggur.embeddings.openai import OpenAIEmbeddingProvider as provider_cls
            except ImportError as exc:
                raise RuntimeError("Install with pip install gloggur[openai]") from exc
        return provider_cls(model=config.openai_embedding_model)
    if provider == "gemini":
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
    raise ValueError(f"Unknown embedding provider: {provider}")
