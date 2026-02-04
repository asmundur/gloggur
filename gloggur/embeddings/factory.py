from __future__ import annotations

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.local import LocalEmbeddingProvider


def create_embedding_provider(config: GloggurConfig) -> EmbeddingProvider:
    """Create an embedding provider based on configuration."""
    provider = config.embedding_provider
    if provider == "local":
        return LocalEmbeddingProvider(
            model_name=config.local_embedding_model,
            cache_dir=config.model_cache_dir,
        )
    if provider == "openai":
        try:
            from gloggur.embeddings.openai import OpenAIEmbeddingProvider
        except ImportError as exc:
            raise RuntimeError("Install with pip install gloggur[openai]") from exc
        return OpenAIEmbeddingProvider(model=config.openai_embedding_model)
    if provider == "gemini":
        try:
            from gloggur.embeddings.gemini import GeminiEmbeddingProvider
        except ImportError as exc:
            raise RuntimeError("Install with pip install gloggur[gemini]") from exc
        return GeminiEmbeddingProvider(
            model=config.gemini_embedding_model,
            api_key=config.gemini_api_key,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")
