from __future__ import annotations

from gloggur.config import GloggurConfig
from gloggur.embeddings.base import EmbeddingProvider
from gloggur.embeddings.gemini import GeminiEmbeddingProvider
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.embeddings.openai import OpenAIEmbeddingProvider


def create_embedding_provider(config: GloggurConfig) -> EmbeddingProvider:
    provider = config.embedding_provider
    if provider == "local":
        return LocalEmbeddingProvider(
            model_name=config.local_embedding_model,
            cache_dir=config.model_cache_dir,
        )
    if provider == "openai":
        return OpenAIEmbeddingProvider(model=config.openai_embedding_model)
    if provider == "gemini":
        return GeminiEmbeddingProvider(
            model=config.gemini_embedding_model,
            api_key=config.gemini_api_key,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")
