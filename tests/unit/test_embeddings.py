from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gloggur.embeddings.gemini import GeminiEmbeddingProvider
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.embeddings.openai import OpenAIEmbeddingProvider


def test_local_embedding_fallback_marker_enables_offline_vectors() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)

    provider = LocalEmbeddingProvider("local", cache_dir=cache_dir)
    vector = provider.embed_text("hello world")
    batch = provider.embed_batch(["hello", "world"])

    assert len(vector) == provider.get_dimension()
    assert provider.get_dimension() == 256
    assert len(batch) == 2
    assert len(batch[0]) == provider.get_dimension()


def test_openai_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(model="text-embedding-3-large")


def test_gemini_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbeddingProvider(model="gemini-embedding-001", api_key=None)
