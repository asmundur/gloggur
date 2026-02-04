from __future__ import annotations

import tempfile
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gloggur.embeddings.gemini import GeminiEmbeddingProvider
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.embeddings.openai import OpenAIEmbeddingProvider


def test_local_embedding_fallback_marker_enables_offline_vectors() -> None:
    """Ensure fallback marker enables deterministic local embeddings."""
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
    """OpenAI provider should require API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(model="text-embedding-3-large")


def test_gemini_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini provider should require API key."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbeddingProvider(model="gemini-embedding-001", api_key=None)


def test_openai_provider_embeddings_and_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI provider returns embeddings and dimension."""
    class FakeEmbeddings:
        """Fake OpenAI embeddings client."""
        def create(self, model: str, input: object):
            """Return fake embeddings for testing."""
            if isinstance(input, list):
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=[1.0, 2.0]),
                        SimpleNamespace(embedding=[3.0, 4.0]),
                    ]
                )
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.25])])

    class FakeOpenAI:
        """Fake OpenAI client wrapper."""
        def __init__(self, api_key: str) -> None:
            """Store fake embeddings client."""
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    vector = provider.embed_text("hello")
    assert vector == [0.5, 0.25]
    assert provider.get_dimension() == 2

    batch = provider.embed_batch(["a", "b"])
    assert batch == [[1.0, 2.0], [3.0, 4.0]]
    assert provider.get_dimension() == 2


def test_gemini_embed_text_and_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini provider returns embeddings and dimension."""
    class FakeModels:
        """Fake Gemini models client."""
        def embed_content(self, model: str, contents: object):
            """Return fake embedding response."""
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])

    class FakeClient:
        """Fake Gemini client wrapper."""
        def __init__(self, api_key: str) -> None:
            """Store fake models client."""
            self.models = FakeModels()

    genai_module = ModuleType("google.genai")
    genai_module.Client = FakeClient
    google_module = ModuleType("google")
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001")
    vector = provider.embed_text("hello")
    assert vector == [0.1, 0.2]
    assert provider.get_dimension() == 2


def test_gemini_extract_vectors_variants() -> None:
    """Gemini vector extraction handles multiple response shapes."""
    response_lists = SimpleNamespace(embeddings=[[0.1, 0.2], [0.3, 0.4]])
    assert GeminiEmbeddingProvider._extract_vectors(response_lists) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    response_values = SimpleNamespace(embeddings=[SimpleNamespace(values=(0.5, 0.6))])
    assert GeminiEmbeddingProvider._extract_vectors(response_values) == [[0.5, 0.6]]

    response_embedding = {"embedding": [{"embedding": [0.9, 1.0]}]}
    assert GeminiEmbeddingProvider._extract_vectors(response_embedding) == [[0.9, 1.0]]


def test_local_embedding_fallback_vector_is_normalized(tmp_path: Path) -> None:
    """Fallback embeddings are normalized to unit length."""
    provider = LocalEmbeddingProvider("local", cache_dir=str(tmp_path))
    provider._use_fallback = True

    vector = provider.embed_text("Hello world")
    assert len(vector) == provider.get_dimension()
    assert provider.get_dimension() == 256
    assert sum(v * v for v in vector) == pytest.approx(1.0, rel=1e-6)


def test_local_embedding_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local embedding provider uses model encode paths."""
    class FakeVector:
        """Fake vector object with tolist."""
        def __init__(self, values) -> None:
            """Store fake vector values."""
            self._values = values

        def tolist(self):
            """Return vector values as list."""
            return list(self._values)

    class FakeModel:
        """Fake model with encode/dimension methods."""
        def encode(self, texts, normalize_embeddings: bool = True):
            """Return fake vector values for inputs."""
            if isinstance(texts, list) and len(texts) == 1:
                return [FakeVector([0.1, 0.2])]
            return [FakeVector([0.3, 0.4]), FakeVector([0.5, 0.6])]

        def get_sentence_embedding_dimension(self) -> int:
            """Return fake embedding dimension."""
            return 2

    provider = LocalEmbeddingProvider("local")
    monkeypatch.setattr(provider, "_load_model", lambda: FakeModel())

    vector = provider.embed_text("hello")
    assert vector == [0.1, 0.2]

    batch = provider.embed_batch(["a", "b"])
    assert batch == [[0.3, 0.4], [0.5, 0.6]]
    assert provider.get_dimension() == 2
