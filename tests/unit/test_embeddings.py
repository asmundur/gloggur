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


def test_openai_provider_api_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI failures should include model context and original detail."""

    class FakeEmbeddings:
        def create(self, model: str, input: object):  # noqa: ANN001
            _ = model, input
            raise RuntimeError("401 unauthorized")

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            _ = api_key
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    with pytest.raises(RuntimeError, match="OpenAI embedding request failed"):
        OpenAIEmbeddingProvider.embed_text.__wrapped__(provider, "hello")


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


def test_gemini_provider_api_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini failures should include model context and original detail."""

    class FakeModels:
        def embed_content(self, model: str, contents: object):  # noqa: ANN001
            _ = model, contents
            raise RuntimeError("permission denied")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            _ = api_key
            self.models = FakeModels()

    genai_module = ModuleType("google.genai")
    genai_module.Client = FakeClient
    google_module = ModuleType("google")
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001")
    with pytest.raises(RuntimeError, match="Gemini embedding request failed"):
        provider.embed_text("hello")


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


def _patch_genai(monkeypatch: pytest.MonkeyPatch, fake_client_cls: type) -> None:
    """Patch google.genai with a fake client class for Gemini tests."""
    genai_module = ModuleType("google.genai")
    genai_module.Client = fake_client_cls
    google_module = ModuleType("google")
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)


def test_gemini_embed_batch_empty_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_batch([]) must return [] without calling the API."""
    call_count = 0

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001")
    result = provider.embed_batch([])

    assert result == []
    assert call_count == 0


def test_gemini_embed_batch_single_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_batch with one item returns a 1-element list."""

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001")
    result = provider.embed_batch(["hello"])

    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3]
    assert all(isinstance(v, float) for v in result[0])


def test_gemini_gloggur_api_key_env_var_used_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """GLOGGUR_GEMINI_API_KEY takes precedence over GEMINI_API_KEY and GOOGLE_API_KEY."""
    captured_keys: list[str] = []

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])])

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            captured_keys.append(api_key)
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.setenv("GLOGGUR_GEMINI_API_KEY", "gloggur-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001")

    assert provider.api_key == "gloggur-key"
    assert captured_keys[-1] == "gloggur-key"


def test_gemini_gloggur_api_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """GLOGGUR_GEMINI_API_KEY works when it is the only key set."""
    captured_keys: list[str] = []

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])])

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            captured_keys.append(api_key)
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GLOGGUR_GEMINI_API_KEY", "only-key")

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001")

    assert provider.api_key == "only-key"
    assert captured_keys[-1] == "only-key"


def test_gemini_embed_batch_retries_on_rate_limit_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embed_batch retries when a rate-limit error occurs and returns result on success."""
    call_count = 0

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("429 quota exceeded")
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])] * len(list(contents))
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda _: None)

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001", _chunk_size=10)
    result = provider.embed_batch(["hello", "world"])

    assert len(result) == 2
    assert call_count >= 2


def test_gemini_embed_batch_rate_limit_multiple_retries_does_not_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embed_batch retries through multiple rate-limit errors and must finish."""
    fail_count = 0
    max_fails = 3

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            nonlocal fail_count
            if fail_count < max_fails:
                fail_count += 1
                raise RuntimeError("rate limit exceeded")
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0, 0.0])] * len(list(contents))
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda _: None)

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001", _chunk_size=10)
    result = provider.embed_batch(["a"])

    assert len(result) == 1
    assert fail_count == max_fails


def test_gemini_embed_batch_prefers_single_large_request_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When supported, embed_batch should send one large Gemini request first."""
    call_sizes: list[int] = []

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            _ = model
            size = len(list(contents))
            call_sizes.append(size)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])] * size
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            _ = api_key
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = GeminiEmbeddingProvider(
        model="gemini-embedding-001",
        _chunk_size=2,
        _batch_first=True,
    )
    result = provider.embed_batch(["a", "b", "c", "d", "e"])

    assert len(result) == 5
    assert call_sizes == [5]


def test_gemini_embed_batch_falls_back_to_chunked_mode_when_large_request_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large-batch failures should fall back to chunked Gemini requests."""
    call_sizes: list[int] = []

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            _ = model
            size = len(list(contents))
            call_sizes.append(size)
            if size > 2:
                raise RuntimeError("400 request payload too large")
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])] * size
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            _ = api_key
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = GeminiEmbeddingProvider(
        model="gemini-embedding-001",
        _chunk_size=2,
        _batch_first=True,
    )
    result = provider.embed_batch(["a", "b", "c", "d", "e"])

    assert len(result) == 5
    assert call_sizes == [5, 2, 2, 1]


def test_local_embedding_fallback_vector_is_normalized(tmp_path: Path) -> None:
    """Fallback embeddings are normalized to unit length."""
    provider = LocalEmbeddingProvider("local", cache_dir=str(tmp_path))
    provider._use_fallback = True

    vector = provider.embed_text("Hello world")
    assert len(vector) == provider.get_dimension()
    assert provider.get_dimension() == 256
    assert sum(v * v for v in vector) == pytest.approx(1.0, rel=1e-6)


def test_local_embedding_fallback_marker_uses_fallback_cache_dir(tmp_path: Path) -> None:
    """Fallback marker should be read from cache dir independent of model cache dir."""
    model_cache_dir = tmp_path / "models"
    fallback_cache_dir = tmp_path / "index-cache"
    marker = fallback_cache_dir / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)

    provider = LocalEmbeddingProvider(
        "local",
        cache_dir=str(model_cache_dir),
        fallback_cache_dir=str(fallback_cache_dir),
    )
    assert provider._fallback_marker == marker
    assert provider._use_fallback is True
    assert provider.get_dimension() == 256


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
