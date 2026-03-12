from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from gloggur.embeddings.gemini import GeminiEmbeddingProvider
from gloggur.embeddings.local import LocalEmbeddingProvider
from gloggur.embeddings.openai import OpenAIEmbeddingProvider


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout_text: str = "",
    stderr_text: str = "",
) -> None:
    """Install a lightweight sentence-transformers stub for bootstrap tests."""

    class FakeVector:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return list(self._values)

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
            _ = model_name, cache_folder
            if stdout_text:
                sys.stdout.write(stdout_text)
            if stderr_text:
                sys.stderr.write(stderr_text)

        def encode(self, texts, normalize_embeddings: bool = True):  # noqa: ANN001
            _ = normalize_embeddings
            return [FakeVector([0.25, 0.75]) for _ in texts]

        def get_sentence_embedding_dimension(self) -> int:
            return 2

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_local_embedding_legacy_fallback_controls_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy fallback marker should not enable hash-vector behavior."""
    marker = tmp_path / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)

    class FakeVector:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return list(self._values)

    class FakeModel:
        def encode(self, texts, normalize_embeddings: bool = True):  # noqa: ANN001
            _ = texts, normalize_embeddings
            return [FakeVector([0.25, 0.75])]

    provider = LocalEmbeddingProvider("local", cache_dir=str(tmp_path))
    monkeypatch.setattr(provider, "_load_model", lambda: FakeModel())
    assert provider.embed_text("hello") == [0.25, 0.75]


def test_local_embedding_requires_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sentence-transformers should be a hard local-provider failure."""
    import builtins

    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "sentence_transformers":
            raise ImportError("missing sentence_transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    provider = LocalEmbeddingProvider("local")
    with pytest.raises(RuntimeError, match="sentence-transformers is required"):
        provider.embed_text("hello world")


def test_local_embedding_filters_expected_wrapper_warning() -> None:
    """The known sentence-transformers wrapper bootstrap line should be suppressed."""
    filtered = LocalEmbeddingProvider._filter_expected_wrapper_warning(
        "No sentence-transformers model found with name foo. "
        "Creating a new one with mean pooling.\n"
    )

    assert filtered == ""


def test_local_embedding_suppresses_bootstrap_progress_from_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Known local model bootstrap chatter should not leak to stdout/stderr."""
    _install_fake_sentence_transformers(
        monkeypatch,
        stdout_text=(
            "Loading weights:  50%|#####     | 100/199\r"
            "Loading weights: 100%|##########| 199/199\r"
            "Materializing param=encoder.layer.0.output.dense.weight\r"
        ),
        stderr_text=(
            "No sentence-transformers model found with name fake. "
            "Creating a new one with mean pooling.\n"
        ),
    )

    provider = LocalEmbeddingProvider("fake", cache_dir=str(tmp_path))

    assert provider.get_dimension() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_local_embedding_preserves_unexpected_stderr_lines() -> None:
    """Unexpected stderr must remain visible even when the wrapper line is filtered."""
    filtered = LocalEmbeddingProvider._filter_expected_wrapper_warning(
        "No sentence-transformers model found with name foo. "
        "Creating a new one with mean pooling.\n"
        "real warning\n"
    )

    assert filtered == "real warning\n"


def test_local_embedding_redirects_unexpected_bootstrap_stdout_to_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unexpected bootstrap stdout should be preserved, but only via stderr."""
    _install_fake_sentence_transformers(
        monkeypatch,
        stdout_text="unexpected bootstrap stdout\n",
        stderr_text="unexpected bootstrap stderr\n",
    )

    provider = LocalEmbeddingProvider("fake", cache_dir=str(tmp_path))

    assert provider.get_dimension() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unexpected bootstrap stdout" in captured.err
    assert "unexpected bootstrap stderr" in captured.err


def test_openai_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI provider should require API key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY or OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(model="text-embedding-3-large")


def test_openai_provider_accepts_openrouter_key_and_defaults_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter key should activate OpenAI provider with OpenRouter default base URL."""
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["default_headers"] = default_headers
            self.embeddings = SimpleNamespace(create=lambda **_: None)

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("GLOGGUR_OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")

    assert captured["api_key"] == "openrouter-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["default_headers"] is None
    assert provider.credential_source == "OPENROUTER_API_KEY"


def test_openai_provider_prefers_openrouter_key_over_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter key should win when both OpenRouter and OpenAI keys are set."""
    captured_api_key: list[str] = []

    class FakeOpenAI:
        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            _ = base_url, default_headers
            captured_api_key.append(api_key)
            self.embeddings = SimpleNamespace(create=lambda **_: None)

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")

    assert captured_api_key == ["openrouter-key"]
    assert provider.credential_source == "OPENROUTER_API_KEY"


def test_openai_provider_uses_openrouter_base_url_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter base URL override should be used when the OpenRouter key is active."""
    captured_base_urls: list[str | None] = []

    class FakeOpenAI:
        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            _ = api_key, default_headers
            captured_base_urls.append(base_url)
            self.embeddings = SimpleNamespace(create=lambda **_: None)

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("GLOGGUR_OPENROUTER_BASE_URL", "https://router.custom/v1")
    monkeypatch.setenv("GLOGGUR_ALLOW_CUSTOM_EMBEDDING_ENDPOINTS", "1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")

    assert captured_base_urls == ["https://router.custom/v1"]
    assert provider.base_url == "https://router.custom/v1"


def test_openai_provider_rejects_custom_base_url_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom OpenAI-compatible endpoints should fail closed unless operator env opts in."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.test/v1")
    monkeypatch.delenv("GLOGGUR_ALLOW_CUSTOM_EMBEDDING_ENDPOINTS", raising=False)

    with pytest.raises(ValueError, match="Custom embedding base URLs are disabled"):
        OpenAIEmbeddingProvider(model="text-embedding-3-large")


def test_openai_provider_sets_openrouter_optional_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter endpoint should forward optional attribution headers."""
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["default_headers"] = default_headers
            self.embeddings = SimpleNamespace(create=lambda **_: None)

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("GLOGGUR_OPENROUTER_SITE_URL", "https://example.com")
    monkeypatch.setenv("GLOGGUR_OPENROUTER_APP_NAME", "gloggur")
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    OpenAIEmbeddingProvider(model="text-embedding-3-large")

    assert captured["api_key"] == "openrouter-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["default_headers"] == {
        "HTTP-Referer": "https://example.com",
        "X-Title": "gloggur",
    }


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

        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            """Store fake embeddings client."""
            _ = api_key, base_url, default_headers
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


def test_openai_embed_batch_empty_returns_empty_without_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI batch embedding should not call the API for an empty batch."""
    call_count = 0

    class FakeEmbeddings:
        def create(self, model: str, input: object):
            nonlocal call_count
            _ = model, input
            call_count += 1
            return SimpleNamespace(data=[])

    class FakeOpenAI:
        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            _ = api_key, base_url, default_headers
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    assert provider.embed_batch([]) == []
    assert call_count == 0


def test_openai_provider_api_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI failures should include model context and original detail."""

    class FakeEmbeddings:
        def create(self, model: str, input: object):  # noqa: ANN001
            _ = model, input
            raise RuntimeError("401 unauthorized")

    class FakeOpenAI:
        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            _ = api_key, base_url, default_headers
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    with pytest.raises(RuntimeError, match="OpenAI embedding request failed"):
        OpenAIEmbeddingProvider.embed_text.__wrapped__(provider, "hello")


def test_openai_provider_fails_closed_on_malformed_batch_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI provider should fail loud on missing vectors instead of returning partial results."""

    class FakeEmbeddings:
        def create(self, model: str, input: object):
            _ = model
            payload = list(input) if isinstance(input, list) else [input]
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2])] * max(0, len(payload) - 1)
            )

    class FakeOpenAI:
        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            _ = api_key, base_url, default_headers
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 inputs"):
        provider.embed_batch(["a", "b"])


def test_openai_provider_fails_closed_on_non_numeric_vector_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI provider should reject malformed vector entries."""

    class FakeEmbeddings:
        def create(self, model: str, input: object):
            _ = model, input
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, "bad"])])

    class FakeOpenAI:
        def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            _ = api_key, base_url, default_headers
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("gloggur.embeddings.openai.OpenAI", FakeOpenAI)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    with pytest.raises(RuntimeError, match="non-numeric vector value"):
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


def test_gemini_extract_vectors_fails_closed_when_response_has_no_embeddings() -> None:
    """Gemini extraction should fail loud instead of returning an empty list."""
    with pytest.raises(RuntimeError, match="returned no embeddings"):
        GeminiEmbeddingProvider._extract_vectors({}, model="gemini-embedding-001")


def test_gemini_extract_vectors_rejects_non_numeric_values() -> None:
    """Gemini extraction should reject malformed vector payloads."""
    response = {"embeddings": [{"values": [0.5, "bad"]}]}
    with pytest.raises(RuntimeError, match="non-numeric vector value"):
        GeminiEmbeddingProvider._extract_vectors(response, model="gemini-embedding-001")


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
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])])

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
    assert provider.retry_attempts_total == max_fails
    assert provider.retry_wait_seconds_total > 0


def test_gemini_embed_batch_rate_limit_exhaustion_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent Gemini rate-limit failures should terminate after a finite retry ceiling."""

    class FakeModels:
        def embed_content(self, model: str, contents: object):
            _ = model, contents
            raise RuntimeError("429 quota exceeded")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    _patch_genai(monkeypatch, FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda _: None)

    provider = GeminiEmbeddingProvider(model="gemini-embedding-001", _chunk_size=10)

    with pytest.raises(RuntimeError, match="exhausted rate-limit retries"):
        provider.embed_batch(["a"])

    assert provider.retry_attempts_total == 4
    assert provider.retry_wait_seconds_total == pytest.approx(30.0)


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
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])] * size)

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
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])] * size)

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


def test_local_embedding_legacy_marker_uses_fallback_cache_dir(tmp_path: Path) -> None:
    """Legacy marker in fallback_cache_dir should not activate hash-vector behavior."""
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
    assert provider.fallback_cache_dir == str(fallback_cache_dir)


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
