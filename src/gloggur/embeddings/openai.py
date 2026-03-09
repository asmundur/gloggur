from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from gloggur.config import (
    ALLOW_CUSTOM_EMBEDDING_ENDPOINTS_ENV,
    OPENAI_DEFAULT_BASE_URL,
    OPENROUTER_DEFAULT_BASE_URL,
    custom_embedding_endpoints_allowed,
    embedding_endpoint_host,
    is_custom_embedding_endpoint,
    normalize_base_url,
)
from gloggur.embeddings.base import EmbeddingProvider


def _optional_string(value: str | None) -> str | None:
    """Normalize optional strings by trimming whitespace and collapsing empty values."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _is_openrouter_endpoint(base_url: str | None) -> bool:
    """Return True when ``base_url`` points at an OpenRouter endpoint."""
    if not base_url:
        return False
    return "openrouter.ai" in base_url.lower()


def _enforce_allowed_base_url(base_url: str | None) -> None:
    """Reject non-default OpenAI-compatible endpoints unless operator env opts in."""
    if not is_custom_embedding_endpoint(
        base_url,
        allowed_defaults=(OPENAI_DEFAULT_BASE_URL, OPENROUTER_DEFAULT_BASE_URL),
    ):
        return
    if custom_embedding_endpoints_allowed():
        return
    raise ValueError(
        "Custom embedding base URLs are disabled by default; "
        f"set {ALLOW_CUSTOM_EMBEDDING_ENDPOINTS_ENV}=1 to allow non-default endpoints."
    )


def _normalize_vector(vector: object, *, model: str, context: str) -> list[float]:
    """Validate and normalize one embedding vector from the OpenAI client."""
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise RuntimeError(
            f"OpenAI embeddings returned invalid vector "
            f"payload for model '{model}' during {context}"
        )
    normalized: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"OpenAI embeddings returned non-numeric "
                f"vector value for model '{model}' "
                f"during {context}"
            )
        normalized.append(float(value))
    if not normalized:
        raise RuntimeError(
            f"OpenAI embeddings returned an empty vector for model '{model}' during {context}"
        )
    return normalized


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls the OpenAI embeddings API."""

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
        """Initialize the OpenAI client and model selection."""
        self.provider = "openai"
        openrouter_key = _optional_string(openrouter_api_key) or _optional_string(
            os.getenv("OPENROUTER_API_KEY")
        )
        openai_key = _optional_string(openai_api_key) or _optional_string(
            os.getenv("OPENAI_API_KEY")
        )
        credential_source = "OPENROUTER_API_KEY" if openrouter_key else "OPENAI_API_KEY"
        api_key = openrouter_key or openai_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY is not set")

        resolved_base_url = _optional_string(openai_base_url) or _optional_string(
            os.getenv("OPENAI_BASE_URL")
        )
        if openrouter_key and not resolved_base_url:
            resolved_base_url = _optional_string(openrouter_base_url) or _optional_string(
                os.getenv("GLOGGUR_OPENROUTER_BASE_URL")
            )
            if not resolved_base_url:
                resolved_base_url = OPENROUTER_DEFAULT_BASE_URL

        normalized_base_url = normalize_base_url(resolved_base_url)
        _enforce_allowed_base_url(normalized_base_url)

        headers: dict[str, str] = {}
        if openrouter_key or _is_openrouter_endpoint(normalized_base_url):
            site_url = _optional_string(openrouter_site_url) or _optional_string(
                os.getenv("GLOGGUR_OPENROUTER_SITE_URL")
            )
            app_name = _optional_string(openrouter_app_name) or _optional_string(
                os.getenv("GLOGGUR_OPENROUTER_APP_NAME")
            )
            if site_url:
                headers["HTTP-Referer"] = site_url
            if app_name:
                headers["X-Title"] = app_name

        self.model = model
        self.credential_source = credential_source
        self.base_url = normalized_base_url
        endpoint_base_url = normalized_base_url or (
            OPENROUTER_DEFAULT_BASE_URL if openrouter_key else OPENAI_DEFAULT_BASE_URL
        )
        self.endpoint_host = embedding_endpoint_host(endpoint_base_url) or "api.openai.com"
        self.default_headers = dict(headers)
        self.client = OpenAI(
            api_key=api_key,
            base_url=normalized_base_url,
            default_headers=headers or None,
        )
        self._dimension: int | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI embedding request failed for model '{self.model}': {exc}"
            ) from exc
        data = getattr(response, "data", None)
        if not isinstance(data, list) or len(data) != 1:
            raise RuntimeError(
                f"OpenAI embeddings returned invalid item "
                f"count for model '{self.model}' during "
                f"single-text embedding"
            )
        vector = _normalize_vector(
            getattr(data[0], "embedding", None),
            model=self.model,
            context="single-text embedding",
        )
        self._dimension = len(vector)
        return vector

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        payload = list(texts)
        if not payload:
            return []
        try:
            response = self.client.embeddings.create(model=self.model, input=payload)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI embedding request failed for model '{self.model}': {exc}"
            ) from exc
        data = getattr(response, "data", None)
        if not isinstance(data, list):
            raise RuntimeError(
                f"OpenAI embeddings returned invalid response "
                f"payload for model '{self.model}' "
                f"during batch embedding"
            )
        vectors = [
            _normalize_vector(
                getattr(item, "embedding", None),
                model=self.model,
                context="batch embedding",
            )
            for item in data
        ]
        if len(vectors) != len(payload):
            raise RuntimeError(
                f"OpenAI embeddings returned "
                f"{len(vectors)} vectors for "
                f"{len(payload)} inputs with "
                f"model '{self.model}'"
            )
        expected_dimension = len(vectors[0]) if vectors else 0
        if any(len(vector) != expected_dimension for vector in vectors):
            raise RuntimeError(
                f"OpenAI embeddings returned inconsistent "
                f"vector dimensions for model '{self.model}'"
            )
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def get_dimension(self) -> int:
        """Return the embedding dimension (probe if unknown)."""
        if self._dimension is None:
            _ = self.embed_text("dimension probe")
        return self._dimension or 0
