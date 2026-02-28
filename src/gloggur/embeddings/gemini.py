from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from tenacity import retry, retry_if_exception_type, wait_exponential

from gloggur.embeddings.base import EmbeddingProvider


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception looks like a Gemini rate-limit / quota error."""
    msg = str(exc).lower()
    type_name = type(exc).__name__.lower()
    return (
        "429" in msg
        or "quota" in msg
        or "rate" in msg
        or "resource_exhausted" in type_name
        or "ratelimit" in type_name
        or "quotaexceeded" in type_name
    )


class _RateLimitError(Exception):
    """Sentinel re-raised when a Gemini call hits a rate-limit / quota error."""


def _normalize_vector(vector: object, *, model: str, context: str) -> list[float]:
    """Validate and normalize one embedding vector from Gemini."""
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise RuntimeError(
            f"Gemini embeddings returned invalid vector "
            f"payload for model '{model}' during {context}"
        )
    normalized: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"Gemini embeddings returned non-numeric "
                f"vector value for model '{model}' "
                f"during {context}"
            )
        normalized.append(float(value))
    if not normalized:
        raise RuntimeError(
            f"Gemini embeddings returned an empty vector for model '{model}' during {context}"
        )
    return normalized


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls the Gemini embeddings API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        _chunk_size: int = 50,
        _batch_first: bool = True,
    ) -> None:
        """Initialize the Gemini client and model selection."""
        self.provider = "gemini"
        self.model = model
        self.api_key = (
            api_key
            or os.getenv("GLOGGUR_GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "GLOGGUR_GEMINI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY is required "
                "for Gemini embeddings"
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini embeddings") from exc
        self._client = genai.Client(api_key=self.api_key)
        self._dimension: int | None = None
        self._chunk_size = _chunk_size
        self._batch_first = _batch_first

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        vectors = self.embed_batch([text])
        if not vectors:
            raise RuntimeError("Gemini embeddings returned no vectors")
        return vectors[0]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings, preferring one full request then chunk fallback."""
        payload = list(texts)
        if not payload:
            return []
        if self._batch_first and len(payload) > self._chunk_size:
            try:
                vectors = self._embed_chunk_with_retry(payload)
                if vectors:
                    self._dimension = len(vectors[0])
                return vectors
            except RuntimeError:
                # Fall back to chunked mode when a provider/account cannot accept large requests.
                pass
        results: list[list[float]] = []
        for i in range(0, len(payload), self._chunk_size):
            chunk = payload[i : i + self._chunk_size]
            vectors = self._embed_chunk_with_retry(chunk)
            results.extend(vectors)
        if results:
            self._dimension = len(results[0])
        return results

    def _embed_chunk_with_retry(self, chunk: list[str]) -> list[list[float]]:
        """Call the Gemini API for one chunk, retrying indefinitely on rate-limit errors."""

        @retry(
            retry=retry_if_exception_type(_RateLimitError),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            reraise=True,
        )
        def _call() -> list[list[float]]:
            """Execute one Gemini batch request and remap rate-limit/runtime failures."""
            try:
                response = self._client.models.embed_content(model=self.model, contents=chunk)
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    raise _RateLimitError(str(exc)) from exc
                raise RuntimeError(
                    f"Gemini embedding request failed for model '{self.model}': {exc}"
                ) from exc
            vectors = self._extract_vectors(
                response,
                model=self.model,
                expected_count=len(chunk),
                context="batch embedding",
            )
            return vectors

        return _call()

    def get_dimension(self) -> int:
        """Return the embedding dimension (probe if unknown)."""
        if self._dimension is None:
            _ = self.embed_text("dimension probe")
        return self._dimension or 0

    @staticmethod
    def _extract_vectors(
        response: object,
        *,
        model: str = "unknown",
        expected_count: int | None = None,
        context: str = "embedding extraction",
    ) -> list[list[float]]:
        """Extract embedding vectors from a Gemini response object."""
        embeddings = getattr(response, "embeddings", None) or getattr(response, "embedding", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings") or response.get("embedding")
        if embeddings is None:
            raise RuntimeError(
                f"Gemini embeddings returned no embeddings for model '{model}' during {context}"
            )
        vectors: list[list[float]] = []
        for item in embeddings:
            if isinstance(item, list):
                vectors.append(_normalize_vector(item, model=model, context=context))
            elif isinstance(item, dict):
                if "values" in item:
                    vectors.append(_normalize_vector(item["values"], model=model, context=context))
                elif "embedding" in item:
                    vectors.append(
                        _normalize_vector(item["embedding"], model=model, context=context)
                    )
            elif hasattr(item, "values"):
                vectors.append(_normalize_vector(item.values, model=model, context=context))
        if not vectors:
            raise RuntimeError(
                f"Gemini embeddings returned no usable vectors for model '{model}' during {context}"
            )
        if expected_count is not None and len(vectors) != expected_count:
            raise RuntimeError(
                f"Gemini embeddings returned {len(vectors)} "
                f"vectors for {expected_count} inputs "
                f"with model '{model}'"
            )
        expected_dimension = len(vectors[0])
        if any(len(vector) != expected_dimension for vector in vectors):
            raise RuntimeError(
                f"Gemini embeddings returned inconsistent "
                f"vector dimensions for model '{model}' "
                f"during {context}"
            )
        return vectors
