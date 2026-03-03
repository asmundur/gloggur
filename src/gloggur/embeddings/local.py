from __future__ import annotations

from collections.abc import Iterable

from gloggur.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        fallback_cache_dir: str | None = None,
    ) -> None:
        """Configure local model loading."""
        self.provider = "local"
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.fallback_cache_dir = fallback_cache_dir
        self._model = None

    def _load_model(self):
        """Load the sentence-transformers model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for local embeddings; "
                "install local extras (pip install -e '.[local]')."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local embedding model '{self.model_name}' "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        model = self._load_model()
        vector = model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        payload = list(texts)
        if not payload:
            return []
        model = self._load_model()
        vectors = model.encode(payload, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
