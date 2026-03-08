from __future__ import annotations

import io
import sys
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout

from gloggur.embeddings.base import EmbeddingProvider

_EXPECTED_WRAPPER_WARNING_TOKENS = (
    "No sentence-transformers model found",
    "Creating a new one with mean pooling.",
)
_EXPECTED_BOOTSTRAP_PROGRESS_TOKENS = (
    "Loading weights:",
    "Materializing param=",
)


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
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local embedding model '{self.model_name}' "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        filtered_stdout = self._filter_expected_bootstrap_output(stdout_buffer.getvalue())
        filtered_stderr = self._filter_expected_bootstrap_output(stderr_buffer.getvalue())
        unexpected_output = f"{filtered_stdout}{filtered_stderr}"
        if unexpected_output:
            sys.stderr.write(unexpected_output)
        return self._model

    @staticmethod
    def _filter_expected_bootstrap_output(output_text: str) -> str:
        """Suppress expected bootstrap chatter while preserving actionable diagnostics."""
        if not output_text:
            return ""
        kept_lines: list[str] = []
        for line in output_text.splitlines(keepends=True):
            stripped = line.lstrip()
            if not stripped.strip():
                continue
            if all(token in stripped for token in _EXPECTED_WRAPPER_WARNING_TOKENS):
                continue
            if any(token in stripped for token in _EXPECTED_BOOTSTRAP_PROGRESS_TOKENS):
                continue
            kept_lines.append(line)
        return "".join(kept_lines)

    @staticmethod
    def _filter_expected_wrapper_warning(stderr_text: str) -> str:
        """Suppress only the expected sentence-transformers wrapper bootstrap line."""
        return LocalEmbeddingProvider._filter_expected_bootstrap_output(stderr_text)

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
