from __future__ import annotations

from typing import Iterable, List, Optional

from gloggur.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for local embeddings") from exc
        self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        model = self._load_model()
        vector = model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        model = self._load_model()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def get_dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
