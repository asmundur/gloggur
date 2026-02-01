from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
from typing import Iterable, List, Optional

from gloggur.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._fallback_dimension = 256
        self._fallback_marker = Path(self.cache_dir or ".gloggur-cache") / ".local_embedding_fallback"
        env_flag = os.getenv("GLOGGUR_LOCAL_FALLBACK")
        if env_flag is None:
            self._use_fallback = self._fallback_marker.exists()
        else:
            self._use_fallback = env_flag.strip().lower() not in ("", "0", "false", "no", "off")
        self._token_pattern = re.compile(r"[A-Za-z0-9_]+")

    def _load_model(self):
        if self._use_fallback:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            self._enable_fallback()
            self._model = None
            return None
        try:
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            return self._model
        except Exception:
            self._enable_fallback()
            self._model = None
            return None

    def _enable_fallback(self) -> None:
        self._use_fallback = True
        self._fallback_marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fallback_marker.touch(exist_ok=True)
        except OSError:
            pass

    def embed_text(self, text: str) -> List[float]:
        model = self._load_model()
        if model is None:
            return self._tokenized_embedding(text)
        vector = model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        model = self._load_model()
        text_list = list(texts)
        if model is None:
            return [self._tokenized_embedding(text) for text in text_list]
        vectors = model.encode(text_list, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def get_dimension(self) -> int:
        model = self._load_model()
        if model is None:
            return self._fallback_dimension
        return model.get_sentence_embedding_dimension()

    def _tokenized_embedding(self, text: str) -> List[float]:
        tokens = self._token_pattern.findall(text.lower())
        if not tokens:
            tokens = [text]
        values = [0.0] * self._fallback_dimension
        for token in tokens:
            token_vector = self._vector_from_seed(token.encode("utf8"))
            for idx, token_value in enumerate(token_vector):
                values[idx] += token_value
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def _vector_from_seed(self, seed: bytes) -> List[float]:
        values: List[float] = []
        counter = 0
        while len(values) < self._fallback_dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(digest), 4):
                if len(values) >= self._fallback_dimension:
                    break
                chunk = digest[i : i + 4]
                if len(chunk) < 4:
                    continue
                number = int.from_bytes(chunk, "big")
                values.append((number / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        return values
