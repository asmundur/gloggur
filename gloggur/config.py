from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class GloggurConfig:
    """Configuration for indexing, embeddings, and validation."""

    embedding_provider: str = "local"
    local_embedding_model: str = "microsoft/codebert-base"
    openai_embedding_model: str = "text-embedding-3-large"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_api_key: Optional[str] = None
    cache_dir: str = ".gloggur-cache"
    model_cache_dir: Optional[str] = None
    docstring_semantic_threshold: float = 0.2
    docstring_semantic_max_chars: int = 4000
    supported_extensions: List[str] = field(
        default_factory=lambda: [".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java"]
    )
    excluded_dirs: List[str] = field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "venv",
            ".venv",
            ".gloggur-cache",
            "dist",
            "build",
        ]
    )
    index_version: str = "1"

    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        overrides: Optional[Dict[str, object]] = None,
    ) -> "GloggurConfig":
        """Load configuration from file/env with optional overrides."""
        data: Dict[str, object] = {}
        if path:
            data.update(cls._load_file(path))
        else:
            for candidate in (".gloggur.yaml", ".gloggur.yml", ".gloggur.json"):
                if os.path.exists(candidate):
                    data.update(cls._load_file(candidate))
                    break
        data.update(cls._load_env())
        if overrides:
            data.update(overrides)
        return cls(**data)

    @staticmethod
    def _load_file(path: str) -> Dict[str, object]:
        """Load configuration values from a JSON or YAML file."""
        with open(path, "r", encoding="utf8") as handle:
            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(handle) or {}
            return json.load(handle)

    @staticmethod
    def _load_env() -> Dict[str, object]:
        """Load configuration values from environment variables."""
        data: Dict[str, object] = {}
        if os.getenv("GLOGGUR_EMBEDDING_PROVIDER"):
            data["embedding_provider"] = os.getenv("GLOGGUR_EMBEDDING_PROVIDER")
        if os.getenv("GLOGGUR_LOCAL_MODEL"):
            data["local_embedding_model"] = os.getenv("GLOGGUR_LOCAL_MODEL")
        if os.getenv("GLOGGUR_OPENAI_MODEL"):
            data["openai_embedding_model"] = os.getenv("GLOGGUR_OPENAI_MODEL")
        if os.getenv("GLOGGUR_GEMINI_MODEL"):
            data["gemini_embedding_model"] = os.getenv("GLOGGUR_GEMINI_MODEL")
        if os.getenv("GLOGGUR_GEMINI_API_KEY"):
            data["gemini_api_key"] = os.getenv("GLOGGUR_GEMINI_API_KEY")
        if os.getenv("GLOGGUR_CACHE_DIR"):
            data["cache_dir"] = os.getenv("GLOGGUR_CACHE_DIR")
        return data
