from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class GloggurConfig:
    embedding_provider: str = "local"
    local_embedding_model: str = "microsoft/codebert-base"
    openai_embedding_model: str = "text-embedding-3-large"
    cache_dir: str = ".gloggur-cache"
    model_cache_dir: Optional[str] = None
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
        with open(path, "r", encoding="utf8") as handle:
            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(handle) or {}
            return json.load(handle)

    @staticmethod
    def _load_env() -> Dict[str, object]:
        data: Dict[str, object] = {}
        if os.getenv("GLOGGUR_EMBEDDING_PROVIDER"):
            data["embedding_provider"] = os.getenv("GLOGGUR_EMBEDDING_PROVIDER")
        if os.getenv("GLOGGUR_LOCAL_MODEL"):
            data["local_embedding_model"] = os.getenv("GLOGGUR_LOCAL_MODEL")
        if os.getenv("GLOGGUR_OPENAI_MODEL"):
            data["openai_embedding_model"] = os.getenv("GLOGGUR_OPENAI_MODEL")
        if os.getenv("GLOGGUR_CACHE_DIR"):
            data["cache_dir"] = os.getenv("GLOGGUR_CACHE_DIR")
        return data
