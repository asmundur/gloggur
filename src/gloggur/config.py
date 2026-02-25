from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class GloggurConfig:
    """Config dataclass for embeddings, indexing paths, and docstring audit thresholds."""

    embedding_provider: str = "local"
    local_embedding_model: str = "microsoft/codebert-base"
    openai_embedding_model: str = "text-embedding-3-large"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_api_key: Optional[str] = None
    cache_dir: str = ".gloggur-cache"
    model_cache_dir: Optional[str] = None
    watch_enabled: bool = False
    watch_path: str = "."
    watch_debounce_ms: int = 300
    watch_mode: str = "daemon"
    watch_state_file: str = ".gloggur-cache/watch_state.json"
    watch_pid_file: str = ".gloggur-cache/watch.pid"
    watch_log_file: str = ".gloggur-cache/watch.log"
    docstring_semantic_threshold: float = 0.2
    docstring_semantic_min_chars: int = 0
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
            "htmlcov",
        ]
    )
    index_version: str = "1"

    def embedding_profile(self) -> str:
        """Return a stable profile string for the active embedding configuration."""
        provider = self.embedding_provider
        if provider == "local":
            model = self.local_embedding_model
        elif provider == "openai":
            model = self.openai_embedding_model
        elif provider == "gemini":
            model = self.gemini_embedding_model
        else:
            model = "unknown"
        return f"{provider}:{model}"

    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        overrides: Optional[Dict[str, object]] = None,
    ) -> "GloggurConfig":
        """Load config from file/env (yaml/json) and apply overrides."""
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
        """Load config values from a JSON or YAML file path."""
        with open(path, "r", encoding="utf8") as handle:
            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(handle) or {}
            return json.load(handle)

    @staticmethod
    def _load_env() -> Dict[str, object]:
        """Load config values from GLOGGUR_* environment variables."""
        data: Dict[str, object] = {}

        def _env_bool(name: str) -> Optional[bool]:
            value = os.getenv(name)
            if value is None:
                return None
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return None

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
        watch_enabled = _env_bool("GLOGGUR_WATCH_ENABLED")
        if watch_enabled is not None:
            data["watch_enabled"] = watch_enabled
        if os.getenv("GLOGGUR_WATCH_PATH"):
            data["watch_path"] = os.getenv("GLOGGUR_WATCH_PATH")
        if os.getenv("GLOGGUR_WATCH_MODE"):
            data["watch_mode"] = os.getenv("GLOGGUR_WATCH_MODE")
        if os.getenv("GLOGGUR_WATCH_DEBOUNCE_MS"):
            try:
                data["watch_debounce_ms"] = int(
                    os.getenv("GLOGGUR_WATCH_DEBOUNCE_MS", "300")
                )
            except ValueError:
                pass
        if os.getenv("GLOGGUR_WATCH_STATE_FILE"):
            data["watch_state_file"] = os.getenv("GLOGGUR_WATCH_STATE_FILE")
        if os.getenv("GLOGGUR_WATCH_PID_FILE"):
            data["watch_pid_file"] = os.getenv("GLOGGUR_WATCH_PID_FILE")
        if os.getenv("GLOGGUR_WATCH_LOG_FILE"):
            data["watch_log_file"] = os.getenv("GLOGGUR_WATCH_LOG_FILE")
        if os.getenv("GLOGGUR_DOCSTRING_SEMANTIC_MIN_CHARS"):
            try:
                data["docstring_semantic_min_chars"] = int(
                    os.getenv("GLOGGUR_DOCSTRING_SEMANTIC_MIN_CHARS", "0")
                )
            except ValueError:
                pass
        if os.getenv("GLOGGUR_CACHE_DIR"):
            data["cache_dir"] = os.getenv("GLOGGUR_CACHE_DIR")
        return data
