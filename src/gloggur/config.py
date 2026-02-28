from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    # Calibrated for microsoft/codebert-base: median doc-code cosine similarity
    # is ~0.135, so threshold=0.2 flags >50% of symbols (noise, not signal).
    # Threshold=0.10 flags only symbols below the 38th percentile — a defensible
    # "clearly low" signal for this model family.
    docstring_semantic_threshold: float = 0.10
    docstring_semantic_min_chars: int = 0
    docstring_semantic_max_chars: int = 4000
    # Minimum code-body character count (after stripping the docstring) required
    # before semantic scoring is attempted.  Short bodies (<30 chars) produce
    # unreliable cosine similarity signals and are skipped by default.
    docstring_semantic_min_code_chars: int = 30
    # Per-symbol-kind threshold overrides.  ``None`` means the global
    # ``docstring_semantic_threshold`` is used for every kind.
    # Classes and interfaces carry deliberately high-level docstrings; their
    # calibrated threshold is half the global value.
    docstring_semantic_kind_thresholds: Optional[Dict[str, float]] = field(
        default_factory=lambda: {"class": 0.05, "interface": 0.05}
    )
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
    def _load_dotenv(path: str = ".env") -> Dict[str, str]:
        """Load dotenv-style key/value pairs from ``path`` when present."""
        if not os.path.exists(path):
            return {}
        data: Dict[str, str] = {}
        with open(path, "r", encoding="utf8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or any(char.isspace() for char in key):
                    continue
                if (
                    len(value) >= 2
                    and value[0] in {"'", '"'}
                    and value[-1] == value[0]
                ):
                    value = value[1:-1]
                data[key] = value
        return data

    @staticmethod
    def _load_env() -> Dict[str, object]:
        """Load config values from GLOGGUR_* environment variables."""
        data: Dict[str, object] = {}
        env_values: Dict[str, str] = GloggurConfig._load_dotenv()
        env_values.update(os.environ)

        def _env_value(name: str) -> Optional[str]:
            """Return a non-empty environment value or None when unset/blank."""
            value = env_values.get(name)
            if value is None or value == "":
                return None
            return value

        def _env_bool(name: str) -> Optional[bool]:
            """Parse common truthy/falsey strings from the merged environment map."""
            value = _env_value(name)
            if value is None:
                return None
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return None

        if _env_value("GLOGGUR_EMBEDDING_PROVIDER"):
            data["embedding_provider"] = _env_value("GLOGGUR_EMBEDDING_PROVIDER")
        if _env_value("GLOGGUR_LOCAL_MODEL"):
            data["local_embedding_model"] = _env_value("GLOGGUR_LOCAL_MODEL")
        if _env_value("GLOGGUR_OPENAI_MODEL"):
            data["openai_embedding_model"] = _env_value("GLOGGUR_OPENAI_MODEL")
        if _env_value("GLOGGUR_GEMINI_MODEL"):
            data["gemini_embedding_model"] = _env_value("GLOGGUR_GEMINI_MODEL")
        if _env_value("GLOGGUR_GEMINI_API_KEY"):
            data["gemini_api_key"] = _env_value("GLOGGUR_GEMINI_API_KEY")
        watch_enabled = _env_bool("GLOGGUR_WATCH_ENABLED")
        if watch_enabled is not None:
            data["watch_enabled"] = watch_enabled
        if _env_value("GLOGGUR_WATCH_PATH"):
            data["watch_path"] = _env_value("GLOGGUR_WATCH_PATH")
        if _env_value("GLOGGUR_WATCH_MODE"):
            data["watch_mode"] = _env_value("GLOGGUR_WATCH_MODE")
        if _env_value("GLOGGUR_WATCH_DEBOUNCE_MS"):
            try:
                data["watch_debounce_ms"] = int(
                    _env_value("GLOGGUR_WATCH_DEBOUNCE_MS") or "300"
                )
            except ValueError:
                pass
        if _env_value("GLOGGUR_WATCH_STATE_FILE"):
            data["watch_state_file"] = _env_value("GLOGGUR_WATCH_STATE_FILE")
        if _env_value("GLOGGUR_WATCH_PID_FILE"):
            data["watch_pid_file"] = _env_value("GLOGGUR_WATCH_PID_FILE")
        if _env_value("GLOGGUR_WATCH_LOG_FILE"):
            data["watch_log_file"] = _env_value("GLOGGUR_WATCH_LOG_FILE")
        if _env_value("GLOGGUR_DOCSTRING_SEMANTIC_MIN_CHARS"):
            try:
                data["docstring_semantic_min_chars"] = int(
                    _env_value("GLOGGUR_DOCSTRING_SEMANTIC_MIN_CHARS") or "0"
                )
            except ValueError:
                pass
        if _env_value("GLOGGUR_CACHE_DIR"):
            data["cache_dir"] = _env_value("GLOGGUR_CACHE_DIR")
        return data
