from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import yaml


@dataclass
class GloggurConfig:
    """Config dataclass for embeddings, indexing paths, and docstring audit thresholds."""

    embedding_provider: str = "local"
    local_embedding_model: str = "microsoft/codebert-base"
    openai_embedding_model: str = "text-embedding-3-large"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_api_key: str | None = None
    cache_dir: str = ".gloggur-cache"
    model_cache_dir: str | None = None
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
    docstring_semantic_kind_thresholds: dict[str, float] | None = field(
        default_factory=lambda: {"class": 0.05, "interface": 0.05}
    )
    supported_extensions: list[str] = field(
        default_factory=lambda: [".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java"]
    )
    parser_extension_map: dict[str, str] = field(default_factory=dict)
    excluded_dirs: list[str] = field(
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
    adapters: dict[str, object] = field(
        default_factory=lambda: {
            "parsers": {},
            "coverage_importers": {},
            "embedding_providers": {},
            "storage": {},
            "runtime": {},
        }
    )
    storage: dict[str, str] = field(default_factory=lambda: {"backend": "sqlite_faiss"})
    runtime: dict[str, str] = field(default_factory=lambda: {"host": "python_local"})
    index_version: str = "1"
    max_symbol_chunk_bytes: int = 12000
    max_symbol_chunk_tokens: int = 2000

    def embedding_profile(self) -> str:
        """Return a stable profile string for the active embedding configuration."""
        provider = self.embedding_provider
        if provider == "local":
            model = self.local_embedding_model
        elif provider == "openai":
            model = self.openai_embedding_model
        elif provider == "gemini":
            model = self.gemini_embedding_model
        elif provider == "test":
            model = self.local_embedding_model
        else:
            model = "unknown"
        return f"{provider}:{model}"

    def adapter_module_override(self, category: str, adapter_id: str) -> str | None:
        """Return optional module-path override for one adapter category/id pair."""
        category_map = self.adapters.get(category) if isinstance(self.adapters, dict) else None
        if not isinstance(category_map, dict):
            return None
        value = category_map.get(adapter_id)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def storage_backend(self) -> str:
        """Return configured storage backend id with compatibility default."""
        value = self.storage.get("backend") if isinstance(self.storage, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "sqlite_faiss"

    def runtime_host(self) -> str:
        """Return configured runtime host id with compatibility default."""
        value = self.runtime.get("host") if isinstance(self.runtime, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "python_local"

    @classmethod
    def load(
        cls,
        path: str | None = None,
        overrides: dict[str, object] | None = None,
    ) -> GloggurConfig:
        """Load config from file/env (yaml/json) and apply overrides."""
        data: dict[str, object] = {}
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
    def _load_file(path: str) -> dict[str, object]:
        """Load config values from a JSON or YAML file path."""
        with open(path, encoding="utf8") as handle:
            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(handle) or {}
            return json.load(handle)

    @staticmethod
    def _load_dotenv(path: str = ".env") -> dict[str, str]:
        """Load dotenv-style key/value pairs from ``path`` when present."""
        if not os.path.exists(path):
            return {}
        data: dict[str, str] = {}
        with open(path, encoding="utf8") as handle:
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
                if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
                    value = value[1:-1]
                data[key] = value
        return data

    @staticmethod
    def _load_env() -> dict[str, object]:
        """Load config values from GLOGGUR_* environment variables."""
        data: dict[str, object] = {}
        env_values: dict[str, str] = GloggurConfig._load_dotenv()
        env_values.update(os.environ)

        def _env_value(name: str) -> str | None:
            """Return a non-empty environment value or None when unset/blank."""
            value = env_values.get(name)
            if value is None or value == "":
                return None
            return value

        def _env_bool(name: str) -> bool | None:
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
                data["watch_debounce_ms"] = int(_env_value("GLOGGUR_WATCH_DEBOUNCE_MS") or "300")
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
        if _env_value("GLOGGUR_STORAGE_BACKEND"):
            data["storage"] = {"backend": _env_value("GLOGGUR_STORAGE_BACKEND")}
        if _env_value("GLOGGUR_RUNTIME_HOST"):
            data["runtime"] = {"host": _env_value("GLOGGUR_RUNTIME_HOST")}
        if _env_value("GLOGGUR_MAX_SYMBOL_CHUNK_BYTES"):
            try:
                data["max_symbol_chunk_bytes"] = int(
                    _env_value("GLOGGUR_MAX_SYMBOL_CHUNK_BYTES") or "12000"
                )
            except ValueError:
                pass
        if _env_value("GLOGGUR_MAX_SYMBOL_CHUNK_TOKENS"):
            try:
                data["max_symbol_chunk_tokens"] = int(
                    _env_value("GLOGGUR_MAX_SYMBOL_CHUNK_TOKENS") or "2000"
                )
            except ValueError:
                pass
        return data
