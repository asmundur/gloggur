from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

import yaml

DEFAULT_REPO_CONFIG_TRUST_MODE = "auto"
REPO_CONFIG_TRUST_MODES = frozenset({"auto", "trusted", "untrusted"})
ALLOW_CUSTOM_EMBEDDING_ENDPOINTS_ENV = "GLOGGUR_ALLOW_CUSTOM_EMBEDDING_ENDPOINTS"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
REMOTE_EMBEDDING_PROVIDER_IDS = frozenset({"openai", "gemini"})


def _optional_string(value: str | None) -> str | None:
    """Normalize optional strings by trimming whitespace and collapsing empty values."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def normalize_base_url(value: str | None) -> str | None:
    """Return a normalized base URL without a trailing slash."""
    normalized = _optional_string(value)
    if normalized is None:
        return None
    return normalized.rstrip("/")


def embedding_endpoint_host(value: str | None) -> str | None:
    """Extract hostname from an endpoint URL when present."""
    normalized = normalize_base_url(value)
    if normalized is None:
        return None
    parsed = urlparse(normalized)
    return parsed.hostname


def custom_embedding_endpoints_allowed() -> bool:
    """Return whether operator env explicitly allows custom embedding endpoints."""
    raw_value = os.environ.get(ALLOW_CUSTOM_EMBEDDING_ENDPOINTS_ENV)
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def is_custom_embedding_endpoint(
    value: str | None,
    *,
    allowed_defaults: tuple[str, ...],
) -> bool:
    """Return True when the endpoint differs from the built-in provider defaults."""
    normalized = normalize_base_url(value)
    if normalized is None:
        return False
    allowed = {normalize_base_url(item) for item in allowed_defaults}
    return normalized not in allowed


@dataclass
class GloggurConfig:
    """Config dataclass for embeddings, indexing paths, and docstring audit thresholds."""

    embedding_provider: str = "local"
    embed_graph_edges: bool = False
    local_embedding_model: str = "microsoft/codebert-base"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None
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
        default_factory=lambda: [
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".c",
            ".h",
            ".cpp",
            ".cc",
            ".cxx",
            ".hpp",
            ".hh",
            ".hxx",
            ".rs",
            ".go",
            ".java",
        ]
    )
    parser_extension_map: dict[str, str] = field(default_factory=dict)
    include_minified_js: bool = False
    excluded_dirs: list[str] = field(
        default_factory=lambda: [
            ".git",
            "node_modules",
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
    extract_symbols_timeout_seconds: float = 60.0
    config_source: str = "defaults"
    config_source_path: str | None = None
    config_trust_mode: str = DEFAULT_REPO_CONFIG_TRUST_MODE
    security_warning_codes: list[str] = field(default_factory=list)

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
        return f"{provider}:{model}|embed_graph_edges={int(self.embed_graph_edges)}"

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

    def is_remote_embedding_provider(self) -> bool:
        """Return whether the active provider can send embeddings to a remote service."""
        provider = _optional_string(self.embedding_provider)
        return provider in REMOTE_EMBEDDING_PROVIDER_IDS

    def resolved_openai_base_url(self) -> str:
        """Return the effective base URL used by the OpenAI-compatible provider."""
        resolved = normalize_base_url(self.openai_base_url)
        if resolved:
            return resolved
        openrouter_key = _optional_string(self.openrouter_api_key)
        if openrouter_key:
            return normalize_base_url(self.openrouter_base_url) or OPENROUTER_DEFAULT_BASE_URL
        return OPENAI_DEFAULT_BASE_URL

    def remote_embedding_destination(self) -> dict[str, str] | None:
        """Return remote embedding provider/host diagnostics when remote embeddings are active."""
        provider = _optional_string(self.embedding_provider)
        if provider == "openai":
            base_url = self.resolved_openai_base_url()
            host = embedding_endpoint_host(base_url) or "api.openai.com"
            return {
                "provider": provider,
                "host": host,
            }
        if provider == "gemini":
            return {
                "provider": provider,
                "host": "gemini.googleapis.com",
            }
        return None

    def requests_custom_embedding_endpoint(self) -> bool:
        """Return whether the active remote embedding provider points at a non-default endpoint."""
        provider = _optional_string(self.embedding_provider)
        if provider != "openai":
            return False
        return is_custom_embedding_endpoint(
            self.resolved_openai_base_url(),
            allowed_defaults=(OPENAI_DEFAULT_BASE_URL, OPENROUTER_DEFAULT_BASE_URL),
        )

    @classmethod
    def load(
        cls,
        path: str | None = None,
        overrides: dict[str, object] | None = None,
        trust_mode: str = DEFAULT_REPO_CONFIG_TRUST_MODE,
    ) -> GloggurConfig:
        """Load config from file/env (yaml/json) and apply overrides."""
        if trust_mode not in REPO_CONFIG_TRUST_MODES:
            raise ValueError(f"Unsupported repo config trust mode: {trust_mode}")

        resolved_path = path
        config_source = "explicit" if path else "defaults"
        if resolved_path is None:
            for candidate in (".gloggur.yaml", ".gloggur.yml", ".gloggur.json"):
                if os.path.exists(candidate):
                    resolved_path = candidate
                    config_source = "auto_discovered"
                    break

        file_data: dict[str, object] = {}
        if resolved_path:
            file_data.update(cls._load_file(resolved_path))

        dotenv_values = cls._load_dotenv()
        env_data = cls._extract_env_config(dict(os.environ))
        merged_env_values = dict(dotenv_values)
        merged_env_values.update(os.environ)
        effective_env_data = cls._extract_env_config(merged_env_values)
        effective_dotenv_data = {
            key: value for key, value in effective_env_data.items() if key not in env_data
        }
        data: dict[str, object] = {}
        data.update(file_data)
        data.update(effective_env_data)
        if overrides:
            data.update(overrides)
        if config_source == "defaults" and effective_dotenv_data:
            config_source = "repo_dotenv"

        config = cls(**data)
        config.config_source = config_source
        config.config_source_path = os.path.abspath(resolved_path) if resolved_path else None
        config.config_trust_mode = trust_mode
        config.security_warning_codes = cls._compute_security_warning_codes(
            config=config,
            config_source=config_source,
            trust_mode=trust_mode,
            file_data=file_data,
            dotenv_data=effective_dotenv_data,
            env_data=env_data,
            overrides=overrides or {},
            config_path_present=resolved_path is not None,
        )
        return config

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
        env_values: dict[str, str] = GloggurConfig._load_dotenv()
        env_values.update(os.environ)
        return GloggurConfig._extract_env_config(env_values)

    @staticmethod
    def _extract_env_config(env_values: dict[str, str]) -> dict[str, object]:
        """Map merged environment values into config keys."""
        data: dict[str, object] = {}

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
        embed_graph_edges = _env_bool("GLOGGUR_EMBED_GRAPH_EDGES")
        if embed_graph_edges is not None:
            data["embed_graph_edges"] = embed_graph_edges
        if _env_value("GLOGGUR_LOCAL_MODEL"):
            data["local_embedding_model"] = _env_value("GLOGGUR_LOCAL_MODEL")
        if _env_value("GLOGGUR_OPENAI_MODEL"):
            data["openai_embedding_model"] = _env_value("GLOGGUR_OPENAI_MODEL")
        if _env_value("OPENAI_API_KEY"):
            data["openai_api_key"] = _env_value("OPENAI_API_KEY")
        if _env_value("OPENAI_BASE_URL"):
            data["openai_base_url"] = _env_value("OPENAI_BASE_URL")
        if _env_value("OPENROUTER_API_KEY"):
            data["openrouter_api_key"] = _env_value("OPENROUTER_API_KEY")
        if _env_value("GLOGGUR_OPENROUTER_BASE_URL"):
            data["openrouter_base_url"] = _env_value("GLOGGUR_OPENROUTER_BASE_URL")
        if _env_value("GLOGGUR_OPENROUTER_SITE_URL"):
            data["openrouter_site_url"] = _env_value("GLOGGUR_OPENROUTER_SITE_URL")
        if _env_value("GLOGGUR_OPENROUTER_APP_NAME"):
            data["openrouter_app_name"] = _env_value("GLOGGUR_OPENROUTER_APP_NAME")
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
        include_minified_js = _env_bool("GLOGGUR_INCLUDE_MINIFIED_JS")
        if include_minified_js is not None:
            data["include_minified_js"] = include_minified_js
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
        if _env_value("GLOGGUR_EXTRACT_SYMBOLS_TIMEOUT_SECONDS"):
            try:
                data["extract_symbols_timeout_seconds"] = float(
                    _env_value("GLOGGUR_EXTRACT_SYMBOLS_TIMEOUT_SECONDS") or "60"
                )
            except ValueError:
                pass
        return data

    @staticmethod
    def _compute_security_warning_codes(
        *,
        config: GloggurConfig,
        config_source: str,
        trust_mode: str,
        file_data: dict[str, object],
        dotenv_data: dict[str, object],
        env_data: dict[str, object],
        overrides: dict[str, object],
        config_path_present: bool,
    ) -> list[str]:
        """Compute stable security warning codes from resolved config provenance."""

        def _final_source_for_key(key: str) -> str:
            """Return the highest-precedence source category for one resolved config key."""
            if key in overrides:
                return "override"
            if key in env_data:
                return "env"
            if key in dotenv_data:
                return "dotenv"
            if key in file_data:
                return "config"
            return "default"

        def _has_adapter_overrides(adapter_map: object) -> bool:
            """Return True when the adapters mapping includes any explicit module override."""
            if not isinstance(adapter_map, dict):
                return False
            for category_map in adapter_map.values():
                if not isinstance(category_map, dict):
                    continue
                for value in category_map.values():
                    if isinstance(value, str) and value.strip():
                        return True
            return False

        warning_codes: set[str] = set()
        untrusted_repo_config = False
        if trust_mode != "trusted":
            if config_source in {"auto_discovered", "repo_dotenv"}:
                untrusted_repo_config = True
            elif trust_mode == "untrusted" and (config_path_present or dotenv_data):
                untrusted_repo_config = True
        if untrusted_repo_config:
            warning_codes.add("untrusted_repo_config")
        if untrusted_repo_config and _has_adapter_overrides(file_data.get("adapters")):
            warning_codes.add("untrusted_adapter_override_requested")
        if (
            untrusted_repo_config
            and config.is_remote_embedding_provider()
            and _final_source_for_key("embedding_provider") in {"config", "dotenv"}
        ):
            warning_codes.add("untrusted_remote_provider_requested")
        if config.requests_custom_embedding_endpoint():
            warning_codes.add("custom_embedding_endpoint_requested")
        return sorted(warning_codes)
