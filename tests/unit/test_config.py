from __future__ import annotations

import json

from gloggur.config import GloggurConfig


def test_load_from_yaml_and_env_overrides(tmp_path, monkeypatch) -> None:
    """Config loads YAML and applies env overrides."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "embedding_provider: openai\ncache_dir: file-cache\nopenai_embedding_model: model-a\n",
        encoding="utf8",
    )
    monkeypatch.setenv("GLOGGUR_CACHE_DIR", "env-cache")

    config = GloggurConfig.load(path=str(config_path))

    assert config.embedding_provider == "openai"
    assert config.openai_embedding_model == "model-a"
    assert config.cache_dir == "env-cache"


def test_load_auto_detect_json_and_overrides(tmp_path, monkeypatch) -> None:
    """Config auto-detects JSON and applies overrides."""
    config_path = tmp_path / ".gloggur.json"
    config_path.write_text(json.dumps({"embedding_provider": "gemini"}), encoding="utf8")
    monkeypatch.chdir(tmp_path)

    config = GloggurConfig.load(overrides={"embedding_provider": "local"})

    assert config.embedding_provider == "local"


def test_load_env_values(monkeypatch) -> None:
    """Config loads values from environment variables."""
    monkeypatch.setenv("GLOGGUR_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("GLOGGUR_LOCAL_MODEL", "local-model")
    monkeypatch.setenv("GLOGGUR_OPENAI_MODEL", "openai-model")
    monkeypatch.setenv("GLOGGUR_GEMINI_MODEL", "gemini-model")
    monkeypatch.setenv("GLOGGUR_GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GLOGGUR_CACHE_DIR", "cache-dir")

    config = GloggurConfig.load(path=None)

    assert config.embedding_provider == "openai"
    assert config.local_embedding_model == "local-model"
    assert config.openai_embedding_model == "openai-model"
    assert config.gemini_embedding_model == "gemini-model"
    assert config.gemini_api_key == "gemini-key"
    assert config.cache_dir == "cache-dir"


def test_load_env_watch_values(monkeypatch) -> None:
    """Config loads watch settings from environment variables."""
    monkeypatch.setenv("GLOGGUR_WATCH_ENABLED", "true")
    monkeypatch.setenv("GLOGGUR_WATCH_PATH", "/tmp/repo")
    monkeypatch.setenv("GLOGGUR_WATCH_MODE", "foreground")
    monkeypatch.setenv("GLOGGUR_WATCH_DEBOUNCE_MS", "150")
    monkeypatch.setenv("GLOGGUR_WATCH_STATE_FILE", "/tmp/state.json")
    monkeypatch.setenv("GLOGGUR_WATCH_PID_FILE", "/tmp/watch.pid")
    monkeypatch.setenv("GLOGGUR_WATCH_LOG_FILE", "/tmp/watch.log")

    config = GloggurConfig.load(path=None)

    assert config.watch_enabled is True
    assert config.watch_path == "/tmp/repo"
    assert config.watch_mode == "foreground"
    assert config.watch_debounce_ms == 150
    assert config.watch_state_file == "/tmp/state.json"
    assert config.watch_pid_file == "/tmp/watch.pid"
    assert config.watch_log_file == "/tmp/watch.log"


def test_load_env_invalid_watch_debounce_keeps_default(monkeypatch) -> None:
    """Invalid debounce env should not override the default value."""
    monkeypatch.setenv("GLOGGUR_WATCH_DEBOUNCE_MS", "not-a-number")

    config = GloggurConfig.load(path=None)

    assert config.watch_debounce_ms == 300


def test_embedding_profile_uses_active_provider_model() -> None:
    """Embedding profile should encode active provider and its configured model."""
    local = GloggurConfig(embedding_provider="local", local_embedding_model="local-a")
    openai = GloggurConfig(embedding_provider="openai", openai_embedding_model="openai-a")
    gemini = GloggurConfig(embedding_provider="gemini", gemini_embedding_model="gemini-a")
    test = GloggurConfig(embedding_provider="test", local_embedding_model="test-a")

    assert local.embedding_profile() == "local:local-a"
    assert openai.embedding_profile() == "openai:openai-a"
    assert gemini.embedding_profile() == "gemini:gemini-a"
    assert test.embedding_profile() == "test:test-a"


def test_load_env_reads_dotenv_when_process_env_unset(tmp_path, monkeypatch) -> None:
    """Config should read embedding settings from local .env when process env is unset."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GLOGGUR_EMBEDDING_PROVIDER=gemini\n"
        "GLOGGUR_GEMINI_MODEL=dotenv-gemini-model\n"
        "GLOGGUR_GEMINI_API_KEY=dotenv-gemini-key\n",
        encoding="utf8",
    )
    monkeypatch.delenv("GLOGGUR_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("GLOGGUR_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GLOGGUR_GEMINI_API_KEY", raising=False)

    config = GloggurConfig.load(path=None)

    assert config.embedding_provider == "gemini"
    assert config.gemini_embedding_model == "dotenv-gemini-model"
    assert config.gemini_api_key == "dotenv-gemini-key"


def test_process_env_overrides_dotenv(tmp_path, monkeypatch) -> None:
    """Process environment values should override .env values."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GLOGGUR_EMBEDDING_PROVIDER=gemini\n"
        "GLOGGUR_GEMINI_MODEL=dotenv-model\n"
        "GLOGGUR_GEMINI_API_KEY=dotenv-key\n",
        encoding="utf8",
    )
    monkeypatch.setenv("GLOGGUR_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("GLOGGUR_OPENAI_MODEL", "openai-env-model")
    monkeypatch.setenv("GLOGGUR_GEMINI_MODEL", "gemini-env-model")
    monkeypatch.setenv("GLOGGUR_GEMINI_API_KEY", "gemini-env-key")

    config = GloggurConfig.load(path=None)

    assert config.embedding_provider == "openai"
    assert config.openai_embedding_model == "openai-env-model"
    assert config.gemini_embedding_model == "gemini-env-model"
    assert config.gemini_api_key == "gemini-env-key"


def test_dotenv_model_mapping_for_gemini(tmp_path, monkeypatch) -> None:
    """Gemini model from .env should flow into embedding profile computation."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GLOGGUR_EMBEDDING_PROVIDER=gemini\n"
        "GLOGGUR_GEMINI_MODEL=sentinel-gemini-model\n",
        encoding="utf8",
    )
    monkeypatch.delenv("GLOGGUR_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("GLOGGUR_GEMINI_MODEL", raising=False)

    config = GloggurConfig.load(path=None)

    assert config.embedding_provider == "gemini"
    assert config.gemini_embedding_model == "sentinel-gemini-model"
    assert config.embedding_profile() == "gemini:sentinel-gemini-model"


def test_dotenv_ignores_malformed_lines(tmp_path, monkeypatch) -> None:
    """Malformed .env entries should be skipped instead of breaking config load."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "NOT_AN_ASSIGNMENT\n"
        " =missing_key\n"
        "BAD KEY=bad\n"
        "export GLOGGUR_CACHE_DIR=dotenv-cache\n",
        encoding="utf8",
    )
    monkeypatch.delenv("GLOGGUR_CACHE_DIR", raising=False)

    config = GloggurConfig.load(path=None)

    assert config.cache_dir == "dotenv-cache"
    assert config.embedding_provider == "local"


def test_dotenv_empty_values_do_not_override(tmp_path, monkeypatch) -> None:
    """Empty .env values should behave like unset variables."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GLOGGUR_EMBEDDING_PROVIDER=\n"
        "GLOGGUR_GEMINI_MODEL=\n"
        "GLOGGUR_GEMINI_API_KEY=\n",
        encoding="utf8",
    )
    monkeypatch.delenv("GLOGGUR_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("GLOGGUR_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GLOGGUR_GEMINI_API_KEY", raising=False)

    config = GloggurConfig.load(path=None)

    assert config.embedding_provider == "local"
    assert config.gemini_embedding_model == "gemini-embedding-001"
    assert config.gemini_api_key is None
