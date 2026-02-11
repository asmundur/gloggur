from __future__ import annotations

import json

from gloggur.config import GloggurConfig


def test_load_from_yaml_and_env_overrides(tmp_path, monkeypatch) -> None:
    """Config loads YAML and applies env overrides."""
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


def test_embedding_profile_uses_active_provider_model() -> None:
    """Embedding profile should encode active provider and its configured model."""
    local = GloggurConfig(embedding_provider="local", local_embedding_model="local-a")
    openai = GloggurConfig(embedding_provider="openai", openai_embedding_model="openai-a")
    gemini = GloggurConfig(embedding_provider="gemini", gemini_embedding_model="gemini-a")

    assert local.embedding_profile() == "local:local-a"
    assert openai.embedding_profile() == "openai:openai-a"
    assert gemini.embedding_profile() == "gemini:gemini-a"
