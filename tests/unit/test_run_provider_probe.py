from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gloggur.config import GloggurConfig
from scripts.run_provider_probe import test_openai_embeddings as _probe_openai_embeddings
from scripts.run_provider_probe import _vector_store_stats
from scripts.run_provider_probe import test_gemini_embeddings as _probe_gemini_embeddings


class _FakeTempDirectory:
    """Minimal context manager replacement for tempfile.TemporaryDirectory."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type, exc, tb
        return False


def _write_vector_files(cache_dir: Path, id_map_payload: object) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "vectors.index").write_text("placeholder", encoding="utf8")
    (cache_dir / "vectors.json").write_text(json.dumps(id_map_payload), encoding="utf8")


def test_vector_store_stats_accepts_schema_v2_id_map(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _write_vector_files(
        cache_dir,
        {
            "schema_version": 2,
            "next_vector_id": 3,
            "symbol_to_vector_id": {"symbol-1": 1, "symbol-2": 2},
            "fallback_order": ["symbol-1", "symbol-2"],
        },
    )

    ok, message, details = _vector_store_stats(str(cache_dir))

    assert ok is True
    assert message == "Vector store populated"
    assert details["id_count"] == 2


def test_vector_store_stats_rejects_schema_v2_missing_symbol_map(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _write_vector_files(
        cache_dir,
        {
            "schema_version": 2,
            "next_vector_id": 1,
        },
    )

    ok, message, details = _vector_store_stats(str(cache_dir))

    assert ok is False
    assert message == "Vector ID map missing symbol_to_vector_id"
    assert str(details["id_map_path"]).endswith("vectors.json")


def test_gemini_probe_skip_message_lists_all_supported_api_key_env_vars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("GLOGGUR_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = _probe_gemini_embeddings(tmp_path, GloggurConfig())

    assert result.status == "skipped"
    assert result.details == {
        "missing_env": "GLOGGUR_GEMINI_API_KEY or GEMINI_API_KEY or GOOGLE_API_KEY"
    }


def test_openai_probe_skip_message_lists_openrouter_and_openai_env_vars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = _probe_openai_embeddings(tmp_path, GloggurConfig())

    assert result.status == "skipped"
    assert result.details == {"missing_env": "OPENROUTER_API_KEY or OPENAI_API_KEY"}


def test_openai_probe_reports_credential_env_with_openrouter_precedence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    def _fake_run_provider_test(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            name="Test 2.2: OpenAI Embeddings",
            status="passed",
            message="ok",
            details={"indexed_files": 1},
        )

    monkeypatch.setattr("scripts.run_provider_probe._run_provider_test", _fake_run_provider_test)

    result = _probe_openai_embeddings(tmp_path, GloggurConfig())

    assert result.status == "passed"
    assert result.details is not None
    assert result.details["credential_env"] == "OPENROUTER_API_KEY"


def test_openai_probe_uses_isolated_temp_cache_for_fresh_indexing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    probe_cache_dir = tmp_path / "openai-probe-cache"
    captured: dict[str, object] = {}

    def _fake_tempdir(*args, **kwargs):  # type: ignore[no-untyped-def]
        prefix = kwargs.get("prefix")
        if prefix is None and args:
            prefix = args[0]
        captured["prefix"] = prefix
        return _FakeTempDirectory(str(probe_cache_dir))

    def _fake_run_provider_test(
        *,
        name: str,
        provider: str,
        query: str,
        cache_dir: str,
        runner,
        model_name: str | None = None,
    ) -> SimpleNamespace:
        captured["name"] = name
        captured["provider"] = provider
        captured["query"] = query
        captured["cache_dir"] = cache_dir
        captured["runner_env"] = dict(runner.env or {})
        captured["model_name"] = model_name
        return SimpleNamespace(
            name=name,
            status="passed",
            message="ok",
            details={"indexed_files": 1},
        )

    monkeypatch.setattr("scripts.run_provider_probe.tempfile.TemporaryDirectory", _fake_tempdir)
    monkeypatch.setattr("scripts.run_provider_probe._run_provider_test", _fake_run_provider_test)

    result = _probe_openai_embeddings(tmp_path, GloggurConfig())

    assert result.status == "passed"
    assert captured["prefix"] == "gloggur-phase2-openai-"
    assert captured["provider"] == "openai"
    assert captured["query"] == "vector store"
    assert captured["cache_dir"] == str(probe_cache_dir)
    assert isinstance(captured["runner_env"], dict)
    assert captured["runner_env"]["GLOGGUR_CACHE_DIR"] == str(probe_cache_dir)


def test_gemini_probe_uses_isolated_temp_cache_for_fresh_indexing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.delenv("GLOGGUR_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    probe_cache_dir = tmp_path / "gemini-probe-cache"
    captured: dict[str, object] = {}

    def _fake_tempdir(*args, **kwargs):  # type: ignore[no-untyped-def]
        prefix = kwargs.get("prefix")
        if prefix is None and args:
            prefix = args[0]
        captured["prefix"] = prefix
        return _FakeTempDirectory(str(probe_cache_dir))

    def _fake_run_provider_test(
        *,
        name: str,
        provider: str,
        query: str,
        cache_dir: str,
        runner,
        model_name: str | None = None,
    ) -> SimpleNamespace:
        captured["name"] = name
        captured["provider"] = provider
        captured["query"] = query
        captured["cache_dir"] = cache_dir
        captured["runner_env"] = dict(runner.env or {})
        captured["model_name"] = model_name
        return SimpleNamespace(
            name=name,
            status="passed",
            message="ok",
            details={"indexed_files": 1},
        )

    monkeypatch.setattr("scripts.run_provider_probe.tempfile.TemporaryDirectory", _fake_tempdir)
    monkeypatch.setattr("scripts.run_provider_probe._run_provider_test", _fake_run_provider_test)

    result = _probe_gemini_embeddings(tmp_path, GloggurConfig())

    assert result.status == "passed"
    assert captured["prefix"] == "gloggur-phase2-gemini-"
    assert captured["provider"] == "gemini"
    assert captured["query"] == "hybrid search"
    assert captured["cache_dir"] == str(probe_cache_dir)
    assert isinstance(captured["runner_env"], dict)
    assert captured["runner_env"]["GLOGGUR_CACHE_DIR"] == str(probe_cache_dir)
