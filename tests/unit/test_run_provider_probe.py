from __future__ import annotations

import json
from pathlib import Path

from gloggur.config import GloggurConfig
from scripts.run_provider_probe import _vector_store_stats
from scripts.run_provider_probe import test_gemini_embeddings as _probe_gemini_embeddings


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
