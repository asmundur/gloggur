from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from gloggur.cli import main as cli_main
from gloggur.config import GloggurConfig
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.embedding_ledger import EmbeddingLedger, embedding_text_hash


def _parse_json_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {output!r}")
    return json.loads(output[start:])


def test_resume_candidate_payload_marks_compatible_manifest() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-1"
    stage_dir = cache.prepare_staged_build(build_id)
    stage_cache = CacheManager(CacheConfig(stage_dir))
    cache.write_build_state(
        {
            "state": "interrupted",
            "build_id": build_id,
            "pid": 123,
            "started_at": "2026-03-13T00:00:00+00:00",
            "updated_at": "2026-03-13T00:00:01+00:00",
            "stage": "embed_chunks",
            "cleanup_pending": True,
        }
    )
    cache.write_staged_build_resume_manifest(
        build_id,
        {
            "build_id": build_id,
            "source": "manifest",
            "workspace_path_hash": cli_main._current_workspace_path_hash(),
            "index_target_path": str(Path.cwd()),
            "embedding_profile": "gemini:gemini-embedding-001|embed_graph_edges=0",
            "schema_version": stage_cache.get_schema_version(),
            "tool_version": cli_main.GLOGGUR_VERSION,
            "stage_cache_dir": stage_dir,
            "state": "interrupted",
            "started_at": "2026-03-13T00:00:00+00:00",
            "updated_at": "2026-03-13T00:00:01+00:00",
            "stage": "embed_chunks",
            "counts": {"files": 1, "symbols": 2, "chunks": 2, "embedded_chunks": 2},
        },
    )

    candidate = cli_main._build_resume_candidate_payload(
        cache,
        build_state=cache.get_build_state(),
        expected_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        index_target_path=str(Path.cwd()),
    )

    assert candidate is not None
    assert candidate["build_id"] == build_id
    assert candidate["source"] == "manifest"
    assert candidate["counts"]["embedded_chunks"] == 2
    assert candidate["compatibility"] == {"compatible": True, "reason_codes": []}


def test_status_payload_reports_interrupted_resume_candidate() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-1"
    stage_dir = cache.prepare_staged_build(build_id)
    stage_cache = CacheManager(CacheConfig(stage_dir))
    cache.write_build_state(
        {
            "state": "interrupted",
            "build_id": build_id,
            "pid": 123,
            "started_at": "2026-03-13T00:00:00+00:00",
            "updated_at": "2026-03-13T00:00:01+00:00",
            "stage": "embed_chunks",
            "cleanup_pending": True,
        }
    )
    cache.write_staged_build_resume_manifest(
        build_id,
        {
            "build_id": build_id,
            "workspace_path_hash": cli_main._current_workspace_path_hash(),
            "index_target_path": str(Path.cwd()),
            "embedding_profile": "gemini:gemini-embedding-001|embed_graph_edges=0",
            "schema_version": stage_cache.get_schema_version(),
            "tool_version": cli_main.GLOGGUR_VERSION,
            "stage_cache_dir": stage_dir,
            "state": "interrupted",
            "started_at": "2026-03-13T00:00:00+00:00",
            "updated_at": "2026-03-13T00:00:01+00:00",
            "stage": "embed_chunks",
            "counts": {"files": 1, "symbols": 2, "chunks": 2, "embedded_chunks": 2},
        },
    )

    payload = cli_main._build_status_payload(
        GloggurConfig(cache_dir=cache_dir, embedding_provider="gemini"),
        cache,
    )

    assert payload["interrupted_resume_available"] is True
    resume_candidate = payload["resume_candidate"]
    assert isinstance(resume_candidate, dict)
    assert resume_candidate["build_id"] == build_id
    assert resume_candidate["compatibility"] == {"compatible": True, "reason_codes": []}
    remediation = payload["resume_remediation"]["index_interrupted"]
    assert remediation[0].startswith("A compatible interrupted staged build is available")


def test_clear_cache_preserves_or_purges_embedding_ledger() -> None:
    runner = CliRunner()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    ledger = EmbeddingLedger(cache_dir)
    ledger.upsert_vectors(
        embedding_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        record_kind="chunk",
        entries=[(embedding_text_hash("alpha"), [0.1, 0.2])],
    )
    ledger_path = Path(cache_dir) / "embedding-ledger.db"
    env = {"GLOGGUR_CACHE_DIR": cache_dir}

    preserved = runner.invoke(cli_main.cli, ["clear-cache", "--json"], env=env)

    assert preserved.exit_code == 0, preserved.output
    preserved_payload = _parse_json_output(preserved.output)
    assert preserved_payload["purged_embedding_ledger"] is False
    assert ledger_path.exists()

    purged = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json", "--purge-embedding-ledger"],
        env=env,
    )

    assert purged.exit_code == 0, purged.output
    purged_payload = _parse_json_output(purged.output)
    assert purged_payload["purged_embedding_ledger"] is True
    assert not ledger_path.exists()
