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


def _write_resume_manifest(
    cache: CacheManager,
    *,
    build_id: str,
    index_target_path: str,
    embedding_profile: str = "gemini:gemini-embedding-001|embed_graph_edges=0",
    counts: dict[str, int] | None = None,
    progress: dict[str, object] | None = None,
) -> str:
    stage_dir = cache.prepare_staged_build(build_id)
    stage_cache = CacheManager(CacheConfig(stage_dir))
    payload = {
        "build_id": build_id,
        "source": "manifest",
        "workspace_path_hash": cli_main._current_workspace_path_hash(index_target_path),
        "index_target_path": index_target_path,
        "embedding_profile": embedding_profile,
        "schema_version": stage_cache.get_schema_version(),
        "tool_version": cli_main.GLOGGUR_VERSION,
        "stage_cache_dir": stage_dir,
        "state": "interrupted",
        "started_at": "2026-03-13T00:00:00+00:00",
        "updated_at": "2026-03-13T00:00:01+00:00",
        "stage": "embed_chunks",
        "counts": counts or {"files": 1, "symbols": 2, "chunks": 2, "embedded_chunks": 2},
    }
    if progress is not None:
        payload["progress"] = progress
    cache.write_staged_build_resume_manifest(build_id, payload)
    return stage_dir


def test_resume_candidate_payload_marks_compatible_manifest() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-1"
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
    _write_resume_manifest(cache, build_id=build_id, index_target_path=str(Path.cwd()))
    cache.get_staged_build_index_stats = lambda _build_id: {
        "files": 1,
        "symbols": 2,
        "chunks": 2,
        "embedded_chunks": 2,
        "embedded_edges": 0,
        "extract_completed_files": 2,
        "embedded_completed_files": 1,
        "pending_embed_files": 1,
    }

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
    assert candidate["counts"]["extract_completed_files"] == 2
    assert candidate["counts"]["embedded_completed_files"] == 1
    assert candidate["counts"]["pending_embed_files"] == 1
    assert candidate["compatibility"] == {"compatible": True, "reason_codes": []}


def test_resume_candidate_payload_tolerates_legacy_cwd_bound_manifest_when_target_matches() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-legacy-cwd"
    index_target_path = str(Path.cwd() / "subject-repo")
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
    stage_dir = _write_resume_manifest(
        cache,
        build_id=build_id,
        index_target_path=index_target_path,
    )
    cache.write_staged_build_resume_manifest(
        build_id,
        {
            "build_id": build_id,
            "source": "manifest",
            "workspace_path_hash": cli_main._current_workspace_path_hash(str(Path.cwd())),
            "index_target_path": index_target_path,
            "embedding_profile": "gemini:gemini-embedding-001|embed_graph_edges=0",
            "schema_version": CacheManager(CacheConfig(stage_dir)).get_schema_version(),
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
        index_target_path=index_target_path,
    )

    assert candidate is not None
    assert candidate["compatibility"] == {"compatible": True, "reason_codes": []}


def test_status_payload_reports_interrupted_resume_candidate() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-1"
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
    _write_resume_manifest(cache, build_id=build_id, index_target_path=str(Path.cwd()))

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


def test_resume_candidate_payload_preserves_manifest_progress() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-progress"
    progress = {
        "current_file": "sample.py",
        "subphase": "prepare_file",
        "files_done": 1,
        "files_total": 3,
        "started_at": "2026-03-14T10:00:00+00:00",
        "updated_at": "2026-03-14T10:00:05+00:00",
    }
    cache.write_build_state(
        {
            "state": "interrupted",
            "build_id": build_id,
            "pid": 123,
            "started_at": "2026-03-14T10:00:00+00:00",
            "updated_at": "2026-03-14T10:00:05+00:00",
            "stage": "extract_symbols",
            "cleanup_pending": True,
            "progress": progress,
        }
    )
    _write_resume_manifest(
        cache,
        build_id=build_id,
        index_target_path=str(Path.cwd()),
        progress=progress,
    )

    candidate = cli_main._build_resume_candidate_payload(
        cache,
        build_state=cache.get_build_state(),
        expected_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        index_target_path=str(Path.cwd()),
    )
    assert isinstance(candidate, dict)
    assert candidate["progress"] == progress

    payload = cli_main._build_status_payload(
        GloggurConfig(cache_dir=cache_dir, embedding_provider="gemini"),
        cache,
    )
    resume_candidate = payload["resume_candidate"]
    assert isinstance(resume_candidate, dict)
    assert resume_candidate["progress"] == progress


def test_discover_resume_candidates_finds_orphaned_manifest_without_persisted_build_state() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-orphaned"
    _write_resume_manifest(cache, build_id=build_id, index_target_path=str(Path.cwd()))

    discovery = cli_main._discover_resume_candidates(
        cache,
        build_state=None,
        expected_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        index_target_path=str(Path.cwd()),
    )

    assert discovery["interrupted_resume_available"] is True
    assert discovery["resume_blocked"] is False
    resume_candidate = discovery["resume_candidate"]
    assert isinstance(resume_candidate, dict)
    assert resume_candidate["build_id"] == build_id


def test_discover_resume_candidates_ignores_missing_build_state_hint_and_selects_orphaned_build() -> (
    None
):
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-actual"
    _write_resume_manifest(cache, build_id=build_id, index_target_path=str(Path.cwd()))

    discovery = cli_main._discover_resume_candidates(
        cache,
        build_state={
            "state": "interrupted",
            "build_id": "missing-build",
            "pid": 123,
            "started_at": "2026-03-13T00:00:00+00:00",
            "updated_at": "2026-03-13T00:00:01+00:00",
            "stage": "embed_chunks",
            "cleanup_pending": True,
        },
        expected_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        index_target_path=str(Path.cwd()),
    )

    resume_candidate = discovery["resume_candidate"]
    assert isinstance(resume_candidate, dict)
    assert resume_candidate["build_id"] == build_id
    assert resume_candidate["preferred"] is False


def test_discover_resume_candidates_ranks_using_refreshed_stage_counts() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    _write_resume_manifest(
        cache,
        build_id="build-small",
        index_target_path=str(Path.cwd()),
        counts={"files": 0, "symbols": 0, "chunks": 0, "embedded_chunks": 0},
    )
    _write_resume_manifest(
        cache,
        build_id="build-large",
        index_target_path=str(Path.cwd()),
        counts={"files": 0, "symbols": 0, "chunks": 0, "embedded_chunks": 0},
    )
    cache.get_staged_build_index_stats = lambda build_id: {
        "files": 1,
        "symbols": 2,
        "chunks": 2,
        "embedded_chunks": 1 if build_id == "build-large" else 5,
        "embedded_edges": 0,
        "extract_completed_files": 4 if build_id == "build-large" else 1,
        "embedded_completed_files": 2 if build_id == "build-large" else 1,
        "pending_embed_files": 2 if build_id == "build-large" else 0,
    }

    discovery = cli_main._discover_resume_candidates(
        cache,
        build_state=None,
        expected_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        index_target_path=str(Path.cwd()),
    )

    resume_candidate = discovery["resume_candidate"]
    assert isinstance(resume_candidate, dict)
    assert resume_candidate["build_id"] == "build-large"
    assert resume_candidate["counts"]["embedded_completed_files"] == 2
    assert resume_candidate["counts"]["extract_completed_files"] == 4


def test_discover_resume_candidates_blocks_manifestless_or_incompatible_staged_work() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    cache.prepare_staged_build("legacy-build")
    _write_resume_manifest(
        cache,
        build_id="wrong-profile",
        index_target_path=str(Path.cwd()),
        embedding_profile="local:sentence-transformers/all-MiniLM-L6-v2|embed_graph_edges=0",
    )

    discovery = cli_main._discover_resume_candidates(
        cache,
        build_state=None,
        expected_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        index_target_path=str(Path.cwd()),
    )

    assert discovery["interrupted_resume_available"] is False
    assert discovery["resume_blocked"] is True
    assert "no_compatible_staged_build" in discovery["resume_block_reason_codes"]
    assert "resume_manifest_missing" in discovery["resume_block_reason_codes"]
    assert "embedding_profile_changed" in discovery["resume_block_reason_codes"]
    blocked_candidates = discovery["resume_block_candidates"]
    assert isinstance(blocked_candidates, list)
    assert {candidate["build_id"] for candidate in blocked_candidates} == {
        "legacy-build",
        "wrong-profile",
    }


def test_status_payload_reports_orphaned_resume_candidate_without_persisted_build_state() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    cache = CacheManager(CacheConfig(cache_dir))
    build_id = "build-orphaned"
    _write_resume_manifest(cache, build_id=build_id, index_target_path=str(Path.cwd()))

    payload = cli_main._build_status_payload(
        GloggurConfig(cache_dir=cache_dir, embedding_provider="gemini"),
        cache,
    )

    assert payload["build_state"] is None
    assert payload["interrupted_resume_available"] is True
    resume_candidate = payload["resume_candidate"]
    assert isinstance(resume_candidate, dict)
    assert resume_candidate["build_id"] == build_id


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
