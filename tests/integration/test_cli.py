from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from gloggur.cli.main import cli
from gloggur.indexer.cache import CacheConfig, CacheManager
from scripts.verification.fixtures import TestFixtures


def _write_fallback_marker(cache_dir: str) -> None:
    """Create the local embedding fallback marker file."""
    marker = Path(cache_dir) / ".local_embedding_fallback"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def _parse_json_output(output: str) -> dict[str, object]:
    """Parse JSON output from CLI command output."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    return payload


def _tamper_vector_id_map_drop_one_symbol(cache_dir: str) -> None:
    """Drop one symbol id from vectors.json to simulate vector/cache drift."""
    id_map_path = Path(cache_dir) / "vectors.json"
    payload = json.loads(id_map_path.read_text(encoding="utf8"))
    symbol_map = payload.get("symbol_to_vector_id")
    if not isinstance(symbol_map, dict) or not symbol_map:
        raise AssertionError("expected non-empty symbol_to_vector_id map")
    drop_id = sorted(symbol_map)[0]
    symbol_map.pop(drop_id, None)
    payload["symbol_to_vector_id"] = symbol_map
    fallback_order = payload.get("fallback_order")
    if isinstance(fallback_order, list):
        payload["fallback_order"] = [
            str(symbol_id) for symbol_id in fallback_order if str(symbol_id) in symbol_map
        ]
    id_map_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")


def test_cli_index_search_status_and_clear_cache() -> None:
    """End-to-end CLI test for index/search/status/clear-cache."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0
        index_payload = _parse_json_output(index_result.output)
        assert index_payload["indexed_files"] == 1
        assert index_payload["indexed_symbols"] > 0

        status_result = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_result.exit_code == 0
        status_payload = _parse_json_output(status_result.output)
        assert status_payload["total_symbols"] > 0

        search_result = runner.invoke(cli, ["search", "add", "--json", "--top-k", "3"], env=env)
        assert search_result.exit_code == 0
        search_payload = _parse_json_output(search_result.output)
        assert search_payload["metadata"]["total_results"] > 0

        clear_result = runner.invoke(cli, ["clear-cache", "--json"], env=env)
        assert clear_result.exit_code == 0
        clear_payload = _parse_json_output(clear_result.output)
        assert clear_payload["cleared"] is True


def test_cli_detects_model_change_and_rebuilds_on_index() -> None:
    """Status/search should require reindex on model change; index should self-rebuild."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)

        env_model_a = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "model-a",
        }
        env_model_b = {
            "GLOGGUR_CACHE_DIR": cache_dir,
            "GLOGGUR_LOCAL_MODEL": "model-b",
        }

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env_model_a)
        assert first_index.exit_code == 0

        status_before = runner.invoke(cli, ["status", "--json"], env=env_model_a)
        assert status_before.exit_code == 0
        status_before_payload = _parse_json_output(status_before.output)
        assert status_before_payload["needs_reindex"] is False
        assert status_before_payload["expected_index_profile"] == "local:model-a"
        assert status_before_payload["cached_index_profile"] == "local:model-a"
        assert status_before_payload["resume_decision"] == "resume_ok"
        assert status_before_payload["resume_reason_codes"] == []
        assert status_before_payload["resume_fingerprint_match"] is True
        assert status_before_payload["last_success_resume_fingerprint"] == status_before_payload[
            "expected_resume_fingerprint"
        ]
        assert status_before_payload["last_success_resume_fingerprint_match"] is True
        assert isinstance(status_before_payload["last_success_resume_at"], str)
        assert isinstance(status_before_payload["tool_version"], str)
        assert status_before_payload["last_success_tool_version"] == status_before_payload[
            "tool_version"
        ]
        assert status_before_payload["last_success_tool_version_match"] is True
        first_success_fingerprint = status_before_payload["last_success_resume_fingerprint"]
        assert isinstance(first_success_fingerprint, str)

        status_changed = runner.invoke(cli, ["status", "--json"], env=env_model_b)
        assert status_changed.exit_code == 0
        status_changed_payload = _parse_json_output(status_changed.output)
        assert status_changed_payload["needs_reindex"] is True
        assert status_changed_payload["expected_index_profile"] == "local:model-b"
        assert status_changed_payload["cached_index_profile"] == "local:model-a"
        assert "embedding profile changed" in str(status_changed_payload["reindex_reason"])
        assert status_changed_payload["resume_decision"] == "reindex_required"
        assert status_changed_payload["resume_reason_codes"] == ["embedding_profile_changed"]
        assert status_changed_payload["resume_fingerprint_match"] is False
        assert status_changed_payload["last_success_resume_fingerprint"] == first_success_fingerprint
        assert status_changed_payload["last_success_resume_fingerprint_match"] is False
        assert status_changed_payload["last_success_tool_version_match"] is True

        search_changed = runner.invoke(cli, ["search", "add", "--json"], env=env_model_b)
        assert search_changed.exit_code == 0
        search_changed_payload = _parse_json_output(search_changed.output)
        assert search_changed_payload["results"] == []
        assert search_changed_payload["metadata"]["needs_reindex"] is True
        assert "embedding profile changed" in str(
            search_changed_payload["metadata"]["reindex_reason"]
        )
        assert search_changed_payload["metadata"]["resume_decision"] == "reindex_required"
        assert search_changed_payload["metadata"]["resume_reason_codes"] == [
            "embedding_profile_changed"
        ]
        assert search_changed_payload["metadata"]["resume_fingerprint_match"] is False
        assert (
            search_changed_payload["metadata"]["last_success_resume_fingerprint"]
            == first_success_fingerprint
        )
        assert search_changed_payload["metadata"]["last_success_resume_fingerprint_match"] is False
        assert search_changed_payload["metadata"]["last_success_tool_version_match"] is True

        rebuild_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env_model_b)
        assert rebuild_index.exit_code == 0
        rebuild_payload = _parse_json_output(rebuild_index.output)
        assert rebuild_payload["indexed_files"] == 1
        assert rebuild_payload["indexed_symbols"] > 0

        status_after = runner.invoke(cli, ["status", "--json"], env=env_model_b)
        assert status_after.exit_code == 0
        status_after_payload = _parse_json_output(status_after.output)
        assert status_after_payload["needs_reindex"] is False
        assert status_after_payload["expected_index_profile"] == "local:model-b"
        assert status_after_payload["cached_index_profile"] == "local:model-b"
        assert status_after_payload["resume_decision"] == "resume_ok"
        assert status_after_payload["resume_reason_codes"] == []
        assert status_after_payload["resume_fingerprint_match"] is True
        assert status_after_payload["last_success_resume_fingerprint"] == status_after_payload[
            "expected_resume_fingerprint"
        ]
        assert status_after_payload["last_success_resume_fingerprint_match"] is True
        assert status_after_payload["last_success_tool_version"] == status_after_payload["tool_version"]
        assert status_after_payload["last_success_tool_version_match"] is True
        assert status_after_payload["last_success_resume_fingerprint"] != first_success_fingerprint


def test_cli_index_reports_incremental_observability_and_prunes_deleted_files() -> None:
    """Index payload should expose incremental counters and prune deleted-file symbols."""
    runner = CliRunner()
    source = (
        "def keep_me(value: int) -> int:\n"
        "    return value + 1\n"
    )
    delete_source = (
        "def remove_me(value: int) -> int:\n"
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": source, "remove.py": delete_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0
        first_payload = _parse_json_output(first_index.output)
        for field in (
            "files_scanned",
            "files_changed",
            "files_removed",
            "symbols_added",
            "symbols_updated",
            "symbols_removed",
        ):
            assert field in first_payload
        assert int(first_payload["files_changed"]) == 2
        assert int(first_payload["files_removed"]) == 0
        assert int(first_payload["symbols_added"]) > 0
        initial_symbols = int(first_payload["indexed_symbols"])
        assert initial_symbols > 0

        status_before = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_before.exit_code == 0
        status_before_payload = _parse_json_output(status_before.output)
        total_before = int(status_before_payload["total_symbols"])
        assert total_before == initial_symbols

        (repo / "remove.py").unlink()
        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 0
        second_payload = _parse_json_output(second_index.output)
        assert int(second_payload["indexed_files"]) == 0
        assert int(second_payload["skipped_files"]) == 1
        assert int(second_payload["files_removed"]) == 1
        assert int(second_payload["symbols_removed"]) > 0
        assert int(second_payload["files_changed"]) == 0

        status_after = runner.invoke(cli, ["status", "--json"], env=env)
        assert status_after.exit_code == 0
        status_after_payload = _parse_json_output(status_after.output)
        total_after = int(status_after_payload["total_symbols"])
        assert total_after < total_before


def test_cli_index_rename_does_not_leave_ghost_symbols() -> None:
    """Rename flows should not report success while retaining stale old-path symbols."""
    runner = CliRunner()
    old_source = (
        'def legacy_symbol(value: int) -> int:\n'
        '    """legacy rename ghost token old-only"""\n'
        "    return value + 1\n"
    )
    new_source = (
        'def modern_symbol(value: int) -> int:\n'
        '    """fresh rename token new-only"""\n'
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"module_old.py": old_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0

        old_path = repo / "module_old.py"
        new_path = repo / "module_new.py"
        old_path.rename(new_path)
        new_path.write_text(new_source, encoding="utf8")

        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 0, second_index.output
        second_payload = _parse_json_output(second_index.output)
        assert int(second_payload["indexed_files"]) == 1
        assert int(second_payload["files_removed"]) == 1
        assert int(second_payload["symbols_removed"]) > 0
        assert int(second_payload["symbols_added"]) > 0

        cache = CacheManager(CacheConfig(cache_dir))
        assert cache.get_file_metadata(str(old_path)) is None
        assert cache.list_symbols_for_file(str(old_path)) == []
        assert cache.get_file_metadata(str(new_path)) is not None


def test_cli_index_docstring_only_change_is_not_skipped() -> None:
    """Docstring-only edits should trigger reindex and refresh cached symbol docstrings."""
    runner = CliRunner()
    original_source = (
        "def summarize(value: int) -> int:\n"
        '    """doc-old-token-for-regression"""\n'
        "    return value + 1\n"
    )
    updated_source = (
        "def summarize(value: int) -> int:\n"
        '    """doc-new-token-for-regression"""\n'
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": original_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output

        file_path = repo / "sample.py"
        file_path.write_text(updated_source, encoding="utf8")
        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 0, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["indexed_files"]) == 1
        assert int(payload["files_changed"]) == 1
        assert int(payload["skipped_files"]) == 0
        assert int(payload["symbols_updated"]) >= 1

        cache = CacheManager(CacheConfig(cache_dir))
        symbols = cache.list_symbols_for_file(str(file_path))
        assert len(symbols) == 1
        docstring = symbols[0].docstring
        assert isinstance(docstring, str)
        assert "doc-new-token-for-regression" in docstring
        assert "doc-old-token-for-regression" not in docstring


def test_cli_index_reports_vector_metadata_mismatch_on_tampered_vector_map() -> None:
    """Index should fail closed with deterministic JSON when vectors.json drifts from cache symbols."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output
        _tamper_vector_id_map_drop_one_symbol(cache_dir)

        second_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert second_index.exit_code == 1, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["failed"]) == 1
        reasons = payload["failed_reasons"]
        assert isinstance(reasons, dict)
        assert reasons == {"vector_metadata_mismatch": 1}
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["vector_metadata_mismatch"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "vector_metadata_mismatch" in guidance
        assert isinstance(guidance["vector_metadata_mismatch"], list)
        assert guidance["vector_metadata_mismatch"]


def test_cli_single_file_index_fails_closed_on_tampered_vector_map() -> None:
    """Single-file index should not report success when vectors.json drifts from cache symbols."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output
        _tamper_vector_id_map_drop_one_symbol(cache_dir)

        second_index = runner.invoke(cli, ["index", str(repo / "sample.py"), "--json"], env=env)
        assert second_index.exit_code == 1, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["files_considered"]) == 1
        assert int(payload["failed"]) == 1
        reasons = payload["failed_reasons"]
        assert isinstance(reasons, dict)
        assert reasons == {"vector_metadata_mismatch": 1}
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["vector_metadata_mismatch"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "vector_metadata_mismatch" in guidance
        assert isinstance(guidance["vector_metadata_mismatch"], list)
        assert guidance["vector_metadata_mismatch"]


def test_cli_single_file_index_rename_prunes_missing_old_path_entries() -> None:
    """Single-file indexing should prune stale old-path rows after rename flows."""
    runner = CliRunner()
    old_source = (
        'def legacy_symbol(value: int) -> int:\n'
        '    """legacy single-file rename token old-only"""\n'
        "    return value + 1\n"
    )
    new_source = (
        'def modern_symbol(value: int) -> int:\n'
        '    """single-file rename token new-only"""\n'
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"module_old.py": old_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output

        old_path = repo / "module_old.py"
        new_path = repo / "module_new.py"
        old_path.rename(new_path)
        new_path.write_text(new_source, encoding="utf8")

        second_index = runner.invoke(cli, ["index", str(new_path), "--json"], env=env)
        assert second_index.exit_code == 0, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["files_considered"]) == 1
        assert int(payload["indexed_files"]) == 1
        assert int(payload["files_changed"]) == 1
        assert int(payload["files_removed"]) == 1
        assert int(payload["symbols_removed"]) > 0

        cache = CacheManager(CacheConfig(cache_dir))
        assert cache.get_file_metadata(str(old_path)) is None
        assert cache.list_symbols_for_file(str(old_path)) == []
        assert cache.get_file_metadata(str(new_path)) is not None


def test_cli_single_file_index_surfaces_stale_cleanup_error_with_failure_contract(
    monkeypatch,
) -> None:
    """Single-file index should fail closed with deterministic stale cleanup codes/remediation."""
    runner = CliRunner()
    keep_source = (
        "def keep_me(value: int) -> int:\n"
        "    return value + 1\n"
    )
    drop_source = (
        "def drop_me(value: int) -> int:\n"
        "    return value + 2\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"keep.py": keep_source, "drop.py": drop_source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        first_index = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert first_index.exit_code == 0, first_index.output

        stale_path = str(repo / "drop.py")
        (repo / "drop.py").unlink()
        original_delete = CacheManager.delete_file_metadata

        def _raise_on_stale_delete(self, path: str) -> None:
            if path == stale_path:
                raise OSError("simulated stale cleanup failure")
            original_delete(self, path)

        monkeypatch.setattr(CacheManager, "delete_file_metadata", _raise_on_stale_delete)
        second_index = runner.invoke(cli, ["index", str(repo / "keep.py"), "--json"], env=env)

        assert second_index.exit_code == 1, second_index.output
        payload = _parse_json_output(second_index.output)
        assert int(payload["files_considered"]) == 1
        assert int(payload["failed"]) == 1
        reasons = payload["failed_reasons"]
        assert isinstance(reasons, dict)
        assert reasons == {"stale_cleanup_error": 1}
        failure_codes = payload["failure_codes"]
        assert isinstance(failure_codes, list)
        assert failure_codes == ["stale_cleanup_error"]
        guidance = payload["failure_guidance"]
        assert isinstance(guidance, dict)
        assert "stale_cleanup_error" in guidance
        assert isinstance(guidance["stale_cleanup_error"], list)
        assert guidance["stale_cleanup_error"]


def test_cli_status_and_index_self_heal_corrupted_cache_idempotently() -> None:
    """Status/index should self-heal a corrupted DB once and remain stable afterwards."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        db_path = Path(cache_dir) / "index.db"
        db_path.write_bytes(b"broken sqlite bytes")
        Path(f"{db_path}-wal").write_bytes(b"broken wal")
        Path(f"{db_path}-shm").write_bytes(b"broken shm")

        first_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert first_status.exit_code == 0
        assert "Cache corruption detected at" in first_status.output
        first_payload = _parse_json_output(first_status.output)
        assert first_payload["needs_reindex"] is True
        assert int(first_payload["total_symbols"]) == 0
        quarantined_after_first = sorted(
            path.name for path in Path(cache_dir).iterdir() if ".corrupt." in path.name
        )
        assert quarantined_after_first
        assert not Path(f"{db_path}-wal").exists()
        assert not Path(f"{db_path}-shm").exists()

        second_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert second_status.exit_code == 0
        assert "Cache corruption detected at" not in second_status.output
        second_payload = _parse_json_output(second_status.output)
        assert second_payload["needs_reindex"] is True
        quarantined_after_second = sorted(
            path.name for path in Path(cache_dir).iterdir() if ".corrupt." in path.name
        )
        assert quarantined_after_second == quarantined_after_first

        index_result = runner.invoke(cli, ["index", str(repo), "--json"], env=env)
        assert index_result.exit_code == 0
        index_payload = _parse_json_output(index_result.output)
        assert index_payload["indexed_files"] == 1
        assert index_payload["indexed_symbols"] > 0

        final_status = runner.invoke(cli, ["status", "--json"], env=env)
        assert final_status.exit_code == 0
        final_payload = _parse_json_output(final_status.output)
        assert final_payload["needs_reindex"] is False
        assert int(final_payload["total_symbols"]) > 0


def test_cli_search_self_heals_corrupted_cache() -> None:
    """Search should recover from cache corruption and return reindex metadata, not crash."""
    runner = CliRunner()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {"GLOGGUR_CACHE_DIR": cache_dir}
    db_path = Path(cache_dir) / "index.db"
    db_path.write_bytes(b"broken sqlite bytes")

    first_search = runner.invoke(cli, ["search", "add", "--json"], env=env)
    assert first_search.exit_code == 0
    assert "Cache corruption detected at" in first_search.output
    first_payload = _parse_json_output(first_search.output)
    assert first_payload["results"] == []
    metadata = first_payload["metadata"]
    assert isinstance(metadata, dict)
    assert int(metadata["total_results"]) == 0

    second_search = runner.invoke(cli, ["search", "add", "--json"], env=env)
    assert second_search.exit_code == 0
    assert "Cache corruption detected at" not in second_search.output


def test_cli_clear_cache_self_heals_corrupted_cache() -> None:
    """clear-cache should recover from corruption instead of failing with a DB error."""
    runner = CliRunner()
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    _write_fallback_marker(cache_dir)
    env = {"GLOGGUR_CACHE_DIR": cache_dir}
    db_path = Path(cache_dir) / "index.db"
    db_path.write_bytes(b"broken sqlite bytes")

    result = runner.invoke(cli, ["clear-cache", "--json"], env=env)
    assert result.exit_code == 0
    assert "Cache corruption detected at" in result.output
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True


def test_cli_inspect_self_heals_corrupted_cache() -> None:
    """Inspect command should run after automatic corruption recovery."""
    runner = CliRunner()
    source = TestFixtures.create_sample_python_file()
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"sample.py": source})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}
        db_path = Path(cache_dir) / "index.db"
        db_path.write_bytes(b"broken sqlite bytes")

        result = runner.invoke(cli, ["inspect", str(repo), "--json"], env=env)
        assert result.exit_code == 0
        assert "Cache corruption detected at" in result.output
        payload = _parse_json_output(result.output)
        assert int(payload["reports_total"]) >= 0


def test_cli_inspect_defaults_to_source_focus_with_opt_in_full_audit() -> None:
    """Inspect should default to src focus and include tests/scripts only when explicitly requested."""
    runner = CliRunner()
    source_no_docstring = (
        "def sample(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo(
            {
                "src/main.py": source_no_docstring,
                "tests/test_main.py": source_no_docstring,
                "scripts/tool.py": source_no_docstring,
            }
        )
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        default_result = runner.invoke(cli, ["inspect", str(repo), "--json", "--force"], env=env)
        assert default_result.exit_code == 0, default_result.output
        default_payload = _parse_json_output(default_result.output)
        default_scope = default_payload["inspect_scope"]
        assert isinstance(default_scope, dict)
        assert default_scope["include_tests"] is False
        assert default_scope["include_scripts"] is False
        default_summary = default_payload["warning_summary"]
        assert isinstance(default_summary, dict)
        default_by_class = default_summary["by_path_class"]
        assert isinstance(default_by_class, dict)
        assert int(default_by_class["src"]) > 0
        assert int(default_by_class["tests"]) == 0
        assert int(default_by_class["scripts"]) == 0
        default_warnings = default_payload["warnings"]
        assert isinstance(default_warnings, list)
        assert all("/tests/" not in str(item["symbol_id"]) for item in default_warnings)
        assert all("/scripts/" not in str(item["symbol_id"]) for item in default_warnings)

        full_result = runner.invoke(
            cli,
            [
                "inspect",
                str(repo),
                "--json",
                "--force",
                "--include-tests",
                "--include-scripts",
            ],
            env=env,
        )
        assert full_result.exit_code == 0, full_result.output
        full_payload = _parse_json_output(full_result.output)
        full_scope = full_payload["inspect_scope"]
        assert isinstance(full_scope, dict)
        assert full_scope["include_tests"] is True
        assert full_scope["include_scripts"] is True
        full_summary = full_payload["warning_summary"]
        assert isinstance(full_summary, dict)
        full_by_class = full_summary["by_path_class"]
        assert isinstance(full_by_class, dict)
        assert int(full_by_class["src"]) > 0
        assert int(full_by_class["tests"]) > 0
        assert int(full_by_class["scripts"]) > 0


def test_cli_inspect_explicit_tests_path_remains_included_without_flags() -> None:
    """Explicit tests-path inspect should not be filtered out by default src-focus behavior."""
    runner = CliRunner()
    source_no_docstring = (
        "def sample(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"tests/test_main.py": source_no_docstring})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        result = runner.invoke(
            cli,
            ["inspect", str(repo / "tests"), "--json", "--force"],
            env=env,
        )
        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)
        summary = payload["warning_summary"]
        assert isinstance(summary, dict)
        by_class = summary["by_path_class"]
        assert isinstance(by_class, dict)
        assert int(by_class["tests"]) > 0
        assert int(by_class["src"]) == 0


def test_cli_inspect_warning_summary_payload_schema_is_stable() -> None:
    """Inspect payload should keep stable summary fields and value types for automations."""
    runner = CliRunner()
    source_no_docstring = (
        "def sample(value: int) -> int:\n"
        "    return value + 1\n"
    )
    with TestFixtures() as fixtures:
        repo = fixtures.create_temp_repo({"src/main.py": source_no_docstring})
        cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
        _write_fallback_marker(cache_dir)
        env = {"GLOGGUR_CACHE_DIR": cache_dir}

        result = runner.invoke(cli, ["inspect", str(repo), "--json", "--force"], env=env)
        assert result.exit_code == 0, result.output
        payload = _parse_json_output(result.output)

        for legacy_field in (
            "warnings",
            "total",
            "reports",
            "reports_total",
            "files_considered",
            "inspected_files",
            "skipped_files",
        ):
            assert legacy_field in payload

        assert "inspect_payload_schema_version" in payload
        assert payload["inspect_payload_schema_version"] == "1"

        scope = payload["inspect_scope"]
        assert isinstance(scope, dict)
        assert set(scope.keys()) == {"default_src_focus", "include_tests", "include_scripts"}
        assert isinstance(scope["default_src_focus"], bool)
        assert isinstance(scope["include_tests"], bool)
        assert isinstance(scope["include_scripts"], bool)

        summary = payload["warning_summary"]
        assert isinstance(summary, dict)
        assert set(summary.keys()) == {
            "total_warnings",
            "by_warning_type",
            "by_path_class",
            "reports_by_path_class",
            "top_files",
        }
        assert isinstance(summary["total_warnings"], int)
        assert isinstance(summary["by_warning_type"], dict)
        assert isinstance(summary["by_path_class"], dict)
        assert isinstance(summary["reports_by_path_class"], dict)
        assert isinstance(summary["top_files"], list)

        by_path_class = summary["by_path_class"]
        assert set(by_path_class.keys()) == {"src", "tests", "scripts", "other"}
        assert all(isinstance(value, int) for value in by_path_class.values())

        reports_by_path_class = summary["reports_by_path_class"]
        assert set(reports_by_path_class.keys()) == {"src", "tests", "scripts", "other"}
        assert all(isinstance(value, int) for value in reports_by_path_class.values())

        assert summary["top_files"], "expected at least one top file entry"
        for item in summary["top_files"]:
            assert isinstance(item, dict)
            assert set(item.keys()) == {"file", "warnings", "path_class"}
            assert isinstance(item["file"], str)
            assert isinstance(item["warnings"], int)
            assert item["path_class"] in {"src", "tests", "scripts", "other"}
