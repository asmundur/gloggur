from __future__ import annotations

import errno
import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

import gloggur.indexer.cache as cache_module
import gloggur.storage.metadata_store as metadata_store_module
import gloggur.storage.vector_store as vector_store_module
from gloggur.cli import main as cli_main
from gloggur.cli.main import _metadata_reindex_reason, _profile_reindex_reason
from gloggur.indexer.cache import CacheConfig, CacheManager, CacheRecoveryError
from gloggur.io_failures import StorageIOError
from gloggur.models import IndexMetadata


def test_profile_reindex_reason_no_metadata_and_no_profile() -> None:
    """No index metadata/profile should not force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=False,
        cached_profile=None,
        expected_profile="local:model-a",
    )
    assert reason is None


def test_profile_reindex_reason_unknown_profile_with_metadata() -> None:
    """Index metadata without cached profile should force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile=None,
        expected_profile="local:model-a",
    )
    assert reason == "cached embedding profile is unknown"


def test_profile_reindex_reason_profile_changed() -> None:
    """Mismatched cached/expected profile should force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:model-a",
        expected_profile="local:model-b",
    )
    assert reason == "embedding profile changed (cached=local:model-a, current=local:model-b)"


def test_profile_reindex_reason_profile_matches() -> None:
    """Matching profile should not force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:model-a",
        expected_profile="local:model-a",
    )
    assert reason is None


def test_metadata_reindex_reason_missing_metadata() -> None:
    """Missing index metadata should report an explicit rebuild reason."""
    reason = _metadata_reindex_reason(metadata_present=False)
    assert reason is not None
    assert "index metadata missing" in reason


def test_metadata_reindex_reason_present_metadata() -> None:
    """Existing metadata should not add metadata-specific reindex reason."""
    reason = _metadata_reindex_reason(metadata_present=True)
    assert reason is None


def test_status_supports_tilde_expanded_config_path(tmp_path: Path) -> None:
    """status should expand `~` in --config paths before loading files."""
    runner = CliRunner()
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    config_path = fake_home / ".gloggur.yaml"
    config_path.write_text(
        f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )

    result = runner.invoke(
        cli_main.cli,
        ["status", "--json", "--config", "~/.gloggur.yaml"],
        env={"HOME": str(fake_home)},
    )

    assert result.exit_code == 0, result.output
    payload = _parse_json_output(result.output)
    assert payload["cache_dir"] == str(cache_dir)


def test_create_runtime_applies_embedding_override_without_reloading_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime override should not trigger an unwrapped second config-file load."""

    base_config = cli_main.GloggurConfig(
        embedding_provider="local",
        cache_dir=str(tmp_path / "cache"),
    )

    class FakeCache:
        last_reset_reason = None

        def get_index_metadata(self) -> None:
            return None

        def get_index_profile(self) -> None:
            return None

    monkeypatch.setattr(cli_main, "_load_config", lambda _path: base_config)
    monkeypatch.setattr(cli_main, "_create_cache_manager", lambda _cache_dir: FakeCache())
    monkeypatch.setattr(cli_main, "VectorStore", lambda _cfg: object())

    def _unexpected_reload(*_args: object, **_kwargs: object) -> cli_main.GloggurConfig:
        raise AssertionError("GloggurConfig.load should not be called from _create_runtime")

    monkeypatch.setattr(cli_main.GloggurConfig, "load", _unexpected_reload)

    config, _cache, _vector_store = cli_main._create_runtime(
        config_path=None,
        embedding_provider="openai",
    )

    assert config.embedding_provider == "openai"


def test_create_cache_manager_wraps_cache_recovery_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecoverable cache corruption should map to structured IO failures."""

    class BrokenCacheManager:
        def __init__(self, _config: object) -> None:
            raise CacheRecoveryError("recovery failed")

    monkeypatch.setattr(cli_main, "CacheManager", BrokenCacheManager)
    with pytest.raises(StorageIOError) as exc_info:
        cli_main._create_cache_manager("/tmp/gloggur-cache")

    error = exc_info.value
    assert error.category == "unknown_io_error"
    assert error.operation == "recover corrupted cache database"
    assert error.path.endswith("index.db")
    assert "recovery failed" in error.detail


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo: ["status", "--json"]),
        ("search", lambda repo: ["search", "needle", "--json"]),
        ("inspect", lambda repo: ["inspect", str(repo), "--json"]),
        ("clear-cache", lambda repo: ["clear-cache", "--json"]),
        ("index", lambda repo: ["index", str(repo), "--json"]),
    ],
)
def test_core_commands_surface_cache_recovery_failure_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path], list[str]],
) -> None:
    """Core CLI commands should fail non-zero on unrecoverable cache recovery."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )

    class BrokenCacheManager:
        def __init__(self, _config: object) -> None:
            raise CacheRecoveryError("recovery failed")

    monkeypatch.setattr(cli_main, "CacheManager", BrokenCacheManager)
    result = runner.invoke(cli_main.cli, build_args(repo))

    assert result.exit_code != 0
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "recover corrupted cache database"
    assert str(error["path"]).endswith("index.db")
    assert "recovery failed" in result.output
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize("as_json", [False, True])
def test_status_surfaces_unrecoverable_corruption_recovery_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    as_json: bool,
) -> None:
    """status should fail with remediation guidance if corruption recovery cannot proceed."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "index.db"
    db_path.write_bytes(b"not-a-sqlite-db")

    def _always_fail_replace(_src: str, _dst: str) -> None:
        raise OSError("replace denied")

    def _always_fail_remove(_path: str) -> None:
        raise OSError("remove denied")

    monkeypatch.setattr(cache_module.os, "replace", _always_fail_replace)
    monkeypatch.setattr(cache_module.os, "remove", _always_fail_remove)

    args = ["status", "--json"] if as_json else ["status"]
    result = runner.invoke(cli_main.cli, args, env={"GLOGGUR_CACHE_DIR": str(cache_dir)})
    assert result.exit_code != 0
    assert "Cache corruption detected but recovery failed" in result.output
    assert "Fix permissions and remove corrupted cache files manually." in result.output
    assert "Traceback (most recent call last)" not in result.output


def _parse_json_output(output: str) -> dict[str, object]:
    """Parse JSON payload from click output that may include stderr text."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    return payload


@pytest.mark.parametrize(
    ("exception_factory", "category", "detail_substring"),
    [
        (
            lambda: PermissionError(errno.EACCES, "permission denied"),
            "permission_denied",
            "PermissionError",
        ),
        (
            lambda: OSError(errno.EROFS, "read-only filesystem"),
            "read_only_filesystem",
            "read-only filesystem",
        ),
        (
            lambda: sqlite3.OperationalError("database or disk is full"),
            "disk_full_or_quota",
            "database or disk is full",
        ),
        (
            lambda: sqlite3.OperationalError("unable to open database file"),
            "path_not_writable",
            "unable to open database file",
        ),
        (
            lambda: OSError(errno.EIO, "i/o error"),
            "unknown_io_error",
            "i/o error",
        ),
    ],
)
def test_status_json_reports_structured_io_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_factory: Callable[[], Exception],
    category: str,
    detail_substring: str,
) -> None:
    """status --json should emit stable machine-readable payloads for IO failure categories."""
    runner = CliRunner()

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise exception_factory()

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(
        cli_main.cli,
        ["status", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == category
    assert "open cache database connection" in str(error["operation"])
    assert str(error["path"]).endswith("index.db")
    assert detail_substring in str(error["detail"])
    remediation = error["remediation"]
    assert isinstance(remediation, list) and remediation


def test_status_plain_output_includes_actionable_io_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """status (non-json) should include stable actionable guidance on stderr."""
    runner = CliRunner()

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(
        cli_main.cli,
        ["status"],
        env={"GLOGGUR_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.exit_code == 1
    assert "IO failure [permission_denied]" in result.output
    assert "Probable cause:" in result.output
    assert "Remediation:" in result.output
    assert "PermissionError" in result.output


def test_index_json_reports_vector_store_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """index --json should surface vector id-map write errors with structured payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf8",
    )

    def _raise_disk_full(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(vector_store_module.json, "dump", _raise_disk_full)
    result = runner.invoke(
        cli_main.cli,
        ["index", str(repo), "--json"],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_LOCAL_FALLBACK": "1",
        },
    )
    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["category"] == "disk_full_or_quota"
    assert "write vector id map" in str(error["operation"])
    assert str(error["path"]).endswith("vectors.json")
    assert "no space left on device" in str(error["detail"])


def test_clear_cache_json_reports_vector_artifact_delete_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """clear-cache --json should surface vector artifact delete failures."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_map = cache_dir / "vectors.json"
    vectors_map.write_text("{}", encoding="utf8")
    original_remove = vector_store_module.os.remove

    def _remove_with_permission_denied(path: str | os.PathLike[str]) -> None:
        normalized_path = os.fspath(path)
        if normalized_path.endswith("vectors.json"):
            raise PermissionError(errno.EACCES, "permission denied")
        original_remove(normalized_path)

    monkeypatch.setattr(vector_store_module.os, "remove", _remove_with_permission_denied)
    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "permission_denied"
    assert error["operation"] == "delete vector artifact"
    assert str(error["path"]).endswith("vectors.json")
    assert "permission denied" in str(error["detail"]).lower()


def test_clear_cache_json_ignores_invalid_faiss_index_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear vector files without loading existing index artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_file = cache_dir / "vectors.index"
    index_file.write_text("invalid-faiss-bytes", encoding="utf8")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not index_file.exists()


def test_clear_cache_json_ignores_invalid_vector_id_map_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear malformed vectors.json artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_map = cache_dir / "vectors.json"
    vectors_map.write_text("{this-is-not-json", encoding="utf8")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not vectors_map.exists()


def test_clear_cache_json_ignores_invalid_fallback_vector_matrix_and_clears_artifacts(
    tmp_path: Path,
) -> None:
    """clear-cache --json should clear malformed fallback matrix artifacts."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback_matrix = cache_dir / "vectors.npy"
    fallback_matrix.write_bytes(b"not-a-valid-npy")

    result = runner.invoke(
        cli_main.cli,
        ["clear-cache", "--json"],
        env={"GLOGGUR_CACHE_DIR": str(cache_dir)},
    )

    assert result.exit_code == 0
    payload = _parse_json_output(result.output)
    assert payload["cleared"] is True
    assert not fallback_matrix.exists()


def test_index_json_reports_missing_embedding_provider_configuration(
    tmp_path: Path,
) -> None:
    """index --json should report structured provider error when embedding provider is unset."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / ".gloggur.yaml"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )
    config_path.write_text(
        'embedding_provider: ""\n'
        f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )

    result = runner.invoke(
        cli_main.cli,
        ["index", str(repo), "--json", "--config", str(config_path)],
        env={"GLOGGUR_EMBEDDING_PROVIDER": ""},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "embedding_provider_error"
    assert error["provider"] == "unknown"
    assert error["operation"] == "initialize embedding provider"
    assert "embedding provider is not configured" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


def test_search_json_reports_missing_embedding_provider_configuration(
    tmp_path: Path,
) -> None:
    """search --json should report structured provider error when embedding provider is unset."""
    runner = CliRunner()
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / ".gloggur.yaml"
    config_path.write_text(
        'embedding_provider: ""\n'
        f"cache_dir: {cache_dir}\n",
        encoding="utf8",
    )
    cache = CacheManager(CacheConfig(str(cache_dir)))
    cache.set_index_metadata(IndexMetadata(version="1", total_symbols=0, indexed_files=0))
    cache.set_index_profile(":unknown")

    result = runner.invoke(
        cli_main.cli,
        ["search", "needle", "--json", "--config", str(config_path)],
        env={"GLOGGUR_EMBEDDING_PROVIDER": ""},
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "embedding_provider_error"
    assert error["provider"] == "unknown"
    assert error["operation"] == "initialize embedding provider"
    assert "embedding provider is not configured" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


def test_search_json_wraps_metadata_store_connect_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """search --json should map metadata-store sqlite failures to structured io_failure."""
    runner = CliRunner()

    class FakeCache:
        def get_index_metadata(self) -> IndexMetadata:
            return IndexMetadata(version="1", total_symbols=1, indexed_files=1)

        def get_index_profile(self) -> str:
            return "local:microsoft/codebert-base"

    class FakeVectorStore:
        def search(self, _query_vector: list[float], k: int) -> list[tuple[str, float]]:
            _ = k
            return [("symbol-1", 0.0)]

    class FakeEmbedding:
        provider = "local"

        def embed_text(self, _text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    config = cli_main.GloggurConfig(
        embedding_provider="local",
        local_embedding_model="microsoft/codebert-base",
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **_kwargs: (config, FakeCache(), FakeVectorStore()),
    )
    monkeypatch.setattr(
        cli_main,
        "_create_embedding_provider_for_command",
        lambda *_args, **_kwargs: FakeEmbedding(),
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(metadata_store_module.sqlite3, "connect", _raise_connect)
    result = runner.invoke(cli_main.cli, ["search", "needle", "--json"])

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "path_not_writable"
    assert error["operation"] == "open metadata database connection"
    assert str(error["path"]).endswith("index.db")
    assert "unable to open database file" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo, cfg: ["status", "--json", "--config", str(cfg)]),
        ("search", lambda repo, cfg: ["search", "needle", "--json", "--config", str(cfg)]),
        ("inspect", lambda repo, cfg: ["inspect", str(repo), "--json", "--config", str(cfg)]),
        ("clear-cache", lambda repo, cfg: ["clear-cache", "--json", "--config", str(cfg)]),
        ("index", lambda repo, cfg: ["index", str(repo), "--json", "--config", str(cfg)]),
    ],
)
def test_core_commands_wrap_malformed_config_as_structured_io_failure(
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path, Path], list[str]],
) -> None:
    """Core commands should map malformed config files to stable io_failure payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )
    bad_config = tmp_path / "bad.gloggur.json"
    bad_config.write_text("{not-valid-json", encoding="utf8")

    result = runner.invoke(cli_main.cli, build_args(repo, bad_config))

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "read gloggur config"
    assert str(error["path"]) == str(bad_config)
    assert "malformed" in str(error["probable_cause"]).lower()
    remediation = error.get("remediation")
    assert isinstance(remediation, list)
    assert remediation
    assert "fix config syntax" in str(remediation[0]).lower()
    assert "JSONDecodeError" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    ("command", "build_args"),
    [
        ("status", lambda repo, cfg: ["status", "--json", "--config", str(cfg)]),
        ("search", lambda repo, cfg: ["search", "needle", "--json", "--config", str(cfg)]),
        ("inspect", lambda repo, cfg: ["inspect", str(repo), "--json", "--config", str(cfg)]),
        ("clear-cache", lambda repo, cfg: ["clear-cache", "--json", "--config", str(cfg)]),
        ("index", lambda repo, cfg: ["index", str(repo), "--json", "--config", str(cfg)]),
    ],
)
def test_core_commands_wrap_non_mapping_config_payload_as_structured_io_failure(
    tmp_path: Path,
    command: str,
    build_args: Callable[[Path, Path], list[str]],
) -> None:
    """Core commands should map non-mapping config payloads to stable io_failure payloads."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf8",
    )
    bad_config = tmp_path / "bad.gloggur.yaml"
    bad_config.write_text("- item1\n- item2\n", encoding="utf8")

    result = runner.invoke(cli_main.cli, build_args(repo, bad_config))

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert error["operation"] == "read gloggur config"
    assert str(error["path"]) == str(bad_config)
    assert "malformed" in str(error["probable_cause"]).lower()
    remediation = error.get("remediation")
    assert isinstance(remediation, list)
    assert remediation
    assert "top-level mapping" in str(remediation[0]).lower()
    assert "ValueError" in str(error["detail"])
    assert "Traceback (most recent call last)" not in result.output


@pytest.mark.parametrize(
    "command",
    ["status", "search", "inspect", "clear-cache", "index"],
)
def test_core_commands_wrap_sqlite_database_error_as_structured_io_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    """Core commands should surface sqlite DatabaseError as structured IO failures."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf8",
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)

    args_map = {
        "status": ["status", "--json"],
        "search": ["search", "add", "--json"],
        "inspect": ["inspect", str(repo), "--json"],
        "clear-cache": ["clear-cache", "--json"],
        "index": ["index", str(repo), "--json"],
    }
    result = runner.invoke(
        cli_main.cli,
        args_map[command],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_LOCAL_FALLBACK": "1",
        },
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == "unknown_io_error"
    assert "cache database" in str(error["operation"])
    assert str(error["path"]).endswith("index.db")
    assert "database disk image is malformed" in str(error["detail"])


@pytest.mark.parametrize(
    ("exception_factory", "category", "detail_substring"),
    [
        (
            lambda: PermissionError(errno.EACCES, "permission denied"),
            "permission_denied",
            "PermissionError",
        ),
        (
            lambda: OSError(errno.EROFS, "read-only filesystem"),
            "read_only_filesystem",
            "read-only filesystem",
        ),
        (
            lambda: sqlite3.OperationalError("database or disk is full"),
            "disk_full_or_quota",
            "database or disk is full",
        ),
        (
            lambda: sqlite3.OperationalError("unable to open database file"),
            "path_not_writable",
            "unable to open database file",
        ),
        (
            lambda: OSError(errno.EIO, "i/o error"),
            "unknown_io_error",
            "i/o error",
        ),
    ],
)
@pytest.mark.parametrize(
    "command",
    ["status", "search", "inspect", "clear-cache", "index"],
)
def test_core_commands_wrap_io_failure_categories_consistently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_factory: Callable[[], Exception],
    category: str,
    detail_substring: str,
    command: str,
) -> None:
    """Core commands should classify IO failure categories consistently."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "sample.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf8",
    )

    def _raise_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise exception_factory()

    monkeypatch.setattr(cache_module.sqlite3, "connect", _raise_connect)
    args_map = {
        "status": ["status", "--json"],
        "search": ["search", "add", "--json"],
        "inspect": ["inspect", str(repo), "--json"],
        "clear-cache": ["clear-cache", "--json"],
        "index": ["index", str(repo), "--json"],
    }
    result = runner.invoke(
        cli_main.cli,
        args_map[command],
        env={
            "GLOGGUR_CACHE_DIR": str(tmp_path / "cache"),
            "GLOGGUR_LOCAL_FALLBACK": "1",
        },
    )

    assert result.exit_code == 1
    payload = _parse_json_output(result.output)
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["type"] == "io_failure"
    assert error["category"] == category
    assert str(error["path"]).endswith("index.db")
    assert detail_substring in str(error["detail"])
    remediation = error.get("remediation")
    assert isinstance(remediation, list) and remediation
