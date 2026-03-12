from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from gloggur.adapters.registry import AdapterResolutionError
from gloggur.cli import main as cli_main
from gloggur.config import GloggurConfig


def _payload(output: str) -> dict[str, object]:
    payload = json.loads(output)
    assert isinstance(payload, dict)
    return payload


def test_main_returns_cli_exit_code_or_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_main.cli,
        "main",
        lambda args, prog_name, standalone_mode: 7,
    )
    assert cli_main.main(["status"]) == 7

    monkeypatch.setattr(
        cli_main.cli,
        "main",
        lambda args, prog_name, standalone_mode: None,
    )
    assert cli_main.main(["status"]) == 0


def test_main_json_click_exception_emits_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_main.cli,
        "main",
        lambda args, prog_name, standalone_mode: (_ for _ in ()).throw(
            click.UsageError("bad flag")
        ),
    )

    exit_code = cli_main.main(["status", "--json"])
    payload = _payload(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["error_code"] == "cli_usage_error"
    assert payload["stage"] == "dispatch"


def test_main_json_unexpected_exception_emits_broken_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_main.cli,
        "main",
        lambda args, prog_name, standalone_mode: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code = cli_main.main(["status", "--json"])
    payload = _payload(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["error_code"] == "broken_environment"
    assert payload["compatibility"]["exception"] == "RuntimeError('boom')"


def test_support_run_json_preserves_child_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(
        cli_main,
        "run_support_command_impl",
        lambda **kwargs: ({"session_id": "session-1", "child_exit_code": 7}, 7),
    )

    result = runner.invoke(
        cli_main.cli,
        ["support", "run", "--json", "--", "search", "needle", "--json"],
    )
    payload = _payload(result.output)

    assert result.exit_code == 7
    assert payload["session_id"] == "session-1"
    assert payload["child_exit_code"] == 7


def test_support_collect_wraps_support_contract_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    def raise_contract_error(**kwargs):
        raise cli_main.SupportContractError("missing session", code="support_session_missing")

    monkeypatch.setattr(cli_main, "collect_support_bundle_impl", raise_contract_error)

    result = runner.invoke(cli_main.cli, ["support", "collect", "--json", "--session", "missing"])
    payload = _payload(result.output)

    assert result.exit_code == 1
    assert payload["error"]["code"] == "support_session_missing"
    assert payload["failure_codes"] == ["support_session_missing"]


def test_artifact_validate_and_restore_routes_emit_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    monkeypatch.setattr(
        cli_main,
        "_validate_artifact_archive",
        lambda artifact_path, verify_file_hashes: {
            "validated": True,
            "artifact_path": artifact_path,
            "verify_file_hashes": verify_file_hashes,
        },
    )
    monkeypatch.setattr(
        cli_main,
        "_restore_artifact_archive",
        lambda artifact_path, destination_dir, overwrite, verify_file_hashes, require_provenance=False, expected_manifest_sha256=None: {
            "restored": True,
            "artifact_path": artifact_path,
            "destination_dir": destination_dir,
            "overwrite": overwrite,
            "verify_file_hashes": verify_file_hashes,
            "require_provenance": require_provenance,
            "expected_manifest_sha256": expected_manifest_sha256,
        },
    )
    monkeypatch.setattr(
        cli_main,
        "_load_config",
        lambda config_path: GloggurConfig(cache_dir=str(tmp_path / "cache")),
    )

    validate_result = runner.invoke(
        cli_main.cli,
        ["artifact", "validate", "--json", "--artifact", "artifact.tar.gz"],
    )
    restore_result = runner.invoke(
        cli_main.cli,
        ["artifact", "restore", "--json", "--artifact", "artifact.tar.gz", "--overwrite"],
    )

    validate_payload = _payload(validate_result.output)
    restore_payload = _payload(restore_result.output)
    assert validate_result.exit_code == 0
    assert restore_result.exit_code == 0
    assert validate_payload["validated"] is True
    assert validate_payload["verify_file_hashes"] is True
    assert restore_payload["restored"] is True
    assert restore_payload["overwrite"] is True


def test_watch_start_json_rejects_conflicting_mode_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_main.cli,
        ["watch", "start", "--json", "--foreground", "--daemon"],
    )
    payload = _payload(result.output)

    assert result.exit_code == 1
    assert payload["error"]["code"] == "watch_mode_conflict"
    assert payload["failure_codes"] == ["watch_mode_conflict"]


def test_coverage_import_unknown_importer_returns_contract_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    source_path = tmp_path / "coverage.json"
    output_path = tmp_path / "gloggur-coverage.json"
    source_path.write_text("{}", encoding="utf8")
    monkeypatch.setattr(
        cli_main,
        "_load_config",
        lambda config_path: GloggurConfig(cache_dir=str(tmp_path / "cache")),
    )
    monkeypatch.setattr(
        cli_main,
        "create_coverage_importer",
        lambda config, importer_id: (_ for _ in ()).throw(AdapterResolutionError("missing")),
    )

    result = runner.invoke(
        cli_main.cli,
        [
            "coverage",
            "import",
            str(source_path),
            "--output",
            str(output_path),
            "--importer",
            "missing",
            "--json",
        ],
    )
    payload = _payload(result.output)

    assert result.exit_code == 1
    assert payload["error"]["code"] == "coverage_file_invalid"
    assert payload["failure_codes"] == ["coverage_file_invalid"]


def test_search_json_debug_router_fails_when_all_backends_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    config = GloggurConfig(cache_dir=str(tmp_path / "cache"), embedding_provider="test")

    class FakePack:
        hits: list[object] = []
        debug = {
            "backend_scores": {"semantic": 0.2},
            "backend_errors": {"semantic": "timed out"},
        }

        def to_dict(self, include_debug: bool = False) -> dict[str, object]:
            return {
                "summary": {"warning_codes": []},
                "hits": [],
                "debug": dict(self.debug),
            }

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def search(self, **kwargs) -> FakePack:
            return FakePack()

    monkeypatch.setattr(
        cli_main,
        "_create_runtime",
        lambda **kwargs: (config, object(), object()),
    )
    monkeypatch.setattr(
        cli_main,
        "_build_search_health_snapshot",
        lambda *args, **kwargs: {
            "entrypoint": "search_cli_v2",
            "contract_version": "contextpack_v2",
            "needs_reindex": False,
            "reindex_reason": None,
            "resume_contract": {},
            "warning_codes": [],
            "semantic_search_allowed": True,
            "search_integrity": None,
            "expected_index_profile": "test",
            "cached_index_profile": "test",
        },
    )
    monkeypatch.setattr(cli_main, "_create_metadata_store", lambda config: object())
    monkeypatch.setattr(
        cli_main, "_create_embedding_provider_for_command", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_main, "_resolve_router_repo_root", lambda metadata_store, fallback: tmp_path
    )
    monkeypatch.setattr(cli_main, "SymbolIndexStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_main, "load_search_router_config", lambda repo_root: object())
    monkeypatch.setattr(cli_main, "SearchRouter", FakeRouter)

    result = runner.invoke(
        cli_main.cli,
        ["search", "needle", "--json", "--debug-router"],
    )
    payload = _payload(result.output)

    assert result.exit_code == 1
    assert payload["error"]["code"] == "search_router_backends_failed"
    assert payload["failure_codes"] == ["search_router_backends_failed"]
