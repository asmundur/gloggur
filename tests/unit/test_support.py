from __future__ import annotations

import json
from pathlib import Path

import pytest

import gloggur.support as support_module


def test_sanitize_object_redacts_secret_values_and_local_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    payload = {
        "openai_api_key": "secret-value",
        "nested": {
            "token": "abc123",
            "path": str(repo_root / "src" / "main.py"),
            "home": str(Path.home() / "work" / "file.txt"),
        },
    }

    sanitized = support_module._sanitize_object(payload, repo_root)

    assert sanitized["openai_api_key"] == "<REDACTED>"
    nested = sanitized["nested"]
    assert isinstance(nested, dict)
    assert nested["token"] == "<REDACTED>"
    assert "<REPO_ROOT>" in str(nested["path"])
    assert "<HOME>" in str(nested["home"])


def test_snapshot_optional_text_file_tails_and_records_capture_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    session_dir = repo_root / ".gloggur" / "support" / "sessions" / "session-1"
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True)
    source_path = tmp_path / "watch.log"
    source_path.write_text("0123456789abcdef", encoding="utf8")
    monkeypatch.setattr(support_module, "DEFAULT_TAIL_BYTES", 6)

    support_module._snapshot_optional_text_file(
        source_path=source_path,
        destination_path=logs_dir / "watch.log",
        repo_root=repo_root,
        session_dir=session_dir,
    )

    assert (logs_dir / "watch.log").read_text(encoding="utf8") == "abcdef"
    capture_meta = json.loads(
        (session_dir / "diagnostics" / "capture_meta.json").read_text(encoding="utf8")
    )
    entry = capture_meta["logs/watch.log"]
    assert entry["truncated"] is True
    assert entry["source_bytes"] == len("0123456789abcdef".encode("utf8"))


def test_extract_failure_contract_prefers_compatibility_payload() -> None:
    error_code, failure_codes = support_module._extract_failure_contract(
        {
            "ok": False,
            "error_code": "missing_package",
            "compatibility": {
                "error": {
                    "code": "cli_usage_error",
                },
                "failure_codes": ["cli_usage_error"],
            },
        }
    )

    assert error_code == "cli_usage_error"
    assert failure_codes == ["cli_usage_error"]


def test_resolve_existing_session_dir_rejects_invalid_session_id(tmp_path: Path) -> None:
    roots = support_module.SupportRoots(
        repo_root=tmp_path,
        support_root=tmp_path / ".gloggur" / "support",
        sessions_root=tmp_path / ".gloggur" / "support" / "sessions",
        bundles_root=tmp_path / ".gloggur" / "support" / "bundles",
    )

    with pytest.raises(support_module.SupportContractError) as error:
        support_module._resolve_existing_session_dir(roots, "../escape")

    assert error.value.code == "support_session_invalid"


@pytest.mark.parametrize(
    "child_args",
    [
        (),
        ("gloggur", "status"),
        ("support", "collect"),
        ("-h",),
        ("./scripts/gloggur", "status"),
    ],
)
def test_validate_child_args_rejects_invalid_forms(child_args: tuple[str, ...]) -> None:
    with pytest.raises(support_module.SupportContractError) as error:
        support_module._validate_child_args(child_args)

    assert error.value.code == "support_command_invalid"


def test_extract_config_path_supports_split_and_inline_forms() -> None:
    assert support_module._extract_config_path(["status", "--config", "cfg.yaml"]) == "cfg.yaml"
    assert support_module._extract_config_path(["status", "--config=cfg.json"]) == "cfg.json"
    assert support_module._extract_config_path(["status", "--json"]) is None


def test_parse_first_json_object_returns_none_for_non_json_prefix() -> None:
    assert support_module._parse_first_json_object("plain text only") is None
    assert support_module._parse_first_json_object("prefix {not-json") is None


def test_read_session_payload_rejects_invalid_json(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{not-json", encoding="utf8")

    with pytest.raises(support_module.SupportContractError) as error:
        support_module._read_session_payload(session_dir)

    assert error.value.code == "support_session_invalid"


def test_sanitize_optional_string_preserves_none_and_sanitizes_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert support_module._sanitize_optional_string(None, repo_root) is None
    sanitized = support_module._sanitize_optional_string(str(repo_root / "config.yaml"), repo_root)
    assert sanitized == "<REPO_ROOT>/config.yaml"
