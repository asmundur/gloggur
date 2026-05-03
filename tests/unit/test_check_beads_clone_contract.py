from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from scripts.check_beads_clone_contract import (
    EXPECTED_CLONE_CONTRACT,
    check_beads_clone_contract,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")


def _payload_mapping(payload: dict[str, object], key: str) -> dict[str, Any]:
    return cast(dict[str, Any], payload[key])


def _payload_list(payload: dict[str, object], key: str) -> list[str]:
    return cast(list[str], payload[key])


def _write_repo_contract(repo_root: Path, *, issue_count: int = 2) -> None:
    beads_dir = repo_root / ".beads"
    beads_dir.mkdir(parents=True, exist_ok=True)
    _write(beads_dir / "config.yaml", "issue-prefix: gloggur\n")
    _write(beads_dir / "clone-contract.json", json.dumps(EXPECTED_CLONE_CONTRACT, indent=2) + "\n")
    _write(
        beads_dir / ".gitignore",
        "\n".join(
            (
                ".beads-credential-key",
                "metadata.json",
                "README.md",
                "hooks/",
                "dolt-server.lock",
                "dolt-server.log",
                "dolt-server.pid",
                "dolt-server.port",
                "backup/",
                "",
            )
        ),
    )
    issues = [
        json.dumps({"id": f"gloggur-{index}", "title": f"issue {index}"})
        for index in range(issue_count)
    ]
    _write(beads_dir / "issues.jsonl", "\n".join(issues) + "\n")


def _write_fake_bd(
    path: Path,
    *,
    export_payload: str | None = None,
    fail_message: str | None = None,
) -> None:
    if export_payload is not None and fail_message is not None:
        raise ValueError("Specify export_payload or fail_message, not both")
    path.parent.mkdir(parents=True, exist_ok=True)
    if export_payload is not None:
        script = f"""#!/usr/bin/env python3
import pathlib
import sys

if sys.argv[1:] != ["export", "-o", sys.argv[3]]:
    raise SystemExit(2)
path = pathlib.Path(sys.argv[3])
path.write_text({export_payload!r}, encoding="utf8")
"""
    else:
        script = f"""#!/usr/bin/env python3
import sys

sys.stderr.write({fail_message!r} + "\\n")
raise SystemExit(1)
"""
    path.write_text(script, encoding="utf8")
    path.chmod(0o755)


def test_beads_clone_contract_passes_when_contract_files_are_present_and_parity_is_skipped(
    tmp_path: Path,
) -> None:
    _write_repo_contract(tmp_path)

    payload = check_beads_clone_contract(tmp_path, bd_command="definitely-missing-bd")

    assert payload["ok"] is True
    assert payload["missing_files"] == []
    assert payload["config_errors"] == []
    assert payload["clone_contract_errors"] == []
    assert payload["missing_gitignore_entries"] == []
    parity = _payload_mapping(payload, "parity")
    assert parity["status"] == "skipped"
    assert parity["reason"] == "bd_missing"


def test_beads_clone_contract_passes_when_export_matches_tracked_jsonl(tmp_path: Path) -> None:
    _write_repo_contract(tmp_path, issue_count=3)
    fake_bd = tmp_path / "bin" / "bd"
    export_payload = (tmp_path / ".beads" / "issues.jsonl").read_text(encoding="utf8")
    _write_fake_bd(fake_bd, export_payload=export_payload)

    payload = check_beads_clone_contract(tmp_path, bd_command=str(fake_bd))

    assert payload["ok"] is True
    parity = _payload_mapping(payload, "parity")
    assert parity["status"] == "verified"
    assert parity["tracked_count"] == 3


def test_beads_clone_contract_fails_when_tracked_contract_files_are_invalid(tmp_path: Path) -> None:
    _write_repo_contract(tmp_path)
    _write(tmp_path / ".beads" / "config.yaml", "issue-prefix: wrong\n")
    _write(
        tmp_path / ".beads" / "clone-contract.json",
        json.dumps({**EXPECTED_CLONE_CONTRACT, "read_probe": "bd list --json"}, indent=2) + "\n",
    )
    _write(tmp_path / ".beads" / ".gitignore", "metadata.json\nREADME.md\n")

    payload = check_beads_clone_contract(tmp_path, bd_command="definitely-missing-bd")

    assert payload["ok"] is False
    failure = _payload_mapping(payload, "failure")
    config_errors = _payload_list(payload, "config_errors")
    clone_contract_errors = _payload_list(payload, "clone_contract_errors")
    assert failure["code"] == "beads_clone_contract_violation"
    assert "missing config snippet: issue-prefix: gloggur" in config_errors
    assert any(
        error.startswith("read_probe: expected") for error in clone_contract_errors
    )
    assert payload["missing_gitignore_entries"] == [
        ".beads-credential-key",
        "hooks/",
        "dolt-server.lock",
        "dolt-server.log",
        "dolt-server.pid",
        "dolt-server.port",
        "backup/",
    ]


def test_beads_clone_contract_fails_when_live_export_drifts_from_tracked_jsonl(
    tmp_path: Path,
) -> None:
    _write_repo_contract(tmp_path, issue_count=2)
    fake_bd = tmp_path / "bin" / "bd"
    _write_fake_bd(fake_bd, export_payload='{"id":"gloggur-1"}\n')

    payload = check_beads_clone_contract(tmp_path, bd_command=str(fake_bd))

    assert payload["ok"] is False
    failure = _payload_mapping(payload, "failure")
    parity = _payload_mapping(payload, "parity")
    assert failure["code"] == "beads_clone_contract_parity_mismatch"
    assert parity["status"] == "mismatch"
    assert parity["tracked_count"] == 2
    assert parity["exported_count"] == 1


def test_beads_clone_contract_fails_when_bd_export_errors_for_non_clone_reason(
    tmp_path: Path,
) -> None:
    _write_repo_contract(tmp_path)
    fake_bd = tmp_path / "bin" / "bd"
    _write_fake_bd(fake_bd, fail_message="panic: export exploded")

    payload = check_beads_clone_contract(tmp_path, bd_command=str(fake_bd))

    assert payload["ok"] is False
    failure = _payload_mapping(payload, "failure")
    parity = _payload_mapping(payload, "parity")
    assert failure["code"] == "beads_clone_contract_parity_check_failed"
    assert parity["status"] == "error"
    assert parity["reason"] == "export_failed"
