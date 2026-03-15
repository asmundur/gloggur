from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_path() -> Path:
    return _repo_root() / "scripts" / "grant_gloggur_access.sh"


def _parse_json_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    assert isinstance(payload, dict)
    return payload


def test_access_wrapper_applies_grant_via_cli(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf8")

    env = os.environ.copy()
    env["GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS"] = sys.executable

    completed = subprocess.run(
        [str(_script_path()), str(repo), "--yes", "--json"],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _parse_json_output(completed.stdout)
    assert payload["access_ready"] is True
    assert (repo / ".gloggur-cache").exists()
    assert (repo / ".gloggur" / "access_grants.json").exists()
