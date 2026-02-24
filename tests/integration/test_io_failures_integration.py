from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from gloggur.cli.main import cli


def _parse_json_output(output: str) -> dict[str, object]:
    """Parse a JSON payload from click output."""
    start = output.find("{")
    if start == -1:
        raise ValueError(f"No JSON payload found in output: {output!r}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output[start:])
    return payload


@pytest.mark.skipif(os.name == "nt", reason="chmod-based read-only checks are POSIX-specific")
def test_status_fails_when_cache_parent_is_not_writable() -> None:
    """Integration-style check: status should fail loudly when cache path cannot be created."""
    runner = CliRunner()
    root = Path(tempfile.mkdtemp(prefix="gloggur-io-failure-"))
    readonly_parent = root / "readonly"
    readonly_parent.mkdir(parents=True, exist_ok=True)
    readonly_parent.chmod(0o555)
    if os.access(readonly_parent, os.W_OK):
        pytest.skip("Environment does not enforce read-only permissions for this test.")
    try:
        result = runner.invoke(
            cli,
            ["status", "--json"],
            env={"GLOGGUR_CACHE_DIR": str(readonly_parent / "cache")},
        )
        assert result.exit_code == 1
        payload = _parse_json_output(result.output)
        error = payload["error"]
        assert isinstance(error, dict)
        assert str(error["category"]) in {
            "permission_denied",
            "read_only_filesystem",
            "path_not_writable",
        }
    finally:
        readonly_parent.chmod(0o755)
