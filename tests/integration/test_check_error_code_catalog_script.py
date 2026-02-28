from __future__ import annotations

import json
import subprocess
import sys


def test_check_error_code_catalog_script_passes_for_repo_docs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_error_code_catalog.py", "--format", "json"],
        text=True,
        capture_output=True,
        timeout=120.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
