from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_reference_agent(args: list[str], *, timeout: float = 240.0) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "scripts/run_reference_agent_eval.py", "--format", "json", *args]
    return subprocess.run(
        command,
        cwd=str(repo_root),
        env={**os.environ, "GLOGGUR_EMBEDDING_PROVIDER": "test"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_reference_agent_run_mode_passes_end_to_end() -> None:
    completed = _run_reference_agent(["--mode", "run", "--query", "add numbers token"])
    assert completed.returncode == 0, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["mode"] == "run"
    assert payload["ok"] is True
    result = payload["result"]
    assert result["status"] == "grounded"
    assert int(result["attempts_used"]) >= 1
    assert isinstance(result["logs"], list) and result["logs"]
    assert result["top_symbol"] == "add_numbers"


def test_reference_agent_eval_fails_nonzero_below_threshold() -> None:
    completed = _run_reference_agent(
        [
            "--mode",
            "eval",
            "--top-k",
            "1",
            "--max-retries",
            "0",
            "--evidence-min-items",
            "2",
            "--min-pass-rate",
            "0.5",
        ]
    )
    assert completed.returncode == 1, f"{completed.stderr}\n{completed.stdout}"
    payload = json.loads(completed.stdout)

    assert payload["mode"] == "eval"
    assert payload["ok"] is False
    summary = payload["summary"]
    assert summary["required_pass_rate"] == 0.5
    assert float(summary["pass_rate"]) < float(summary["required_pass_rate"])
    failure = payload["failure"]
    assert failure["code"] == "agent_eval_threshold_failed"
