#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_GLOGGUR = REPO_ROOT / "scripts" / "gloggur"
VALID_WATCH_STATUSES = {"stopped", "running", "running_with_errors"}


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return

    if payload.get("ok") is True:
        print("startup readiness check passed", file=sys.stderr)
        return

    error = payload.get("error", {})
    assert isinstance(error, dict)
    print(
        f"startup readiness check failed ({error.get('code')}): {error.get('detail')}",
        file=sys.stderr,
    )
    remediation = error.get("remediation", [])
    if isinstance(remediation, list) and remediation:
        print("Remediation:", file=sys.stderr)
        for index, step in enumerate(remediation, start=1):
            print(f"{index}. {step}", file=sys.stderr)


def _probe(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _parse_json_output(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    output = (completed.stdout or "").strip()
    if not output:
        raise ValueError("command returned empty stdout")
    payload = json.loads(output)
    if not isinstance(payload, dict):
        raise ValueError("command did not return a JSON object")
    return payload


def _failure(
    code: str,
    detail: str,
    *,
    command: list[str] | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    remediation = {
        "startup_status_probe_failed": [
            "Run `scripts/gloggur status --json` directly and fix the reported startup or cache issue.",
            "Re-run `scripts/bootstrap_gloggur_env.sh` if the environment is incomplete.",
        ],
        "startup_watch_status_probe_failed": [
            "Run `scripts/gloggur watch status --json` directly and inspect the reported watch runtime state.",
            "Run `scripts/gloggur watch stop --json` to clear stale runtime state before retrying.",
        ],
        "startup_watch_payload_invalid": [
            "Repair the watch runtime payload so `scripts/gloggur watch status --json` returns a valid JSON contract.",
            "Remove stale watch runtime files from `.gloggur-cache` and retry bootstrap if needed.",
        ],
        "startup_watch_state_contradictory": [
            "Run `scripts/gloggur watch stop --json` to clear stale daemon state.",
            "If the contradiction persists, remove stale watch runtime files from `.gloggur-cache` and retry.",
        ],
    }[code]
    error: dict[str, object] = {
        "code": code,
        "detail": detail,
        "remediation": remediation,
    }
    if command is not None:
        error["command"] = command
    if payload is not None:
        error["payload"] = payload
    return {"ok": False, "error": error}


def _validate_status_payload(payload: dict[str, object]) -> None:
    if not isinstance(payload.get("needs_reindex"), bool):
        raise ValueError("status payload is missing boolean `needs_reindex`")
    if not isinstance(payload.get("resume_decision"), str):
        raise ValueError("status payload is missing string `resume_decision`")


def _coerce_failure_count(payload: dict[str, object], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool):
        raise ValueError(f"watch payload field `{key}` must not be boolean")
    if value is None:
        return 0
    if not isinstance(value, int):
        raise ValueError(f"watch payload field `{key}` must be an integer")
    return value


def _has_failure_signals(payload: dict[str, object]) -> bool:
    failure_codes = payload.get("failure_codes")
    if failure_codes is not None and not isinstance(failure_codes, list):
        raise ValueError("watch payload field `failure_codes` must be a list when present")
    error_count = _coerce_failure_count(payload, "error_count")
    failed = _coerce_failure_count(payload, "failed")
    return bool(failure_codes) or error_count > 0 or failed > 0


def _validate_watch_payload(payload: dict[str, object]) -> None:
    status = payload.get("status")
    running = payload.get("running")
    pid = payload.get("pid")

    if not isinstance(status, str):
        raise ValueError("watch payload is missing string `status`")
    if status not in VALID_WATCH_STATUSES:
        raise ValueError(f"watch payload status `{status}` is unsupported")
    if not isinstance(running, bool):
        raise ValueError("watch payload is missing boolean `running`")
    if pid is not None and not isinstance(pid, int):
        raise ValueError("watch payload field `pid` must be an integer or null")

    has_failures = _has_failure_signals(payload)

    if status == "stopped":
        if running:
            raise RuntimeError("watch payload reports `stopped` while `running=true`")
        if pid is not None:
            raise RuntimeError("watch payload reports `stopped` with a live pid value")
        if has_failures:
            raise RuntimeError("watch payload reports `stopped` but still carries failure signals")
        return

    if not running:
        raise RuntimeError(f"watch payload reports `{status}` while `running=false`")
    if pid is None or pid <= 0:
        raise RuntimeError(f"watch payload reports `{status}` without a positive pid")

    if status == "running" and has_failures:
        raise RuntimeError("watch payload reports `running` while failure signals are present")
    if status == "running_with_errors" and not has_failures:
        raise RuntimeError("watch payload reports `running_with_errors` without failure signals")


def run_startup_readiness() -> tuple[int, dict[str, object]]:
    status_command = [str(SCRIPTS_GLOGGUR), "status", "--json"]
    status_completed = _probe(status_command)
    if status_completed.returncode != 0:
        detail = (status_completed.stderr or status_completed.stdout or "").strip() or "status probe failed"
        return 1, _failure("startup_status_probe_failed", detail, command=status_command)

    try:
        status_payload = _parse_json_output(status_completed)
        _validate_status_payload(status_payload)
    except (ValueError, json.JSONDecodeError) as exc:
        return 1, _failure(
            "startup_status_probe_failed",
            f"invalid status payload: {exc}",
            command=status_command,
        )

    watch_command = [str(SCRIPTS_GLOGGUR), "watch", "status", "--json"]
    watch_completed = _probe(watch_command)
    if watch_completed.returncode != 0:
        detail = (watch_completed.stderr or watch_completed.stdout or "").strip() or "watch status probe failed"
        return 1, _failure("startup_watch_status_probe_failed", detail, command=watch_command)

    try:
        watch_payload = _parse_json_output(watch_completed)
    except (ValueError, json.JSONDecodeError) as exc:
        return 1, _failure(
            "startup_watch_payload_invalid",
            f"invalid watch status payload: {exc}",
            command=watch_command,
        )

    try:
        _validate_watch_payload(watch_payload)
    except ValueError as exc:
        return 1, _failure(
            "startup_watch_payload_invalid",
            str(exc),
            command=watch_command,
            payload=watch_payload,
        )
    except RuntimeError as exc:
        return 1, _failure(
            "startup_watch_state_contradictory",
            str(exc),
            command=watch_command,
            payload=watch_payload,
        )

    payload = {
        "ok": True,
        "status_probe": {
            "command": status_command,
            "needs_reindex": status_payload["needs_reindex"],
            "resume_decision": status_payload["resume_decision"],
        },
        "watch_probe": {
            "command": watch_command,
            "status": watch_payload["status"],
            "running": watch_payload["running"],
            "pid": watch_payload.get("pid"),
        },
    }
    return 0, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate startup readiness for a local gloggur worktree.")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args()

    exit_code, payload = run_startup_readiness()
    _emit(payload, as_json=args.format == "json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
