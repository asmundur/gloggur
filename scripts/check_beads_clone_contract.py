from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

EXPECTED_CLONE_CONTRACT = {
    "mode": "bootstrap_required",
    "backend": "dolt",
    "issue_prefix": "gloggur",
    "jsonl_export": "issues.jsonl",
    "bootstrap_commands": [
        "bd bootstrap --yes --json",
        "git config core.hooksPath .githooks",
    ],
    "read_probe": "bd status --json",
    "stale_runtime_recovery": {
        "verify_hooks_path": "git config --get core.hooksPath",
        "local_pins": [
            ".beads/dolt-server.port",
        ],
        "clear_when_unowned": [
            ".beads/dolt-server.pid",
            ".beads/dolt-server.log",
            ".beads/dolt-server.lock",
            ".beads/dolt-server.activity",
        ],
        "retry_probes": [
            "bd status --json",
            "bd ready --json",
        ],
    },
}

REQUIRED_GITIGNORE_ENTRIES = [
    ".beads-credential-key",
    "metadata.json",
    "README.md",
    "hooks/",
    "dolt-server.lock",
    "dolt-server.log",
    "dolt-server.pid",
    "dolt-server.port",
    "backup/",
]

REQUIRED_CONFIG_SNIPPETS = [
    "issue-prefix: gloggur",
]


def _command_exists(command: str) -> bool:
    command_path = Path(command)
    if command_path.is_file():
        return True
    return shutil.which(command) is not None


def _compare_export_parity(
    repo_root: Path,
    issues_path: Path,
    bd_command: str,
) -> dict[str, object]:
    if not _command_exists(bd_command):
        return {"status": "skipped", "reason": "bd_missing"}

    with tempfile.TemporaryDirectory(prefix="gloggur-beads-export-") as temp_dir:
        export_path = Path(temp_dir) / "issues.jsonl"
        completed = subprocess.run(
            [bd_command, "export", "-o", str(export_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        output = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        if completed.returncode != 0:
            if "no beads database found" in output:
                return {"status": "skipped", "reason": "no_local_db"}
            return {
                "status": "error",
                "reason": "export_failed",
                "detail": output,
            }

        tracked = issues_path.read_text(encoding="utf8")
        exported = export_path.read_text(encoding="utf8")
        if tracked != exported:
            return {
                "status": "mismatch",
                "reason": "export_differs",
                "tracked_count": len(tracked.splitlines()),
                "exported_count": len(exported.splitlines()),
            }
        return {
            "status": "verified",
            "tracked_count": len(tracked.splitlines()),
            "exported_count": len(exported.splitlines()),
        }


def check_beads_clone_contract(repo_root: Path, *, bd_command: str = "bd") -> dict[str, object]:
    beads_dir = repo_root / ".beads"
    config_path = beads_dir / "config.yaml"
    clone_contract_path = beads_dir / "clone-contract.json"
    gitignore_path = beads_dir / ".gitignore"
    issues_path = beads_dir / "issues.jsonl"

    missing_files = [
        str(path.relative_to(repo_root))
        for path in (config_path, clone_contract_path, gitignore_path, issues_path)
        if not path.exists()
    ]

    config_errors: list[str] = []
    clone_contract_errors: list[str] = []
    missing_gitignore_entries: list[str] = []

    if config_path.exists():
        config_text = config_path.read_text(encoding="utf8")
        for snippet in REQUIRED_CONFIG_SNIPPETS:
            if snippet not in config_text:
                config_errors.append(f"missing config snippet: {snippet}")

    if clone_contract_path.exists():
        try:
            clone_contract = json.loads(clone_contract_path.read_text(encoding="utf8"))
        except json.JSONDecodeError as exc:
            clone_contract_errors.append(f"invalid JSON: {exc}")
        else:
            for key, expected_value in EXPECTED_CLONE_CONTRACT.items():
                if clone_contract.get(key) != expected_value:
                    clone_contract_errors.append(
                        f"{key}: expected {expected_value!r}, found {clone_contract.get(key)!r}"
                    )
            for unexpected_key in sorted(set(clone_contract) - set(EXPECTED_CLONE_CONTRACT)):
                clone_contract_errors.append(f"unexpected key present: {unexpected_key}")

    if gitignore_path.exists():
        gitignore_text = gitignore_path.read_text(encoding="utf8")
        missing_gitignore_entries = [
            entry for entry in REQUIRED_GITIGNORE_ENTRIES if entry not in gitignore_text
        ]

    parity = (
        {"status": "skipped", "reason": "issues_jsonl_missing"}
        if not issues_path.exists()
        else _compare_export_parity(repo_root, issues_path, bd_command)
    )

    ok = (
        not missing_files
        and not config_errors
        and not clone_contract_errors
        and not missing_gitignore_entries
        and parity.get("status") not in {"error", "mismatch"}
    )

    payload: dict[str, object] = {
        "ok": ok,
        "summary": {
            "repo_root": str(repo_root),
            "bd_command": bd_command,
        },
        "missing_files": missing_files,
        "config_errors": config_errors,
        "clone_contract_errors": clone_contract_errors,
        "missing_gitignore_entries": missing_gitignore_entries,
        "parity": parity,
    }

    if not ok:
        failure_code = "beads_clone_contract_violation"
        if parity.get("status") == "mismatch":
            failure_code = "beads_clone_contract_parity_mismatch"
        elif parity.get("status") == "error":
            failure_code = "beads_clone_contract_parity_check_failed"
        payload["failure"] = {
            "code": failure_code,
            "detail": "Beads fresh-clone contract files or live export parity are invalid.",
            "remediation": (
                "Fix .beads/config.yaml, .beads/clone-contract.json, and .beads/.gitignore, "
                "then refresh .beads/issues.jsonl from the live Beads DB before "
                "rerunning the check."
            ),
        }

    return payload


def _render_markdown(payload: dict[str, object]) -> str:
    lines = ["# Beads Clone Contract", ""]
    lines.append(f"- ok: `{payload.get('ok')}`")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        lines.append(f"- repo_root: `{summary.get('repo_root')}`")
        lines.append(f"- bd_command: `{summary.get('bd_command')}`")
    for field in (
        "missing_files",
        "config_errors",
        "clone_contract_errors",
        "missing_gitignore_entries",
    ):
        value = payload.get(field)
        if value:
            lines.append(f"- {field}: `{value}`")
    parity = payload.get("parity")
    if isinstance(parity, dict):
        lines.append(f"- parity: `{parity}`")
    failure = payload.get("failure")
    if isinstance(failure, dict):
        lines.append(f"- failure.code: `{failure.get('code')}`")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the tracked Beads clone contract.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--bd-command", default="bd")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = check_beads_clone_contract(args.repo_root.resolve(), bd_command=args.bd_command)
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload), end="")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
