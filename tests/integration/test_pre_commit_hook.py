from __future__ import annotations

import subprocess
from pathlib import Path


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_pre_commit_hook_allows_metadata_deletion_while_refreshing_beads_exports(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert _run(["git", "init", "-q"], cwd=repo_root).returncode == 0
    assert _run(["git", "config", "user.name", "Codex"], cwd=repo_root).returncode == 0
    assert _run(["git", "config", "user.email", "codex@example.com"], cwd=repo_root).returncode == 0

    hook_source = Path(__file__).resolve().parents[2] / ".githooks" / "pre-commit"
    hook_path = repo_root / ".githooks" / "pre-commit"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(hook_source.read_text(encoding="utf8"), encoding="utf8")
    hook_path.chmod(0o755)

    beads_dir = repo_root / ".beads"
    beads_dir.mkdir()
    (beads_dir / ".gitignore").write_text("metadata.json\n", encoding="utf8")
    (beads_dir / "issues.jsonl").write_text('{"id":"bd-1"}\n', encoding="utf8")
    (beads_dir / "interactions.jsonl").write_text("", encoding="utf8")
    (beads_dir / "metadata.json").write_text('{"database":"dolt"}\n', encoding="utf8")

    assert _run(["git", "add", "."], cwd=repo_root).returncode == 0
    assert _run(["git", "add", "-f", ".beads/metadata.json"], cwd=repo_root).returncode == 0
    assert _run(["git", "commit", "-qm", "initial"], cwd=repo_root).returncode == 0

    (beads_dir / "metadata.json").unlink()
    assert _run(["git", "add", "-u", "--", ".beads/metadata.json"], cwd=repo_root).returncode == 0

    fake_bin = repo_root / "fake-bin"
    fake_bin.mkdir()
    bd_path = fake_bin / "bd"
    bd_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-}"
shift || true

case "$cmd" in
  export)
    if [[ "${1:-}" != "-o" || "${2:-}" != ".beads/issues.jsonl" ]]; then
      echo "unexpected bd export args: $*" >&2
      exit 1
    fi
    printf '{"id":"bd-1"}\\n{"id":"bd-2"}\\n' > .beads/issues.jsonl
    ;;
  hooks)
    subcmd="${1:-}"
    shift || true
    case "$subcmd" in
      run)
        exit 0
        ;;
      *)
        echo "unexpected bd hooks subcommand: $subcmd" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "unexpected bd command: $cmd" >&2
    exit 1
    ;;
esac
""",
        encoding="utf8",
    )
    bd_path.chmod(0o755)

    env = {
        "PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}",
        "HOME": str(tmp_path / "home"),
    }

    completed = _run([str(hook_path)], cwd=repo_root, env=env)

    assert completed.returncode == 0, completed.stderr
    assert "Refreshing Beads tracker exports..." in completed.stdout
    assert "Auto-staged .beads/issues.jsonl" in completed.stdout
    assert ".beads/metadata.json" not in completed.stderr

    status = _run(["git", "status", "--short"], cwd=repo_root)
    assert status.returncode == 0
    assert "M  .beads/issues.jsonl" in status.stdout
    assert "D  .beads/metadata.json" in status.stdout
