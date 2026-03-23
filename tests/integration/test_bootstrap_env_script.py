from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_script() -> Path:
    return _repo_root() / "scripts" / "bootstrap_gloggur_env.sh"


def _prepare_temp_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    cli_dir = repo_root / "src" / "gloggur" / "cli"
    scripts_dir.mkdir(parents=True)
    cli_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / "bootstrap_gloggur_env.sh"
    readiness_path = scripts_dir / "check_startup_readiness.py"
    (repo_root / "pyproject.toml").write_text('[project]\nname = "gloggur-test"\n', encoding="utf8")
    (cli_dir / "main.py").write_text("# test fixture marker\n", encoding="utf8")
    script_path.write_text(_source_script().read_text(encoding="utf8"), encoding="utf8")
    script_path.chmod(0o755)
    readiness_path.write_text(
        (_repo_root() / "scripts" / "check_startup_readiness.py").read_text(encoding="utf8"),
        encoding="utf8",
    )
    readiness_path.chmod(0o755)
    return repo_root, script_path


def _base_bootstrap_env(tmp_path: Path) -> dict[str, str]:
    """Return isolated bootstrap env so integration tests cannot mutate real user wrappers."""
    env = os.environ.copy()
    isolated_home = tmp_path / "home"
    isolated_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(isolated_home)
    env["GLOGGUR_BOOTSTRAP_INSTALL_GLOBAL_WRAPPER"] = "0"
    env["GLOGGUR_BOOTSTRAP_GLOBAL_LINK_DIRS"] = ""
    return env


def _run_bootstrap(
    script_path: Path,
    args: list[str],
    env: dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(script_path), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        check=False,
    )


def _prepare_seed_source(tmp_path: Path) -> Path:
    workspace_root = tmp_path / "seed-workspace"
    cache_dir = workspace_root / ".gloggur-cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "index.db").write_text("seed-index", encoding="utf8")
    (cache_dir / "vectors.json").write_text("{}", encoding="utf8")
    return workspace_root


def _prepare_seed_venv_workspace(tmp_path: Path) -> Path:
    workspace_root = tmp_path / "venv-workspace"
    venv_python = workspace_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(f'#!/usr/bin/env bash\nexec {sys.executable!r} "$@"\n', encoding="utf8")
    venv_python.chmod(0o755)
    return workspace_root


def _prepare_fake_gloggur_wrapper(repo_root: Path) -> tuple[Path, Path]:
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    log_file = repo_root / "gloggur-invocations.log"
    state_file = repo_root / "index-ready.flag"
    wrapper = scripts_dir / "gloggur"
    wrapper_template = """#!/usr/bin/env bash
set -euo pipefail

log_file="${BOOTSTRAP_GLOGGUR_LOG_FILE:-}"
state_file="${BOOTSTRAP_GLOGGUR_STATE_FILE:-__BOOTSTRAP_STATE_FILE__}"
if [[ -n "$log_file" ]]; then
  printf '%s\\n' "$*" >> "$log_file" 2>/dev/null || true
fi

command_name="${1:-}"
shift || true

case "$command_name" in
  status)
    status_call_file="${BOOTSTRAP_GLOGGUR_STATUS_CALL_FILE:-}"
    if [[ -n "$status_call_file" ]]; then
      count=0
      if [[ -f "$status_call_file" ]]; then
        count="$(cat "$status_call_file")"
      fi
      count=$((count + 1))
      printf '%s' "$count" > "$status_call_file"
      fail_on_call="${BOOTSTRAP_GLOGGUR_STATUS_FAIL_ON_CALL:-}"
      if [[ -n "$fail_on_call" && "$count" == "$fail_on_call" ]]; then
        printf '%s\\n' "${BOOTSTRAP_GLOGGUR_STATUS_FAIL_MESSAGE:-forced status failure}" >&2
        exit 1
      fi
    fi
    if [[ -f "$state_file" ]]; then
      printf '{"needs_reindex": false, "resume_decision": "resume_ok"}\\n'
    else
      printf '{"needs_reindex": true, "resume_decision": "reindex_required"}\\n'
    fi
    ;;
  watch)
    subcommand="${1:-}"
    shift || true
    if [[ "$subcommand" != "status" ]]; then
      printf '{"error": "unexpected watch command"}\\n'
      exit 1
    fi
    if [[ "${BOOTSTRAP_GLOGGUR_WATCH_FAIL:-0}" == "1" ]]; then
      printf '%s\\n' "${BOOTSTRAP_GLOGGUR_WATCH_FAIL_MESSAGE:-forced watch failure}" >&2
      exit 1
    fi
    watch_payload="${BOOTSTRAP_GLOGGUR_WATCH_PAYLOAD:-}"
    if [[ -z "$watch_payload" ]]; then
      watch_payload='{"status":"stopped","running":false,"pid":null}'
    fi
    printf '%s\\n' "$watch_payload"
    ;;
  index)
    touch "$state_file"
    printf '{"indexed_files": 1, "indexed_symbols": 1}\\n'
    ;;
  *)
    printf '{"error": "unsupported command"}\\n'
    exit 1
    ;;
esac
"""
    wrapper.write_text(
        wrapper_template.replace("__BOOTSTRAP_STATE_FILE__", str(state_file)),
        encoding="utf8",
    )
    wrapper.chmod(0o755)
    return log_file, state_file


def _prepare_fake_beads_export(repo_root: Path, *, issue_count: int = 3) -> Path:
    beads_dir = repo_root / ".beads"
    beads_dir.mkdir(parents=True, exist_ok=True)
    issues_path = beads_dir / "issues.jsonl"
    lines = [
        json.dumps({"id": f"bd-test-{index}", "title": f"issue {index}"})
        for index in range(issue_count)
    ]
    issues_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return issues_path


def _prepare_fake_bd(repo_root: Path) -> tuple[Path, Path, Path]:
    bin_dir = repo_root / "fake-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log_file = repo_root / "bd-invocations.log"
    state_file = repo_root / "bd-ready.flag"
    wrapper = bin_dir / "bd"
    wrapper_template = """#!/usr/bin/env bash
set -euo pipefail

log_file="${BOOTSTRAP_BD_LOG_FILE:-__BOOTSTRAP_BD_LOG_FILE__}"
state_file="${BOOTSTRAP_BD_STATE_FILE:-__BOOTSTRAP_BD_STATE_FILE__}"

if [[ -n "$log_file" ]]; then
  printf '%s\\n' "$*" >> "$log_file"
fi

command_name="${1:-}"
shift || true

case "$command_name" in
  status)
    if [[ -f "$state_file" ]]; then
      printf '{"summary":{"total_issues":1}}\\n'
      exit 0
    fi
    printf 'Error: no beads database found\\n\\n' >&2
    printf 'Found JSONL file: %s/.beads/issues.jsonl\\n' "$(pwd)" >&2
    printf 'This looks like a fresh clone or JSONL-only project.\\n' >&2
    exit 1
    ;;
  init)
    if [[ "${BOOTSTRAP_BD_INIT_FAIL:-0}" == "1" ]]; then
      printf '%s\\n' "${BOOTSTRAP_BD_INIT_FAIL_MESSAGE:-forced init failure}" >&2
      exit 1
    fi
    touch "$state_file"
    printf '{"ok":true}\\n'
    ;;
  import)
    if [[ "${BOOTSTRAP_BD_IMPORT_FAIL:-0}" == "1" ]]; then
      printf '%s\\n' "${BOOTSTRAP_BD_IMPORT_FAIL_MESSAGE:-forced import failure}" >&2
      exit 1
    fi
    if [[ "${1:-}" != "-i" || $# -lt 2 ]]; then
      printf 'missing import input\\n' >&2
      exit 2
    fi
    input_file="$2"
    count="$(wc -l < "$input_file" | tr -d ' ')"
    printf '%s' "$count" > "${state_file}.count"
    printf 'Import complete: %s created, 0 updated\\n' "$count"
    ;;
  count)
    if [[ "${BOOTSTRAP_BD_COUNT_FAIL:-0}" == "1" ]]; then
      printf '%s\\n' "${BOOTSTRAP_BD_COUNT_FAIL_MESSAGE:-forced count failure}" >&2
      exit 1
    fi
    if [[ -n "${BOOTSTRAP_BD_COUNT_OVERRIDE:-}" ]]; then
      count="${BOOTSTRAP_BD_COUNT_OVERRIDE}"
    elif [[ -f "${state_file}.count" ]]; then
      count="$(cat "${state_file}.count")"
    else
      count="0"
    fi
    printf '{"count":%s}\\n' "$count"
    ;;
  *)
    printf 'unsupported bd command: %s\\n' "$command_name" >&2
    exit 1
    ;;
esac
"""
    wrapper.write_text(
        wrapper_template.replace("__BOOTSTRAP_BD_LOG_FILE__", str(log_file)).replace(
            "__BOOTSTRAP_BD_STATE_FILE__", str(state_file)
        ),
        encoding="utf8",
    )
    wrapper.chmod(0o755)
    return bin_dir, log_file, state_file


def _write_legacy_global_wrapper(wrapper_path: Path) -> None:
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${REPO_ROOT}" ] || [ ! -x "${REPO_ROOT}/scripts/gloggur" ]; then
  echo "gloggur wrapper: run inside a gloggur git worktree." >&2
  exit 1
fi

exec "${REPO_ROOT}/scripts/gloggur" "$@"
""",
        encoding="utf8",
    )
    wrapper_path.chmod(0o755)


def test_bootstrap_can_seed_cache_via_symlink(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_source(tmp_path)
    source_cache = source_workspace / ".gloggur-cache"
    env = _base_bootstrap_env(tmp_path)

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-cache-from",
            str(source_workspace),
            "--seed-cache-mode",
            "symlink",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    target_cache = repo_root / ".gloggur-cache"
    assert target_cache.is_symlink()
    assert target_cache.resolve() == source_cache.resolve()
    assert "Cache seed: symlinked:" in completed.stdout


def test_bootstrap_can_seed_cache_via_copy(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_source(tmp_path)
    source_cache = source_workspace / ".gloggur-cache"
    env = _base_bootstrap_env(tmp_path)

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-cache-from",
            str(source_workspace),
            "--seed-cache-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    target_cache = repo_root / ".gloggur-cache"
    assert target_cache.exists()
    assert not target_cache.is_symlink()
    assert (target_cache / "index.db").read_text(encoding="utf8") == "seed-index"
    assert (target_cache / "vectors.json").read_text(encoding="utf8") == "{}"
    assert (source_cache / "index.db").read_text(encoding="utf8") == "seed-index"
    assert "Cache seed: copied:" in completed.stdout


def test_bootstrap_can_seed_venv_via_symlink_and_auto_skip_install(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    source_venv = source_workspace / ".venv"
    env = _base_bootstrap_env(tmp_path)

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "symlink",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    target_venv = repo_root / ".venv"
    assert target_venv.is_symlink()
    assert target_venv.resolve() == source_venv.resolve()
    assert "Venv seed: symlinked:" in completed.stdout
    assert "Install step: skipped:seeded_venv" in completed.stdout


def test_bootstrap_can_seed_venv_via_copy_and_auto_skip_install(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    source_venv = source_workspace / ".venv"
    env = _base_bootstrap_env(tmp_path)

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    target_venv = repo_root / ".venv"
    assert target_venv.exists()
    assert not target_venv.is_symlink()
    assert (target_venv / "bin" / "python").read_text(encoding="utf8") == (
        source_venv / "bin" / "python"
    ).read_text(encoding="utf8")
    assert "Venv seed: copied:" in completed.stdout
    assert "Install step: skipped:seeded_venv" in completed.stdout


def test_bootstrap_rejects_invalid_seed_mode(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    env = _base_bootstrap_env(tmp_path)

    completed = _run_bootstrap(
        script_path=script_path,
        args=["--seed-cache-mode", "invalid"],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 2
    assert "Invalid --seed-cache-mode" in completed.stderr


def test_bootstrap_ensures_index_is_current_during_environment_setup(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    env = _base_bootstrap_env(tmp_path)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["BOOTSTRAP_GLOGGUR_STATUS_CALL_FILE"] = str(repo_root / "status-call-count.txt")

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    assert log_file.exists(), "bootstrap should invoke scripts/gloggur for index freshness checks"
    invocations = log_file.read_text(encoding="utf8").splitlines()
    assert invocations == [
        "status --json",
        "index . --json",
        "status --json",
        "status --json",
        "watch status --json",
    ]


def test_bootstrap_fails_when_startup_status_probe_fails(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    state_file.write_text("ready", encoding="utf8")
    env = _base_bootstrap_env(tmp_path)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["BOOTSTRAP_GLOGGUR_STATUS_CALL_FILE"] = str(repo_root / "status-call-count.txt")
    env["BOOTSTRAP_GLOGGUR_STATUS_FAIL_ON_CALL"] = "2"
    env["BOOTSTRAP_GLOGGUR_STATUS_FAIL_MESSAGE"] = "status probe exploded"

    completed = _run_bootstrap(
        script_path=script_path, args=["--skip-install"], env=env, cwd=repo_root
    )

    assert completed.returncode == 1
    assert "startup_status_probe_failed" in completed.stderr
    invocations = log_file.read_text(encoding="utf8").splitlines()
    assert invocations == ["status --json", "status --json"]


def test_bootstrap_fails_when_startup_watch_status_probe_fails(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    state_file.write_text("ready", encoding="utf8")
    env = _base_bootstrap_env(tmp_path)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["BOOTSTRAP_GLOGGUR_WATCH_FAIL"] = "1"
    env["BOOTSTRAP_GLOGGUR_WATCH_FAIL_MESSAGE"] = "watch probe exploded"

    completed = _run_bootstrap(
        script_path=script_path, args=["--skip-install"], env=env, cwd=repo_root
    )

    assert completed.returncode == 1
    assert "startup_watch_status_probe_failed" in completed.stderr
    invocations = log_file.read_text(encoding="utf8").splitlines()
    assert invocations == ["status --json", "status --json", "watch status --json"]


def test_bootstrap_fails_when_startup_watch_payload_is_contradictory(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    state_file.write_text("ready", encoding="utf8")
    env = _base_bootstrap_env(tmp_path)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["BOOTSTRAP_GLOGGUR_WATCH_PAYLOAD"] = json.dumps(
        {"status": "running", "running": False, "pid": 4321}
    )

    completed = _run_bootstrap(
        script_path=script_path, args=["--skip-install"], env=env, cwd=repo_root
    )

    assert completed.returncode == 1
    assert (
        "startup_watch_state_contradictory" in completed.stderr
        or "startup_watch_payload_invalid" in completed.stderr
    )
    invocations = log_file.read_text(encoding="utf8").splitlines()
    assert invocations == ["status --json", "status --json", "watch status --json"]


def test_bootstrap_bootstraps_beads_from_tracked_jsonl_on_fresh_clone(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    gloggur_log_file, gloggur_state_file = _prepare_fake_gloggur_wrapper(repo_root)
    gloggur_state_file.write_text("ready", encoding="utf8")
    _prepare_fake_beads_export(repo_root, issue_count=4)
    fake_bd_dir, bd_log_file, _ = _prepare_fake_bd(repo_root)

    env = _base_bootstrap_env(tmp_path)
    env["PATH"] = f"{fake_bd_dir}:{env['PATH']}"
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(gloggur_log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(gloggur_state_file)
    env["BOOTSTRAP_BD_LOG_FILE"] = str(bd_log_file)

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Beads readiness: bootstrapped:4" in completed.stdout
    assert bd_log_file.read_text(encoding="utf8").splitlines() == [
        "status --json",
        "init -p bd --json",
        "import -i .beads/issues.jsonl --json",
        "status --json",
        "count --json",
    ]


def test_bootstrap_skips_beads_bootstrap_when_bd_status_is_already_ready(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    gloggur_log_file, gloggur_state_file = _prepare_fake_gloggur_wrapper(repo_root)
    gloggur_state_file.write_text("ready", encoding="utf8")
    _prepare_fake_beads_export(repo_root, issue_count=2)
    fake_bd_dir, bd_log_file, bd_state_file = _prepare_fake_bd(repo_root)
    bd_state_file.write_text("ready", encoding="utf8")

    env = _base_bootstrap_env(tmp_path)
    env["PATH"] = f"{fake_bd_dir}:{env['PATH']}"
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(gloggur_log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(gloggur_state_file)
    env["BOOTSTRAP_BD_LOG_FILE"] = str(bd_log_file)
    env["BOOTSTRAP_BD_STATE_FILE"] = str(bd_state_file)

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Beads readiness: ready" in completed.stdout
    assert bd_log_file.read_text(encoding="utf8").splitlines() == ["status --json"]


def test_bootstrap_fails_loudly_when_bd_is_missing_for_beads_enabled_repo(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    gloggur_log_file, gloggur_state_file = _prepare_fake_gloggur_wrapper(repo_root)
    gloggur_state_file.write_text("ready", encoding="utf8")
    _prepare_fake_beads_export(repo_root, issue_count=1)

    env = _base_bootstrap_env(tmp_path)
    env["PATH"] = "/usr/bin:/bin"
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(gloggur_log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(gloggur_state_file)

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 1
    assert "Beads readiness check failed: bd command not found." in completed.stderr


def test_bootstrap_fails_loudly_when_beads_import_fails(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    gloggur_log_file, gloggur_state_file = _prepare_fake_gloggur_wrapper(repo_root)
    gloggur_state_file.write_text("ready", encoding="utf8")
    _prepare_fake_beads_export(repo_root, issue_count=5)
    fake_bd_dir, bd_log_file, _ = _prepare_fake_bd(repo_root)

    env = _base_bootstrap_env(tmp_path)
    env["PATH"] = f"{fake_bd_dir}:{env['PATH']}"
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(gloggur_log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(gloggur_state_file)
    env["BOOTSTRAP_BD_LOG_FILE"] = str(bd_log_file)
    env["BOOTSTRAP_BD_IMPORT_FAIL"] = "1"
    env["BOOTSTRAP_BD_IMPORT_FAIL_MESSAGE"] = "import exploded"

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 1
    assert "Beads bootstrap failed while running: bd import -i .beads/issues.jsonl --json" in (
        completed.stderr
    )
    assert "import exploded" in completed.stderr
    assert bd_log_file.read_text(encoding="utf8").splitlines() == [
        "status --json",
        "init -p bd --json",
        "import -i .beads/issues.jsonl --json",
    ]


def test_bootstrap_fails_loudly_when_imported_beads_count_mismatches_jsonl(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    gloggur_log_file, gloggur_state_file = _prepare_fake_gloggur_wrapper(repo_root)
    gloggur_state_file.write_text("ready", encoding="utf8")
    _prepare_fake_beads_export(repo_root, issue_count=3)
    fake_bd_dir, bd_log_file, _ = _prepare_fake_bd(repo_root)

    env = _base_bootstrap_env(tmp_path)
    env["PATH"] = f"{fake_bd_dir}:{env['PATH']}"
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(gloggur_log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(gloggur_state_file)
    env["BOOTSTRAP_BD_LOG_FILE"] = str(bd_log_file)
    env["BOOTSTRAP_BD_COUNT_OVERRIDE"] = "99"

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 1
    assert "imported issue count does not match tracked .beads/issues.jsonl" in completed.stderr
    assert "Expected count: 3" in completed.stderr
    assert "Actual count: 99" in completed.stderr
    assert bd_log_file.read_text(encoding="utf8").splitlines() == [
        "status --json",
        "init -p bd --json",
        "import -i .beads/issues.jsonl --json",
        "status --json",
        "count --json",
    ]


def test_bootstrap_rewrites_stale_global_wrapper(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    home_dir = tmp_path / "home"
    wrapper_path = home_dir / ".local" / "bin" / "gloggur"
    _write_legacy_global_wrapper(wrapper_path)

    env = _base_bootstrap_env(tmp_path)
    env["HOME"] = str(home_dir)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["GLOGGUR_BOOTSTRAP_INSTALL_GLOBAL_WRAPPER"] = "1"
    env["GLOGGUR_BOOTSTRAP_GLOBAL_LINK_DIRS"] = ""

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    rewritten = wrapper_path.read_text(encoding="utf8")
    assert "run inside a gloggur git worktree" not in rewritten
    assert 'INSTALL_ROOT="${GLOGGUR_INSTALL_ROOT:-$DEFAULT_INSTALL_ROOT}"' in rewritten
    assert f'DEFAULT_INSTALL_ROOT="{repo_root}"' in rewritten
    assert "wrapper_launch_target_missing" in rewritten
    assert "wrapper_install_root_invalid" in rewritten
    assert "Global wrapper: installed:" in completed.stdout


def test_rewritten_global_wrapper_runs_from_external_cwd(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    home_dir = tmp_path / "home"
    wrapper_path = home_dir / ".local" / "bin" / "gloggur"
    _write_legacy_global_wrapper(wrapper_path)

    env = _base_bootstrap_env(tmp_path)
    env["HOME"] = str(home_dir)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["GLOGGUR_BOOTSTRAP_INSTALL_GLOBAL_WRAPPER"] = "1"
    env["GLOGGUR_BOOTSTRAP_GLOBAL_LINK_DIRS"] = ""

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )
    assert completed.returncode == 0, completed.stderr

    external_cwd = tmp_path / "outside"
    external_cwd.mkdir()
    status = subprocess.run(
        [str(wrapper_path), "status", "--json"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(external_cwd),
        check=False,
    )

    assert status.returncode == 0, status.stderr
    payload = json.loads(status.stdout)
    assert isinstance(payload.get("needs_reindex"), bool)
    assert "run inside a gloggur git worktree" not in status.stderr


def test_bootstrap_installs_wrapper_only_in_isolated_home(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    ambient_wrapper = Path(os.environ.get("HOME", "")) / ".local" / "bin" / "gloggur"
    ambient_before = ambient_wrapper.read_bytes() if ambient_wrapper.exists() else None

    env = _base_bootstrap_env(tmp_path)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["GLOGGUR_BOOTSTRAP_INSTALL_GLOBAL_WRAPPER"] = "1"
    env["GLOGGUR_BOOTSTRAP_GLOBAL_LINK_DIRS"] = ""

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    isolated_wrapper = Path(env["HOME"]) / ".local" / "bin" / "gloggur"
    assert isolated_wrapper.exists()
    if ambient_before is not None:
        assert ambient_wrapper.read_bytes() == ambient_before


def test_rewritten_global_wrapper_fails_deterministically_for_invalid_install_root(
    tmp_path: Path,
) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    home_dir = tmp_path / "home"
    wrapper_path = home_dir / ".local" / "bin" / "gloggur"
    _write_legacy_global_wrapper(wrapper_path)

    env = _base_bootstrap_env(tmp_path)
    env["HOME"] = str(home_dir)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["GLOGGUR_BOOTSTRAP_INSTALL_GLOBAL_WRAPPER"] = "1"
    env["GLOGGUR_BOOTSTRAP_GLOBAL_LINK_DIRS"] = ""

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )
    assert completed.returncode == 0, completed.stderr

    invalid_root = tmp_path / "invalid-root"
    invalid_launcher = invalid_root / "scripts" / "gloggur"
    invalid_launcher.parent.mkdir(parents=True, exist_ok=True)
    invalid_launcher.write_text("#!/usr/bin/env bash\necho invalid\n", encoding="utf8")
    invalid_launcher.chmod(0o755)

    run_env = dict(env)
    run_env["GLOGGUR_INSTALL_ROOT"] = str(invalid_root)
    status = subprocess.run(
        [str(wrapper_path), "status", "--json"],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=str(tmp_path),
        check=False,
    )
    payload = json.loads(status.stdout)

    assert status.returncode == 1
    assert payload["ok"] is False
    assert payload["stage"] == "dispatch"
    assert payload["error_code"] == "wrapper_install_root_invalid"
    assert status.stderr == ""


def test_bootstrap_allows_disabling_global_wrapper_refresh(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_venv_workspace(tmp_path)
    log_file, state_file = _prepare_fake_gloggur_wrapper(repo_root)
    home_dir = tmp_path / "home"
    wrapper_path = home_dir / ".local" / "bin" / "gloggur"
    _write_legacy_global_wrapper(wrapper_path)

    env = _base_bootstrap_env(tmp_path)
    env["HOME"] = str(home_dir)
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)
    env["GLOGGUR_BOOTSTRAP_INSTALL_GLOBAL_WRAPPER"] = "0"
    env["GLOGGUR_BOOTSTRAP_GLOBAL_LINK_DIRS"] = ""

    completed = _run_bootstrap(
        script_path=script_path,
        args=[
            "--skip-install",
            "--seed-venv-from",
            str(source_workspace),
            "--seed-venv-mode",
            "copy",
        ],
        env=env,
        cwd=repo_root,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Global wrapper: skipped:disabled" in completed.stdout
    assert "run inside a gloggur git worktree" in wrapper_path.read_text(encoding="utf8")
