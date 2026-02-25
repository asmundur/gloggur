from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_script() -> Path:
    return _repo_root() / "scripts" / "bootstrap_gloggur_env.sh"


def _prepare_temp_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / "bootstrap_gloggur_env.sh"
    script_path.write_text(_source_script().read_text(encoding="utf8"), encoding="utf8")
    script_path.chmod(0o755)
    return repo_root, script_path


def _run_bootstrap(
    script_path: Path,
    args: list[str],
    env: dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(script_path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
    venv_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf8")
    venv_python.chmod(0o755)
    return workspace_root


def _prepare_fake_gloggur_wrapper(repo_root: Path) -> tuple[Path, Path]:
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    log_file = repo_root / "gloggur-invocations.log"
    state_file = repo_root / "index-ready.flag"
    wrapper = scripts_dir / "gloggur"
    wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

log_file="${BOOTSTRAP_GLOGGUR_LOG_FILE:?}"
state_file="${BOOTSTRAP_GLOGGUR_STATE_FILE:?}"
printf '%s\\n' "$*" >> "$log_file"

command_name="${1:-}"
shift || true

case "$command_name" in
  status)
    if [[ -f "$state_file" ]]; then
      printf '{"needs_reindex": false}\\n'
    else
      printf '{"needs_reindex": true}\\n'
    fi
    ;;
  index)
    touch "$state_file"
    printf '{"indexed_files": 1, "indexed_symbols": 1}\\n'
    ;;
  *)
    printf '{"error": "unexpected command"}\\n'
    exit 1
    ;;
esac
""",
        encoding="utf8",
    )
    wrapper.chmod(0o755)
    return log_file, state_file


def test_bootstrap_can_seed_cache_via_symlink(tmp_path: Path) -> None:
    repo_root, script_path = _prepare_temp_repo(tmp_path)
    source_workspace = _prepare_seed_source(tmp_path)
    source_cache = source_workspace / ".gloggur-cache"
    env = os.environ.copy()

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
    env = os.environ.copy()

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
    env = os.environ.copy()

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
    env = os.environ.copy()

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
    env = os.environ.copy()

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
    env = os.environ.copy()
    env["BOOTSTRAP_GLOGGUR_LOG_FILE"] = str(log_file)
    env["BOOTSTRAP_GLOGGUR_STATE_FILE"] = str(state_file)

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
    ]
