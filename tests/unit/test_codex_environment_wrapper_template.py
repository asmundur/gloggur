from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _wrapper_block() -> str:
    template = (_repo_root() / ".codex" / "environments" / "environment.toml").read_text(encoding="utf8")
    start_marker = 'cat > "$BIN/gloggur" <<\'SH\''
    end_marker = "\nSH\n"
    start = template.index(start_marker) + len(start_marker)
    end = template.index(end_marker, start)
    return template[start:end]


def test_wrapper_template_avoids_invocation_worktree_gate() -> None:
    wrapper = _wrapper_block()
    assert "run inside a gloggur git worktree" not in wrapper
    assert "git rev-parse --show-toplevel 2>/dev/null || true" not in wrapper
    assert 'DEFAULT_INSTALL_ROOT="__GLOGGUR_INSTALL_ROOT__"' in wrapper
    assert 'INSTALL_ROOT="${GLOGGUR_INSTALL_ROOT:-$DEFAULT_INSTALL_ROOT}"' in wrapper


def test_wrapper_template_emits_structured_json_error_for_missing_launcher() -> None:
    wrapper = _wrapper_block()
    assert 'if [[ $is_json -eq 1 ]]; then' in wrapper
    assert '"error": true' in wrapper
    assert '"error_code": "wrapper_launch_target_missing"' in wrapper
    assert '"detected_environment"' in wrapper


def test_wrapper_template_non_json_error_is_stderr_only() -> None:
    wrapper = _wrapper_block()
    assert (
        'echo "gloggur wrapper failed (wrapper_launch_target_missing): launcher not found at ${launcher_path}" >&2'
        in wrapper
    )
    assert (
        'echo "Set GLOGGUR_INSTALL_ROOT to a gloggur checkout and rerun scripts/bootstrap_gloggur_env.sh." >&2'
        in wrapper
    )
    assert "export GLOGGUR_RUN_FROM_CALLER_CWD=1" in wrapper
