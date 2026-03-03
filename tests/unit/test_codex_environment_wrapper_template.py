from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _wrapper_block() -> str:
    template = (_repo_root() / ".codex" / "environments" / "environment.toml").read_text(
        encoding="utf8"
    )
    start_marker = "cat > \"$BIN/gloggur\" <<'SH'"
    end_marker = "\nSH\n"
    start = template.index(start_marker) + len(start_marker)
    end = template.index(end_marker, start)
    return template[start:end]


def _bootstrap_wrapper_block() -> str:
    script = (_repo_root() / "scripts" / "bootstrap_gloggur_env.sh").read_text(encoding="utf8")
    start_marker = "cat > \"$wrapper_path\" <<'SH'"
    end_marker = "\nSH\n"
    start = script.index(start_marker) + len(start_marker)
    end = script.index(end_marker, start)
    return script[start:end]


def test_wrapper_template_avoids_invocation_worktree_gate() -> None:
    wrapper = _wrapper_block()
    assert "run inside a gloggur git worktree" not in wrapper
    assert "git rev-parse --show-toplevel 2>/dev/null || true" not in wrapper
    assert 'DEFAULT_INSTALL_ROOT="__GLOGGUR_INSTALL_ROOT__"' in wrapper
    assert 'INSTALL_ROOT="${GLOGGUR_INSTALL_ROOT:-$DEFAULT_INSTALL_ROOT}"' in wrapper


def test_wrapper_template_emits_structured_json_error_for_missing_launcher() -> None:
    wrapper = _wrapper_block()
    assert "if [[ $is_json -eq 1 ]]; then" in wrapper
    assert '"ok":false' in wrapper
    assert '"stage":"dispatch"' in wrapper
    assert '"compatibility":{"operation":"wrapper","error":true' in wrapper
    assert "wrapper_launch_target_missing" in wrapper


def test_wrapper_template_non_json_error_is_stderr_only() -> None:
    wrapper = _wrapper_block()
    assert 'echo "gloggur wrapper failed (${error_code}): ${message}" >&2' in wrapper
    assert (
        'echo "Set GLOGGUR_INSTALL_ROOT to a gloggur checkout and rerun scripts/bootstrap_gloggur_env.sh." >&2'
        in wrapper
    )
    assert "export GLOGGUR_RUN_FROM_CALLER_CWD=1" in wrapper
    assert "wrapper_install_root_invalid" in wrapper


def test_bootstrap_wrapper_contract_matches_codex_template_invariants() -> None:
    codex_wrapper = _wrapper_block()
    bootstrap_wrapper = _bootstrap_wrapper_block()

    required_fragments = [
        '"stage":"dispatch"',
        '"compatibility":{"operation":"wrapper","error":true',
        'DEFAULT_INSTALL_ROOT="__GLOGGUR_INSTALL_ROOT__"',
        'INSTALL_ROOT="${GLOGGUR_INSTALL_ROOT:-$DEFAULT_INSTALL_ROOT}"',
        "export GLOGGUR_RUN_FROM_CALLER_CWD=1",
        "wrapper_install_root_invalid",
    ]
    for fragment in required_fragments:
        assert fragment in codex_wrapper
        assert fragment in bootstrap_wrapper

    assert "run inside a gloggur git worktree" not in bootstrap_wrapper
    assert "git rev-parse --show-toplevel 2>/dev/null || true" not in bootstrap_wrapper
