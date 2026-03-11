from __future__ import annotations

from pathlib import Path

from gloggur.path_filters import (
    filter_index_walk_dirs,
    is_excluded_index_path,
    is_structural_python_virtualenv_root,
)


def test_structural_virtualenv_root_detects_pyvenv_cfg_without_name_assumptions(
    tmp_path: Path,
) -> None:
    env_root = tmp_path / "custom-python-runtime"
    env_root.mkdir()
    (env_root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf8")
    source = env_root / "lib" / "python3.13" / "site-packages" / "vendor.py"
    source.parent.mkdir(parents=True)
    source.write_text("def vendor() -> int:\n    return 1\n", encoding="utf8")

    assert is_structural_python_virtualenv_root(str(env_root)) is True
    assert is_excluded_index_path(str(source), excluded_dirs=[]) is True


def test_structural_virtualenv_root_detects_windows_layout_without_name_assumptions(
    tmp_path: Path,
) -> None:
    env_root = tmp_path / "_venv"
    scripts_dir = env_root / "Scripts"
    site_packages = env_root / "Lib" / "site-packages"
    scripts_dir.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    (scripts_dir / "python.exe").write_text("", encoding="utf8")
    source = site_packages / "vendor.py"
    source.write_text("def vendor() -> int:\n    return 1\n", encoding="utf8")

    assert is_structural_python_virtualenv_root(str(env_root)) is True
    assert is_excluded_index_path(str(source), excluded_dirs=[]) is True


def test_structural_virtualenv_root_allows_partial_lookalikes(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "bin-tools"
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin" / "python").write_text("", encoding="utf8")
    source = candidate / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("def module() -> int:\n    return 1\n", encoding="utf8")

    assert is_structural_python_virtualenv_root(str(candidate)) is False
    assert is_excluded_index_path(str(source), excluded_dirs=[]) is False


def test_filter_index_walk_dirs_excludes_structural_virtualenv_children(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env_root = repo / "python-runtime"
    (env_root / "bin").mkdir(parents=True)
    (env_root / "bin" / "python").write_text("", encoding="utf8")
    (env_root / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    (repo / "src").mkdir()

    kept = filter_index_walk_dirs(
        str(repo),
        ["python-runtime", "src"],
        excluded_dirs=[".git"],
    )

    assert kept == ["src"]
