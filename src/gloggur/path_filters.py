from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache


def is_minified_javascript(path: str) -> bool:
    """Return True when a path points to a `.min.js` file."""

    return path.lower().endswith(".min.js")


def _path_segments(path: str) -> set[str]:
    """Return normalized path segments for exact-name exclude checks."""

    return {segment for segment in os.path.normpath(path).split(os.sep) if segment}


def _has_activate_script(directory: str) -> bool:
    """Return whether a directory contains any activation helper."""

    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    return any(entry.startswith("activate") for entry in entries)


def _has_posix_site_packages(root: str) -> bool:
    """Return whether a POSIX-style env root exposes lib/python*/site-packages."""

    lib_dir = os.path.join(root, "lib")
    if not os.path.isdir(lib_dir):
        return False
    try:
        entries = os.listdir(lib_dir)
    except OSError:
        return False
    for entry in entries:
        if not entry.startswith("python"):
            continue
        if os.path.isdir(os.path.join(lib_dir, entry, "site-packages")):
            return True
    return False


@lru_cache(maxsize=4096)
def is_structural_python_virtualenv_root(path: str) -> bool:
    """Return whether a directory looks like a Python virtualenv, regardless of name."""

    normalized = os.path.normpath(os.path.abspath(path))
    if not os.path.isdir(normalized):
        return False
    if os.path.isfile(os.path.join(normalized, "pyvenv.cfg")):
        return True

    posix_bin = os.path.join(normalized, "bin")
    if os.path.isdir(posix_bin) and _has_posix_site_packages(normalized):
        if os.path.isfile(os.path.join(posix_bin, "python")) or os.path.isfile(
            os.path.join(posix_bin, "activate")
        ):
            return True

    scripts_dir = os.path.join(normalized, "Scripts")
    windows_site_packages = os.path.join(normalized, "Lib", "site-packages")
    if os.path.isdir(scripts_dir) and os.path.isdir(windows_site_packages):
        if os.path.isfile(os.path.join(scripts_dir, "python.exe")) or _has_activate_script(
            scripts_dir
        ):
            return True

    return False


def is_excluded_index_path(path: str, *, excluded_dirs: Iterable[str]) -> bool:
    """Return whether a file or directory is outside index scope."""

    for excluded in excluded_dirs:
        if excluded and excluded in _path_segments(path):
            return True

    normalized = os.path.normpath(os.path.abspath(path))
    candidate = normalized if os.path.isdir(normalized) else os.path.dirname(normalized)
    previous = ""
    while candidate and candidate != previous:
        if is_structural_python_virtualenv_root(candidate):
            return True
        previous = candidate
        candidate = os.path.dirname(candidate)
    return False


def filter_index_walk_dirs(
    current_root: str,
    dir_names: Iterable[str],
    *,
    excluded_dirs: Iterable[str],
) -> list[str]:
    """Return os.walk child directories that remain in index scope."""

    excluded = {name for name in excluded_dirs if name}
    kept: list[str] = []
    for name in dir_names:
        if name in excluded:
            continue
        if is_structural_python_virtualenv_root(os.path.join(current_root, name)):
            continue
        kept.append(name)
    return kept


def is_indexable_source_path(
    path: str,
    *,
    supported_extensions: Iterable[str],
    excluded_dirs: Iterable[str],
    include_minified_js: bool,
) -> bool:
    """Apply deterministic source-file eligibility checks for indexing flows."""

    normalized = os.path.normpath(path)
    if is_excluded_index_path(normalized, excluded_dirs=excluded_dirs):
        return False

    if not any(normalized.endswith(ext) for ext in supported_extensions):
        return False

    if not include_minified_js and is_minified_javascript(normalized):
        return False

    return True
