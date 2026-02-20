"""Local development package bootstrap.

This repository uses a ``src/`` layout. When running commands directly from the
repo root (for example ``python -m gloggur.cli.main``), Python may first resolve
this top-level package path. Extend ``__path__`` so submodules are loaded from
``src/gloggur``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["__version__"]
__version__ = "0.1.0"

_repo_root = Path(__file__).resolve().parent.parent
_src_pkg = _repo_root / "src" / "gloggur"

if _src_pkg.exists():
    _src_pkg_str = str(_src_pkg)
    if _src_pkg_str not in __path__:
        __path__.append(_src_pkg_str)
