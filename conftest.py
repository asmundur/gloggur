"""Pytest configuration for local src-layout imports.

Ensures tests import the package from ``src/`` even if a stale top-level
``gloggur/`` namespace directory exists from old bytecode artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prepend_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    _prepend_path(SRC_ROOT)
