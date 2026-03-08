from __future__ import annotations

import os
from collections.abc import Iterable


def is_minified_javascript(path: str) -> bool:
    """Return True when a path points to a `.min.js` file."""

    return path.lower().endswith(".min.js")


def is_indexable_source_path(
    path: str,
    *,
    supported_extensions: Iterable[str],
    excluded_dirs: Iterable[str],
    include_minified_js: bool,
) -> bool:
    """Apply deterministic source-file eligibility checks for indexing flows."""

    normalized = os.path.normpath(path)
    segments = {segment for segment in normalized.split(os.sep) if segment}

    for excluded in excluded_dirs:
        if excluded and excluded in segments:
            return False

    if not any(normalized.endswith(ext) for ext in supported_extensions):
        return False

    if not include_minified_js and is_minified_javascript(normalized):
        return False

    return True
