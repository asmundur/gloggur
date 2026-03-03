from __future__ import annotations

import re


def _strip_python_docstring(code_text: str) -> str:
    """Remove the first Python docstring from a code snippet."""
    lines = code_text.splitlines()
    if len(lines) <= 1:
        return code_text
    header = lines[0]
    body = "\n".join(lines[1:])
    body = re.sub(r"^\s*(?P<quote>\"\"\"|''')(?:.|\n)*?(?P=quote)\s*", "", body, count=1)
    body = re.sub(r"^\s*(?P<quote>\"|')(?:.|\n)*?(?P=quote)\s*", "", body, count=1)
    combined = "\n".join([header, body]).strip()
    return combined


_LANGUAGE_CLEANERS = {
    "python": _strip_python_docstring,
}


def prepare_code_text(
    code_text: str,
    *,
    language: str | None,
    max_chars: int,
) -> str:
    """Prepare language-aware code text for semantic similarity scoring."""
    cleaned = code_text
    cleaner = _LANGUAGE_CLEANERS.get(language or "")
    if cleaner is not None:
        cleaned = cleaner(cleaned)
    cleaned = cleaned.strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned
