from __future__ import annotations

from pathlib import Path

QUERY_DOMAINS = {"code", "docs_policy", "workflow_config"}
_DOC_NAME_MARKERS = ("readme", "changelog", "changes", "history", "news", "release-notes")
_WORKFLOW_CONFIG_NAMES = {
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
    "noxfile.py",
    "conftest.py",
}
_SOURCE_CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".m",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".swift",
    ".ts",
    ".tsx",
}


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def is_source_code_path(path: str) -> bool:
    lowered = normalize_path(path).lower()
    normalized = Path(lowered)
    segments = {segment for segment in lowered.split("/") if segment}
    return normalized.suffix in _SOURCE_CODE_EXTENSIONS or "src" in segments


def is_test_path(path: str) -> bool:
    lowered = normalize_path(path).lower()
    basename = Path(lowered).name
    segments = {segment for segment in lowered.split("/") if segment}
    return (
        "tests" in segments
        or "test" in segments
        or basename.startswith("test_")
        or basename.endswith("_test.py")
        or basename.endswith("_test.go")
        or ".spec." in basename
    )


def is_docs_path(path: str) -> bool:
    lowered = normalize_path(path).lower()
    basename = Path(lowered).name
    stem = Path(lowered).stem
    if "/docs/" in lowered or lowered.startswith("docs/") or "/doc/" in lowered:
        return True
    if basename.endswith((".md", ".rst", ".txt")) and stem in _DOC_NAME_MARKERS:
        return True
    return any(marker in stem for marker in _DOC_NAME_MARKERS)


def is_docs_authority_path(path: str) -> bool:
    lowered = normalize_path(path).lower()
    if lowered.startswith("docs/internals/") or "/docs/internals/" in lowered:
        return True
    return is_docs_path(path)


def is_workflow_config_path(path: str) -> bool:
    lowered = normalize_path(path).lower()
    basename = Path(lowered).name
    if lowered.startswith(".github/workflows/") or "/.github/workflows/" in lowered:
        return True
    if basename in _WORKFLOW_CONFIG_NAMES:
        return True
    return False


def is_authority_path(path: str, *, query_domain: str) -> bool:
    if query_domain == "docs_policy":
        return is_docs_authority_path(path) or is_workflow_config_path(path)
    if query_domain == "workflow_config":
        return is_workflow_config_path(path) or is_docs_authority_path(path)
    return is_source_code_path(path) or is_test_path(path)


def path_domain_score(path: str, *, query_domain: str) -> float:
    if query_domain == "docs_policy":
        if is_docs_authority_path(path):
            return 1.0
        if is_workflow_config_path(path):
            return 0.9
        if is_source_code_path(path):
            return 0.54
        if is_test_path(path):
            return 0.42
        return 0.62
    if query_domain == "workflow_config":
        if is_workflow_config_path(path):
            return 1.0
        if is_docs_authority_path(path):
            return 0.95
        if is_source_code_path(path):
            return 0.56
        if is_test_path(path):
            return 0.44
        return 0.64
    if is_source_code_path(path):
        return 1.0
    if is_test_path(path):
        return 0.82
    if is_docs_authority_path(path):
        return 0.28
    if is_workflow_config_path(path):
        return 0.22
    return 0.66


def should_treat_as_auxiliary(path: str, *, query_domain: str) -> bool:
    if query_domain != "code":
        return False
    return is_docs_path(path) or is_workflow_config_path(path)
