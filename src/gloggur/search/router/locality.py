from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from gloggur.byte_spans import to_repo_relative_path
from gloggur.search.router.backends import _load_snippet
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.types import BackendHit

_TEST_DIR_NAMES = ("test", "tests")
_GENERIC_PATH_PARTS = {
    "app",
    "apps",
    "core",
    "helpers",
    "lib",
    "main",
    "models",
    "src",
    "utils",
    "views",
}


@dataclass(frozen=True)
class LocalityAnchor:
    """One local anchor span reused across routed searches."""

    path: str
    start_line: int
    end_line: int
    score: float
    backend: str | None = None
    tags: tuple[str, ...] = ()

    def to_payload(self, *, repo_root: Path) -> dict[str, object]:
        return {
            "path": to_repo_relative_path(repo_root, self.path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.score, 4),
            "backend": self.backend,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class LocalityState:
    """Persisted locality state between CLI invocations."""

    anchor: LocalityAnchor
    updated_at_epoch: float
    ambiguity_streak: int = 0

    def to_payload(self, *, repo_root: Path) -> dict[str, object]:
        return {
            "anchor": self.anchor.to_payload(repo_root=repo_root),
            "updated_at_epoch": round(self.updated_at_epoch, 6),
            "ambiguity_streak": max(0, int(self.ambiguity_streak)),
        }


def _state_file_path(repo_root: Path, config: SearchRouterConfig) -> Path:
    return repo_root / config.state_path


def _normalize_anchor_path(path: str, *, repo_root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((repo_root / candidate).resolve())


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
    os.replace(temp_path, path)


def load_locality_state(
    *,
    repo_root: Path,
    config: SearchRouterConfig,
    now: float | None = None,
) -> LocalityState | None:
    """Load recent locality state from disk when it is still fresh."""

    path = _state_file_path(repo_root, config)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    anchor_payload = payload.get("anchor")
    if not isinstance(anchor_payload, dict):
        return None
    raw_path = anchor_payload.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        start_line = int(anchor_payload.get("start_line", 1))
        end_line = int(anchor_payload.get("end_line", start_line))
        score = float(anchor_payload.get("score", 0.0))
        updated_at_epoch = float(payload.get("updated_at_epoch"))
        ambiguity_streak = int(payload.get("ambiguity_streak", 0))
    except (TypeError, ValueError):
        return None
    current = time.time() if now is None else now
    if current - updated_at_epoch > max(1, config.locality_state_ttl_sec):
        return None
    backend = anchor_payload.get("backend")
    backend_name = backend if isinstance(backend, str) and backend else None
    raw_tags = anchor_payload.get("tags")
    tags = (
        tuple(str(tag) for tag in raw_tags if isinstance(tag, str) and tag)
        if isinstance(raw_tags, list)
        else ()
    )
    return LocalityState(
        anchor=LocalityAnchor(
            path=_normalize_anchor_path(raw_path, repo_root=repo_root),
            start_line=max(1, start_line),
            end_line=max(max(1, start_line), end_line),
            score=max(0.0, min(1.0, score)),
            backend=backend_name,
            tags=tags,
        ),
        updated_at_epoch=updated_at_epoch,
        ambiguity_streak=max(0, ambiguity_streak),
    )


def persist_locality_state(
    *,
    repo_root: Path,
    config: SearchRouterConfig,
    state: LocalityState,
) -> None:
    """Persist locality state best-effort for later router reuse."""

    try:
        _atomic_write_json(
            _state_file_path(repo_root, config),
            state.to_payload(repo_root=repo_root),
        )
    except OSError:
        return


def build_anchor_hit(
    *,
    anchor: LocalityAnchor,
    repo_root: Path,
    config: SearchRouterConfig,
    score: float = 1.1,
) -> BackendHit:
    """Build a synthetic backend hit that re-opens the active anchor span."""

    snippet = _load_snippet(
        repo_root,
        anchor.path,
        start_line=anchor.start_line,
        end_line=anchor.end_line,
        radius=2,
        max_chars=config.max_snippet_chars,
    )
    tags = tuple(sorted(set(anchor.tags) | {"locality_anchor"}))
    return BackendHit(
        backend=anchor.backend or "locality",
        path=anchor.path,
        start_line=anchor.start_line,
        end_line=anchor.end_line,
        snippet=snippet,
        raw_score=max(score, anchor.score),
        tags=tags,
    )


def is_test_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    basename = Path(lowered).name
    return (
        "/test/" in lowered
        or "/tests/" in lowered
        or basename.startswith("test_")
        or basename.endswith("_test.py")
        or ".spec." in basename
    )


def is_source_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    if is_test_path(lowered):
        return False
    return "/src/" in lowered or lowered.endswith(
        (".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java")
    )


def derive_candidate_test_paths(
    *,
    repo_root: Path,
    anchor_path: str,
    observed_test_paths: tuple[str, ...] = (),
    query_tokens: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return deterministic candidate test paths related to an anchor path."""

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        normalized = candidate.replace("\\", "/")
        if normalized in seen:
            return
        seen.add(normalized)
        ordered.append(normalized)

    for candidate in observed_test_paths:
        if is_test_path(candidate):
            _add(candidate)

    test_roots = [repo_root / name for name in _TEST_DIR_NAMES if (repo_root / name).exists()]
    if not test_roots:
        return tuple(ordered)

    anchor = Path(anchor_path)
    stem_terms = {anchor.stem.lower()}
    stem_terms.update(
        part.lower()
        for part in anchor.parts
        if len(part) >= 4 and part.lower() not in _GENERIC_PATH_PARTS and part.isidentifier()
    )
    stem_terms.update(
        token.lower() for token in query_tokens if len(token) >= 4 and token.isidentifier()
    )
    stem_terms.discard("test")
    stem_terms.discard("tests")

    scored: list[tuple[int, str]] = []
    for root in test_roots:
        for path in sorted(root.rglob("*.py")):
            path_text = str(path.resolve()).replace("\\", "/")
            if not is_test_path(path_text):
                continue
            name = path.name.lower()
            rel_parts = {part.lower() for part in path.relative_to(repo_root).parts}
            score = 0
            for term in stem_terms:
                if name == f"test_{term}.py" or name == f"{term}_test.py":
                    score += 4
                elif term in name:
                    score += 2
                if term in rel_parts:
                    score += 1
            if score <= 0:
                continue
            scored.append((score, path_text))

    for _score, candidate in sorted(scored, key=lambda item: (-item[0], item[1]))[:6]:
        _add(candidate)
    return tuple(ordered)
