from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from gloggur.cli.main import (
    CLI_FAILURE_REMEDIATION,
    INSPECT_FAILURE_REMEDIATION,
    RESUME_REMEDIATION,
    WATCH_STATUS_FAILURE_REMEDIATION,
)
from gloggur.indexer.indexer import FAILURE_REMEDIATION
from gloggur.watch.service import WATCH_FAILURE_REMEDIATION

REQUIRED_HEADINGS = [
    "# Error Code Catalog",
    "## CLI Contract Errors",
    "## Embedding Provider Failures",
    "## Index Failure Codes",
    "## Inspect Failure Codes",
    "## Watch Status Failure Codes",
    "## Resume Reason Codes",
]


def default_surface_code_map() -> Dict[str, List[str]]:
    return {
        "cli_preflight": sorted(CLI_FAILURE_REMEDIATION),
        "embedding_provider": ["embedding_provider_error"],
        "index_watch_incremental": sorted(set(FAILURE_REMEDIATION) | set(WATCH_FAILURE_REMEDIATION)),
        "inspect_file": sorted(INSPECT_FAILURE_REMEDIATION),
        "watch_status": sorted(WATCH_STATUS_FAILURE_REMEDIATION),
        "resume": sorted(RESUME_REMEDIATION),
    }


def _missing_snippets(text: str, snippets: Iterable[str]) -> List[str]:
    return [snippet for snippet in snippets if snippet not in text]


def check_error_code_catalog(
    docs_path: Path,
    *,
    surface_code_map: Optional[Mapping[str, List[str]]] = None,
) -> Dict[str, object]:
    if not docs_path.exists():
        return {
            "ok": False,
            "failure": {
                "code": "error_code_catalog_missing",
                "detail": f"error code catalog file does not exist: {docs_path}",
                "remediation": "Create docs/ERROR_CODES.md and rerun the contract check.",
            },
        }

    docs_text = docs_path.read_text(encoding="utf8")
    code_map = dict(default_surface_code_map() if surface_code_map is None else surface_code_map)

    missing_headings = _missing_snippets(docs_text, REQUIRED_HEADINGS)
    missing_codes_by_surface: Dict[str, List[str]] = {}
    documented_codes: set[str] = set()

    for surface, codes in code_map.items():
        missing_codes = []
        for code in codes:
            if f"`{code}`" not in docs_text:
                missing_codes.append(code)
                continue
            documented_codes.add(code)
        if missing_codes:
            missing_codes_by_surface[surface] = missing_codes

    all_expected_codes = sorted({code for codes in code_map.values() for code in codes})
    ok = not missing_headings and not missing_codes_by_surface

    payload: Dict[str, object] = {
        "ok": ok,
        "summary": {
            "docs_path": str(docs_path),
            "required_headings": len(REQUIRED_HEADINGS),
            "surfaces": sorted(code_map),
            "expected_codes_total": len(all_expected_codes),
        },
        "missing_headings": missing_headings,
        "missing_codes_by_surface": missing_codes_by_surface,
        "expected_codes": all_expected_codes,
    }

    if not ok:
        payload["failure"] = {
            "code": "error_code_catalog_violation",
            "detail": "Error code catalog is missing required headings or documented error codes.",
            "remediation": "Update docs/ERROR_CODES.md so every machine-readable code is documented, then rerun the contract check.",
        }
    return payload


def _render_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = ["# Error Code Catalog Contract", ""]
    lines.append(f"- ok: `{payload.get('ok')}`")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        lines.append(f"- docs_path: `{summary.get('docs_path')}`")
        lines.append(f"- expected_codes_total: `{summary.get('expected_codes_total')}`")
        lines.append(f"- surfaces: `{summary.get('surfaces')}`")
    if payload.get("missing_headings"):
        lines.append(f"- missing_headings: `{payload.get('missing_headings')}`")
    if payload.get("missing_codes_by_surface"):
        lines.append(f"- missing_codes_by_surface: `{payload.get('missing_codes_by_surface')}`")
    failure = payload.get("failure")
    if isinstance(failure, dict):
        lines.append(f"- failure.code: `{failure.get('code')}`")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate error code catalog coverage.")
    parser.add_argument("--docs-path", type=Path, default=Path("docs/ERROR_CODES.md"))
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = check_error_code_catalog(args.docs_path)
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload), end="")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
