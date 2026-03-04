from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_HEADINGS = [
    "## CLI Contract Errors",
    "## Index Failure Codes",
    "## Inspect Failure Codes",
    "## Watch Status Failure Codes",
    "## Resume Reason Codes",
]


def _prepend_source_root() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    source_root_str = str(source_root)
    if source_root_str in sys.path:
        sys.path.remove(source_root_str)
    sys.path.insert(0, source_root_str)


def _section_codes() -> dict[str, list[str]]:
    _prepend_source_root()
    from gloggur.cli.main import (
        CLI_FAILURE_REMEDIATION,
        INSPECT_FAILURE_REMEDIATION,
        RESUME_REMEDIATION,
        WATCH_STATUS_FAILURE_REMEDIATION,
    )
    from gloggur.indexer.indexer import FAILURE_REMEDIATION as INDEX_FAILURE_REMEDIATION

    return {
        "cli": sorted(CLI_FAILURE_REMEDIATION),
        "index": sorted(INDEX_FAILURE_REMEDIATION),
        "inspect": sorted(INSPECT_FAILURE_REMEDIATION),
        "watch_status": sorted(WATCH_STATUS_FAILURE_REMEDIATION),
        "resume": sorted(RESUME_REMEDIATION),
    }


def _missing_snippets(text: str, snippets: list[str]) -> list[str]:
    return [snippet for snippet in snippets if snippet not in text]


def check_error_catalog_contract(docs_path: Path) -> dict[str, object]:
    if not docs_path.exists():
        return {
            "ok": False,
            "failure": {
                "code": "error_catalog_docs_missing",
                "detail": f"error catalog docs file does not exist: {docs_path}",
                "remediation": "Create docs/ERROR_CODES.md and rerun this contract check.",
            },
        }

    docs_text = docs_path.read_text(encoding="utf8")
    missing_headings = _missing_snippets(docs_text, REQUIRED_HEADINGS)
    codes_by_section = _section_codes()
    missing_codes_by_section = {
        section: [code for code in codes if f"`{code}`" not in docs_text]
        for section, codes in codes_by_section.items()
    }
    all_missing_codes = {
        section: missing for section, missing in missing_codes_by_section.items() if missing
    }

    ok = not missing_headings and not all_missing_codes
    payload: dict[str, object] = {
        "ok": ok,
        "summary": {
            "docs_path": str(docs_path),
            "required_headings": len(REQUIRED_HEADINGS),
            "section_code_counts": {
                section: len(codes) for section, codes in codes_by_section.items()
            },
        },
        "missing_headings": missing_headings,
        "missing_codes_by_section": all_missing_codes,
    }
    if not ok:
        payload["failure"] = {
            "code": "error_catalog_contract_violation",
            "detail": "Error-code catalog is missing required headings or live source codes.",
            "remediation": (
                "Update docs/ERROR_CODES.md so every live error code is documented, "
                "then rerun the contract check."
            ),
        }
    return payload


def _render_markdown(payload: dict[str, object]) -> str:
    lines: list[str] = ["# Error Catalog Contract", ""]
    lines.append(f"- ok: `{payload.get('ok')}`")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        lines.append(f"- docs_path: `{summary.get('docs_path')}`")
        lines.append(f"- section_code_counts: `{summary.get('section_code_counts')}`")
    missing_headings = payload.get("missing_headings")
    if missing_headings:
        lines.append(f"- missing_headings: `{missing_headings}`")
    missing_codes = payload.get("missing_codes_by_section")
    if missing_codes:
        lines.append(f"- missing_codes_by_section: `{missing_codes}`")
    failure = payload.get("failure")
    if isinstance(failure, dict):
        lines.append(f"- failure.code: `{failure.get('code')}`")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate published error-code catalog contract.")
    parser.add_argument("--docs-path", type=Path, default=Path("docs/ERROR_CODES.md"))
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = check_error_catalog_contract(args.docs_path)
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload), end="")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
