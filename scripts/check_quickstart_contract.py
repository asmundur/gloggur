from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REQUIRED_HEADINGS = [
    "## Install and Bootstrap",
    "## Provider Setup",
    "### OpenAI",
    "### Gemini",
    "## First Run (Index, Watch, Search, Inspect)",
    "## Troubleshooting by Failure Code",
]

REQUIRED_COMMAND_SNIPPETS = [
    "scripts/bootstrap_gloggur_env.sh",
    "scripts/gloggur status --json",
    "scripts/gloggur index . --json",
    "scripts/gloggur watch init . --json",
    "scripts/gloggur watch start --daemon --json",
    "scripts/gloggur watch status --json",
    "scripts/gloggur search \"add numbers token\" --top-k 5 --json",
    "scripts/gloggur inspect . --json",
    "scripts/gloggur watch stop --json",
]

REQUIRED_PROVIDER_SNIPPETS = [
    "GLOGGUR_EMBEDDING_PROVIDER=openai",
    "OPENAI_API_KEY",
    "GLOGGUR_EMBEDDING_PROVIDER=gemini",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
]

REQUIRED_FAILURE_CODES = [
    "embedding_provider_error",
    "watch_mode_conflict",
    "watch_path_missing",
    "search_grounding_validation_failed",
]

DEFAULT_SOURCE_FILES = [
    Path("src/gloggur/cli/main.py"),
    Path("src/gloggur/embeddings/errors.py"),
]


def _missing_snippets(text: str, snippets: Iterable[str]) -> List[str]:
    return [snippet for snippet in snippets if snippet not in text]


def check_quickstart_contract(
    docs_path: Path,
    *,
    source_paths: Optional[List[Path]] = None,
) -> Dict[str, object]:
    if not docs_path.exists():
        return {
            "ok": False,
            "failure": {
                "code": "quickstart_contract_docs_missing",
                "detail": f"quickstart docs file does not exist: {docs_path}",
                "remediation": "Create docs/QUICKSTART.md and rerun this contract check.",
            },
        }

    docs_text = docs_path.read_text(encoding="utf8")
    source_files = list(DEFAULT_SOURCE_FILES if source_paths is None else source_paths)

    missing_headings = _missing_snippets(docs_text, REQUIRED_HEADINGS)
    missing_commands = _missing_snippets(docs_text, REQUIRED_COMMAND_SNIPPETS)
    missing_provider_snippets = _missing_snippets(docs_text, REQUIRED_PROVIDER_SNIPPETS)
    missing_doc_codes = _missing_snippets(docs_text, REQUIRED_FAILURE_CODES)

    source_text_fragments: List[str] = []
    missing_source_files: List[str] = []
    for source_path in source_files:
        if not source_path.exists():
            missing_source_files.append(str(source_path))
            continue
        source_text_fragments.append(source_path.read_text(encoding="utf8"))
    source_union = "\n".join(source_text_fragments)
    missing_cli_codes = _missing_snippets(source_union, REQUIRED_FAILURE_CODES)

    ok = (
        not missing_headings
        and not missing_commands
        and not missing_provider_snippets
        and not missing_doc_codes
        and not missing_source_files
        and not missing_cli_codes
    )

    payload: Dict[str, object] = {
        "ok": ok,
        "summary": {
            "docs_path": str(docs_path),
            "source_files": [str(path) for path in source_files],
            "required_headings": len(REQUIRED_HEADINGS),
            "required_commands": len(REQUIRED_COMMAND_SNIPPETS),
            "required_provider_snippets": len(REQUIRED_PROVIDER_SNIPPETS),
            "required_failure_codes": len(REQUIRED_FAILURE_CODES),
        },
        "missing_headings": missing_headings,
        "missing_commands": missing_commands,
        "missing_provider_snippets": missing_provider_snippets,
        "missing_failure_codes_in_docs": missing_doc_codes,
        "missing_source_files": missing_source_files,
        "missing_failure_codes_in_source": missing_cli_codes,
    }

    if not ok:
        payload["failure"] = {
            "code": "quickstart_contract_violation",
            "detail": "Quickstart docs/source contract is missing required headings, commands, provider snippets, or failure codes.",
            "remediation": "Update docs/QUICKSTART.md and source failure-code references, then rerun the contract check.",
        }
    return payload


def _render_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = ["# Quickstart Contract", ""]
    lines.append(f"- ok: `{payload.get('ok')}`")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        lines.append(f"- docs_path: `{summary.get('docs_path')}`")
        lines.append(f"- source_files: `{summary.get('source_files')}`")
    for field in (
        "missing_headings",
        "missing_commands",
        "missing_provider_snippets",
        "missing_failure_codes_in_docs",
        "missing_source_files",
        "missing_failure_codes_in_source",
    ):
        value = payload.get(field)
        if value:
            lines.append(f"- {field}: `{value}`")
    failure = payload.get("failure")
    if isinstance(failure, dict):
        lines.append(f"- failure.code: `{failure.get('code')}`")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate quickstart docs contract.")
    parser.add_argument("--docs-path", type=Path, default=Path("docs/QUICKSTART.md"))
    parser.add_argument(
        "--source-path",
        action="append",
        dest="source_paths",
        default=None,
        help="Additional source file path to include (repeatable).",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    source_paths = None
    if args.source_paths:
        source_paths = [Path(path) for path in args.source_paths]
    payload = check_quickstart_contract(args.docs_path, source_paths=source_paths)

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload), end="")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
