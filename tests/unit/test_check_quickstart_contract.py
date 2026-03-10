from __future__ import annotations

from pathlib import Path

from scripts.check_quickstart_contract import check_quickstart_contract


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")


def test_quickstart_contract_passes_when_required_sections_and_codes_exist(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs" / "QUICKSTART.md"
    cli_path = tmp_path / "src" / "gloggur" / "cli" / "main.py"
    embedding_errors_path = tmp_path / "src" / "gloggur" / "embeddings" / "errors.py"

    _write(
        docs_path,
        "\n".join(
            [
                "## Install and Bootstrap",
                "scripts/bootstrap_gloggur_env.sh",
                "scripts/gloggur status --json",
                "scripts/gloggur index . --json",
                "scripts/gloggur init . --betatester-support --json",
                "## Provider Setup",
                "### OpenAI",
                "GLOGGUR_EMBEDDING_PROVIDER=openai",
                "OPENROUTER_API_KEY=example",
                "GLOGGUR_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1",
                "GLOGGUR_OPENROUTER_SITE_URL=https://example.com",
                "GLOGGUR_OPENROUTER_APP_NAME=gloggur",
                "OPENAI_API_KEY=example",
                "### Gemini",
                "GLOGGUR_EMBEDDING_PROVIDER=gemini",
                "GEMINI_API_KEY=example",
                "GOOGLE_API_KEY=example",
                "## First Run (Index, Watch, Search, Inspect)",
                "scripts/gloggur watch init . --json",
                "scripts/gloggur watch start --daemon --json",
                "scripts/gloggur watch status --json",
                'scripts/gloggur search "add numbers token" --top-k 5 --json',
                "scripts/gloggur inspect . --json",
                "scripts/gloggur watch stop --json",
                'scripts/gloggur support collect --json --note "manual support snapshot"',
                "## Troubleshooting by Failure Code",
                "embedding_provider_error",
                "watch_mode_conflict",
                "watch_path_missing",
                "search_contract_v1_removed",
                "search_router_backends_failed",
            ]
        ),
    )
    _write(
        cli_path,
        "\n".join(
            [
                "watch_mode_conflict",
                "watch_path_missing",
                "search_contract_v1_removed",
                "search_router_backends_failed",
            ]
        ),
    )
    _write(embedding_errors_path, "embedding_provider_error\n")

    payload = check_quickstart_contract(
        docs_path,
        source_paths=[cli_path, embedding_errors_path],
    )

    assert payload["ok"] is True
    assert payload["missing_headings"] == []
    assert payload["missing_commands"] == []
    assert payload["missing_provider_snippets"] == []
    assert payload["missing_failure_codes_in_docs"] == []
    assert payload["missing_failure_codes_in_source"] == []


def test_quickstart_contract_fails_when_required_content_is_missing(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs" / "QUICKSTART.md"
    source_path = tmp_path / "src" / "gloggur" / "cli" / "main.py"

    _write(docs_path, "# incomplete\n")
    _write(source_path, "watch_mode_conflict\n")

    payload = check_quickstart_contract(docs_path, source_paths=[source_path])

    assert payload["ok"] is False
    assert payload["failure"]["code"] == "quickstart_contract_violation"
    assert "## Install and Bootstrap" in payload["missing_headings"]
    assert "embedding_provider_error" in payload["missing_failure_codes_in_source"]


def test_quickstart_contract_fails_when_docs_file_is_missing(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs" / "QUICKSTART.md"

    payload = check_quickstart_contract(docs_path)

    assert payload["ok"] is False
    assert payload["failure"]["code"] == "quickstart_contract_docs_missing"
