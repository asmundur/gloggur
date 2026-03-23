from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gloggur.search.router.backends import run_exact_backend, run_symbol_backend
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.hints import extract_query_hints
from gloggur.search.router.types import ExecutionHints, SearchIntent
from gloggur.symbol_index.models import SymbolOccurrence


def test_run_exact_backend_uses_remaining_budget_for_single_pattern(
    monkeypatch,
    tmp_path,
) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def Foo(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def caller(value: int) -> int:\n"
        "    return Foo(value)\n",
        encoding="utf8",
    )

    observed_timeouts: list[float] = []
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(tmp_path / "ripgrep-config"))

    def _run(cmd, **kwargs):
        del cmd
        observed_timeouts.append(float(kwargs["timeout"]))
        assert "RIPGREP_CONFIG_PATH" not in kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout="sample.py:4:def caller(value: int) -> int:\n",
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="caller",
        hints=extract_query_hints("caller"),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=1, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert result.hits
    assert len(observed_timeouts) == 1
    assert observed_timeouts[0] > 0.5


def test_run_exact_backend_falls_back_when_ripgrep_is_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def Foo(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def caller(value: int) -> int:\n"
        "    return Foo(value)\n",
        encoding="utf8",
    )
    hidden_binary = tmp_path / ".gloggur" / "index" / "symbols.db"
    hidden_binary.parent.mkdir(parents=True, exist_ok=True)
    hidden_binary.write_bytes(b"SQLite format 3\x00caller\x8d\xfa")

    def _run(*_args, **_kwargs):
        raise FileNotFoundError("rg not found")

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="caller",
        hints=extract_query_hints("caller"),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=1, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert result.hits
    assert result.hits[0].path.endswith("sample.py")
    assert "python_fallback_exact_scan" in result.commands


def test_run_exact_backend_ranks_source_definitions_above_docs_and_tests(
    monkeypatch,
    tmp_path,
) -> None:
    src_file = tmp_path / "src" / "http.py"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text(
        "def escape_leading_slashes(path: str) -> str:\n" "    return path\n",
        encoding="utf8",
    )
    docs_file = tmp_path / "docs" / "changelog.md"
    docs_file.parent.mkdir(parents=True, exist_ok=True)
    docs_file.write_text(
        "escape_leading_slashes now handles more URL forms.\n",
        encoding="utf8",
    )
    test_file = tmp_path / "tests" / "test_http.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "def test_escape_leading_slashes() -> None:\n"
        "    assert escape_leading_slashes('//x') == '/x'\n",
        encoding="utf8",
    )

    def _run(_cmd, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "src/http.py:1:def escape_leading_slashes(path: str) -> str:\n"
                "docs/changelog.md:1:escape_leading_slashes now handles more URL forms.\n"
                "tests/test_http.py:2:    assert escape_leading_slashes('//x') == '/x'\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="escape_leading_slashes",
        hints=extract_query_hints("escape_leading_slashes"),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(fixed_string=True),
        config=SearchRouterConfig(),
    )

    assert [Path(hit.path).as_posix() for hit in result.hits] == [
        str(src_file).replace("\\", "/"),
        str(test_file).replace("\\", "/"),
        str(docs_file).replace("\\", "/"),
    ]


def _write_django_search_fixture(
    tmp_path: Path,
    *,
    definition_path: str,
    definition_line: str,
    importer_path: str,
    importer_line: str,
    reference_path: str | None = None,
    reference_line: str | None = None,
) -> tuple[Path, Path, Path | None]:
    definition_file = tmp_path / definition_path
    definition_file.parent.mkdir(parents=True, exist_ok=True)
    definition_file.write_text(f"{definition_line}\n    return value\n", encoding="utf8")

    importer_file = tmp_path / importer_path
    importer_file.parent.mkdir(parents=True, exist_ok=True)
    importer_file.write_text(f"{importer_line}\n", encoding="utf8")

    reference_file: Path | None = None
    if reference_path is not None and reference_line is not None:
        reference_file = tmp_path / reference_path
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        reference_file.write_text(f"{reference_line}\n", encoding="utf8")

    return definition_file, importer_file, reference_file


@pytest.mark.parametrize(
    ("query", "definition_path", "definition_line", "importer_path", "importer_line"),
    [
        (
            "url_has_allowed_host_and_scheme",
            "django/utils/http.py",
            "def url_has_allowed_host_and_scheme(url, allowed_hosts, require_https=False):",
            "django/contrib/auth/views.py",
            "from django.utils.http import url_has_allowed_host_and_scheme",
        ),
        (
            "escape_leading_slashes",
            "django/utils/http.py",
            "def escape_leading_slashes(url):",
            "django/middleware/common.py",
            "from django.utils.http import escape_leading_slashes",
        ),
        (
            "escapejs",
            "django/utils/html.py",
            "def escapejs(value):",
            "django/template/defaultfilters.py",
            "from django.utils.html import avoid_wrapping, conditional_escape, escape, escapejs",
        ),
    ],
)
def test_run_exact_backend_code_identifier_queries_rank_definitions_before_importers(
    monkeypatch,
    tmp_path,
    query: str,
    definition_path: str,
    definition_line: str,
    importer_path: str,
    importer_line: str,
) -> None:
    definition_file, importer_file, _reference_file = _write_django_search_fixture(
        tmp_path,
        definition_path=definition_path,
        definition_line=definition_line,
        importer_path=importer_path,
        importer_line=importer_line,
        reference_path=definition_path,
        reference_line=f"result = {query}(value)",
    )

    def _run(_cmd, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                f"{importer_path}:1:{importer_line}\n"
                f"{definition_path}:1:{definition_line}\n"
                f"{definition_path}:2:result = {query}(value)\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query=query,
        hints=extract_query_hints(query),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(fixed_string=True),
        config=SearchRouterConfig(),
    )

    assert result.hits
    assert result.hits[0].path == str(definition_file)
    assert result.hits[0].match_role == "definition"
    assert result.hits[1].path == str(definition_file)
    assert result.hits[1].match_role == "same_file_context"
    assert result.hits[2].path == str(importer_file)
    assert result.hits[2].match_role == "import"


@pytest.mark.parametrize(
    ("query", "definition_path", "definition_line", "importer_path", "importer_line"),
    [
        (
            "def url_has_allowed_host_and_scheme",
            "django/utils/http.py",
            "def url_has_allowed_host_and_scheme(url, allowed_hosts, require_https=False):",
            "django/contrib/auth/views.py",
            "from django.utils.http import url_has_allowed_host_and_scheme",
        ),
        (
            "def escape_leading_slashes",
            "django/utils/http.py",
            "def escape_leading_slashes(url):",
            "django/middleware/common.py",
            "from django.utils.http import escape_leading_slashes",
        ),
    ],
)
def test_run_exact_backend_code_declaration_queries_return_only_declarations_when_present(
    monkeypatch,
    tmp_path,
    query: str,
    definition_path: str,
    definition_line: str,
    importer_path: str,
    importer_line: str,
) -> None:
    definition_file, _importer_file, _reference_file = _write_django_search_fixture(
        tmp_path,
        definition_path=definition_path,
        definition_line=definition_line,
        importer_path=importer_path,
        importer_line=importer_line,
        reference_path=definition_path,
        reference_line=f"value = {query.split()[-1]}(request.path)",
    )

    def _run(_cmd, **_kwargs):
        symbol = query.split()[-1]
        return SimpleNamespace(
            returncode=0,
            stdout=(
                f"{importer_path}:1:{importer_line}\n"
                f"{definition_path}:1:{definition_line}\n"
                f"{definition_path}:2:value = {symbol}(request.path)\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query=query,
        hints=extract_query_hints(query),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(fixed_string=True),
        config=SearchRouterConfig(),
    )

    assert [hit.path for hit in result.hits] == [str(definition_file)]
    assert result.hits[0].match_role == "definition"


def test_run_exact_backend_mixed_code_queries_keep_implementation_and_test_near_top(
    monkeypatch,
    tmp_path,
) -> None:
    app_file = tmp_path / "src" / "flask" / "app.py"
    app_file.parent.mkdir(parents=True, exist_ok=True)
    app_file.write_text(
        "def make_response(rv):\n"
        "    return _coerce_response_tuple(rv)\n",
        encoding="utf8",
    )
    helper_file = tmp_path / "src" / "flask" / "helpers.py"
    helper_file.parent.mkdir(parents=True, exist_ok=True)
    helper_file.write_text("from .app import make_response\n", encoding="utf8")
    test_file = tmp_path / "tests" / "test_app.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "def test_make_response_tuple():\n"
        "    assert make_response((body, 200))[1] == 200\n",
        encoding="utf8",
    )

    def _run(_cmd, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "src/flask/helpers.py:1:from .app import make_response\n"
                "src/flask/app.py:1:def make_response(rv):\n"
                "tests/test_app.py:2:    assert make_response((body, 200))[1] == 200\n"
                "src/flask/app.py:2:    return _coerce_response_tuple(rv)\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="make_response tuple",
        hints=extract_query_hints("make_response tuple"),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=4, time_budget_ms=900),
        execution_hints=ExecutionHints(fixed_string=True),
        config=SearchRouterConfig(),
    )

    assert [Path(hit.path).as_posix() for hit in result.hits[:2]] == [
        str(app_file).replace("\\", "/"),
        str(test_file).replace("\\", "/"),
    ]


def test_run_symbol_backend_ranks_definitions_before_references_for_code_identifier_queries(
    tmp_path,
) -> None:
    definition_file = tmp_path / "django" / "utils" / "http.py"
    definition_file.parent.mkdir(parents=True, exist_ok=True)
    definition_file.write_text(
        "def url_has_allowed_host_and_scheme(url, allowed_hosts, require_https=False):\n"
        "    return True\n",
        encoding="utf8",
    )
    reference_file = tmp_path / "django" / "contrib" / "auth" / "views.py"
    reference_file.parent.mkdir(parents=True, exist_ok=True)
    reference_file.write_text(
        "from django.utils.http import url_has_allowed_host_and_scheme\n"
        "url_has_allowed_host_and_scheme(next_page, allowed_hosts)\n",
        encoding="utf8",
    )

    class FakeStore:
        available = True
        unavailability_reason = None

        def list_occurrences(self, **_kwargs):
            return [
                SymbolOccurrence(
                    symbol="url_has_allowed_host_and_scheme",
                    kind="ref",
                    path="django/contrib/auth/views.py",
                    start_line=2,
                    end_line=2,
                ),
                SymbolOccurrence(
                    symbol="url_has_allowed_host_and_scheme",
                    kind="def",
                    path="django/utils/http.py",
                    start_line=1,
                    end_line=1,
                ),
            ]

    result = run_symbol_backend(
        symbol_store=FakeStore(),
        hints=extract_query_hints("url_has_allowed_host_and_scheme"),
        query="url_has_allowed_host_and_scheme",
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=2, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert [hit.path for hit in result.hits] == [str(definition_file), str(reference_file)]
    assert [hit.match_role for hit in result.hits] == ["definition", "reference"]


def test_run_symbol_backend_code_declaration_queries_filter_to_definitions_when_present(
    tmp_path,
) -> None:
    definition_file = tmp_path / "django" / "utils" / "http.py"
    definition_file.parent.mkdir(parents=True, exist_ok=True)
    definition_file.write_text(
        "def escape_leading_slashes(url):\n"
        "    return url\n",
        encoding="utf8",
    )
    reference_file = tmp_path / "django" / "middleware" / "common.py"
    reference_file.parent.mkdir(parents=True, exist_ok=True)
    reference_file.write_text(
        "from django.utils.http import escape_leading_slashes\n"
        "path = escape_leading_slashes(request.path)\n",
        encoding="utf8",
    )

    class FakeStore:
        available = True
        unavailability_reason = None

        def list_occurrences(self, **_kwargs):
            return [
                SymbolOccurrence(
                    symbol="escape_leading_slashes",
                    kind="ref",
                    path="django/middleware/common.py",
                    start_line=2,
                    end_line=2,
                ),
                SymbolOccurrence(
                    symbol="escape_leading_slashes",
                    kind="def",
                    path="django/utils/http.py",
                    start_line=1,
                    end_line=1,
                ),
            ]

    result = run_symbol_backend(
        symbol_store=FakeStore(),
        hints=extract_query_hints("def escape_leading_slashes"),
        query="def escape_leading_slashes",
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=2, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert [hit.path for hit in result.hits] == [str(definition_file)]
    assert result.hits[0].match_role == "definition"


def test_run_exact_backend_keeps_repo_relative_path_filters_when_repo_root_collapses(
    monkeypatch,
    tmp_path,
) -> None:
    repo_root = tmp_path / "src"
    repo_root.mkdir(parents=True, exist_ok=True)
    sample = repo_root / "a_auth_token.py"
    sample.write_text(
        "def refresh_auth_state():\n" "    return 'token-auth'\n",
        encoding="utf8",
    )

    def _run(_cmd, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="a_auth_token.py:2:    return 'token-auth'\n",
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="token",
        hints=extract_query_hints("token"),
        repo_root=repo_root,
        intent=SearchIntent(path_filters=("src/",), max_snippets=1, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert result.hits
    assert result.hits[0].path.replace("\\", "/") == str(sample).replace("\\", "/")


def test_run_exact_backend_locator_verbatim_query_uses_full_literal_fixed_string(
    monkeypatch,
    tmp_path,
) -> None:
    sample = tmp_path / "tests" / "test_http.py"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        "assert normalize_url('http:///example.com') == 'http:///example.com'\n",
        encoding="utf8",
    )

    observed_patterns: list[str] = []

    def _run(cmd, **_kwargs):
        observed_patterns.append(cmd[cmd.index("-e") + 1])
        assert "-F" in cmd
        return SimpleNamespace(
            returncode=0,
            stdout="tests/test_http.py:1:assert normalize_url('http:///example.com') == 'http:///example.com'\n",
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="http:///example.com",
        hints=extract_query_hints("http:///example.com"),
        repo_root=tmp_path,
        intent=SearchIntent(result_profile="locator", max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert observed_patterns == ["http:///example.com"]
    assert result.hits
    assert result.hits[0].path.replace("\\", "/").endswith("tests/test_http.py")


def test_run_exact_backend_locator_verbatim_query_does_not_fallback_to_fragments_after_miss(
    monkeypatch,
    tmp_path,
) -> None:
    observed_patterns: list[str] = []

    def _run(cmd, **_kwargs):
        observed_patterns.append(cmd[cmd.index("-e") + 1])
        assert "-F" in cmd
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="//example.com/path",
        hints=extract_query_hints("//example.com/path"),
        repo_root=tmp_path,
        intent=SearchIntent(result_profile="locator", max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert observed_patterns == ["//example.com/path"]
    assert result.hits == ()


def test_run_exact_backend_locator_literal_first_query_uses_fixed_string_without_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    sample = tmp_path / "docs" / "internals" / "committing-code.txt"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("[5.2.x]\n", encoding="utf8")

    observed_patterns: list[str] = []

    def _run(cmd, **_kwargs):
        observed_patterns.append(cmd[cmd.index("-e") + 1])
        assert "-F" in cmd
        return SimpleNamespace(
            returncode=0,
            stdout="docs/internals/committing-code.txt:1:[5.2.x]\n",
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="[5.2.x]",
        hints=extract_query_hints("[5.2.x]"),
        repo_root=tmp_path,
        intent=SearchIntent(result_profile="locator", max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert observed_patterns == ["[5.2.x]"]
    assert result.hits
    assert result.hits[0].path.replace("\\", "/").endswith("docs/internals/committing-code.txt")


def test_run_exact_backend_workflow_query_searches_hidden_authority_paths(
    monkeypatch,
    tmp_path,
) -> None:
    observed_commands: list[list[str]] = []

    def _run(cmd, **_kwargs):
        observed_commands.append(list(cmd))
        assert "--hidden" in cmd
        return SimpleNamespace(
            returncode=0,
            stdout=".github/workflows/verification.yml:1:name: verification\n",
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="python matrix label source",
        hints=extract_query_hints("python matrix label source"),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=1, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert observed_commands
    assert result.hits
    assert result.hits[0].path.replace("\\", "/").endswith(
        "/.github/workflows/verification.yml"
    )


def test_run_exact_backend_workflow_parity_queries_rank_authority_files_ahead_of_repo_noise(
    monkeypatch,
    tmp_path,
) -> None:
    observed_commands: list[list[str]] = []

    def _run(cmd, **_kwargs):
        observed_commands.append(list(cmd))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "src/docs_helpers.py:1:def docs_ci_parity_check():\n"
                "docs/Makefile:1:html:\n"
                ".github/workflows/docs.yml:1:name: docs\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="docs ci parity",
        hints=extract_query_hints("docs ci parity"),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert observed_commands
    assert [Path(hit.path).as_posix() for hit in result.hits[:2]] == [
        str(tmp_path / ".github" / "workflows" / "docs.yml").replace("\\", "/"),
        str(tmp_path / "docs" / "Makefile").replace("\\", "/"),
    ]


def test_run_exact_backend_non_locator_verbatim_query_falls_back_after_literal_miss(
    monkeypatch,
    tmp_path,
) -> None:
    sample = tmp_path / "tests" / "test_http.py"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("assert scheme == 'http'\n", encoding="utf8")

    observed_patterns: list[str] = []

    def _run(cmd, **_kwargs):
        pattern = cmd[cmd.index("-e") + 1]
        observed_patterns.append(pattern)
        if pattern == "http:///example.com":
            assert "-F" in cmd
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if pattern == "http":
            return SimpleNamespace(
                returncode=0,
                stdout="tests/test_http.py:1:assert scheme == 'http'\n",
                stderr="",
            )
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="http:///example.com",
        hints=extract_query_hints("http:///example.com"),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert observed_patterns[:2] == ["http:///example.com", "http"]
    assert "example.com" in observed_patterns
    assert result.hits
