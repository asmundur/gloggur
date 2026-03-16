from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gloggur.search.router.backends import run_exact_backend
from gloggur.search.router.config import SearchRouterConfig
from gloggur.search.router.hints import extract_query_hints
from gloggur.search.router.types import ExecutionHints, SearchIntent


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
