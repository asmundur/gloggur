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
        "def escape_leading_slashes(path: str) -> str:\n"
        "    return path\n",
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


def test_run_exact_backend_rewards_multi_pattern_mixed_query_coverage(
    monkeypatch,
    tmp_path,
) -> None:
    src_file = tmp_path / "src" / "flask" / "app.py"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text(
        "def process_response(ctx):\n"
        "    # preserve after_request order\n"
        "    for func in ctx._after_request_functions:\n"
        "        pass\n",
        encoding="utf8",
    )
    docs_file = tmp_path / "docs" / "reference.md"
    docs_file.parent.mkdir(parents=True, exist_ok=True)
    docs_file.write_text(
        "after_request hooks exist.\n"
        "registration order is described elsewhere.\n",
        encoding="utf8",
    )
    test_file = tmp_path / "tests" / "test_app.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "def test_after_request_hook() -> None:\n"
        '    assert "after_request" in "after_request"\n',
        encoding="utf8",
    )

    def _run(cmd, **_kwargs):
        pattern = cmd[cmd.index("-e") + 1]
        outputs = {
            "after_request": (
                "src/flask/app.py:2:    # preserve after_request order\n"
                "tests/test_app.py:1:def test_after_request_hook() -> None:\n"
                "docs/reference.md:1:after_request hooks exist.\n"
            ),
            "order": (
                "src/flask/app.py:2:    # preserve after_request order\n"
                "docs/reference.md:2:registration order is described elsewhere.\n"
            ),
        }
        stdout = outputs.get(pattern, "")
        return SimpleNamespace(
            returncode=0 if stdout else 1,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr("gloggur.search.router.backends.subprocess.run", _run)

    result = run_exact_backend(
        query="after_request order",
        hints=extract_query_hints("after_request order"),
        repo_root=tmp_path,
        intent=SearchIntent(max_snippets=3, time_budget_ms=900),
        execution_hints=ExecutionHints(),
        config=SearchRouterConfig(),
    )

    assert result.hits
    assert Path(result.hits[0].path).as_posix() == str(src_file).replace("\\", "/")
    assert len(result.hits) >= 2
    assert Path(result.hits[-1].path).as_posix() == str(docs_file).replace("\\", "/")
