from __future__ import annotations

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
