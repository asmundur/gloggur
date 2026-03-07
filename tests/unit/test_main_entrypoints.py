from __future__ import annotations

import runpy
import sys

import pytest

import gloggur.__main__ as package_main
import gloggur.cli.main as cli_main_module


def test_package_main_returns_cli_main_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(package_main, "cli_main", lambda: 7)

    assert package_main.main() == 7


def test_package_module_execution_raises_system_exit_with_main_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_main_module, "main", lambda: 5)
    sys.modules.pop("gloggur.__main__", None)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("gloggur.__main__", run_name="__main__")

    assert exit_info.value.code == 5
