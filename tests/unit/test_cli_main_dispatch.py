from __future__ import annotations

import json

import click
import pytest

from gloggur.cli import main as cli_main
from gloggur.embeddings.errors import EmbeddingProviderError
from gloggur.io_failures import StorageIOError


def _payload(output: str) -> dict[str, object]:
    payload = json.loads(output.strip())
    assert isinstance(payload, dict)
    return payload


def test_main_returns_click_exit_code_from_click_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_main.cli,
        "main",
        lambda args, prog_name, standalone_mode: (_ for _ in ()).throw(
            click.exceptions.Exit(3)
        ),
    )

    assert cli_main.main(["status"]) == 3


def test_main_non_json_click_exception_shows_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_main.cli,
        "main",
        lambda args, prog_name, standalone_mode: (_ for _ in ()).throw(
            click.UsageError("bad flag")
        ),
    )

    exit_code = cli_main.main(["status"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "bad flag" in captured.err


def test_main_invalid_global_option_json_emits_dispatch_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main.main(["--bogus", "--json"])
    payload = _payload(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["error_code"] == "cli_usage_error"
    assert payload["stage"] == "dispatch"


def test_main_missing_required_argument_json_emits_dispatch_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main.main(["search", "--json"])
    payload = _payload(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["error_code"] == "cli_usage_error"
    assert payload["stage"] == "dispatch"


def test_with_io_failure_handling_json_click_exception_emits_usage_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _callback(**_kwargs: object) -> None:
        raise click.UsageError("bad flag")

    wrapped = cli_main._with_io_failure_handling(_callback)

    with pytest.raises(click.exceptions.Exit) as exc_info:
        wrapped(as_json=True)

    payload = _payload(capsys.readouterr().out)
    assert exc_info.value.exit_code == 2
    error = payload["error"]
    assert isinstance(error, dict)
    assert error["code"] == "cli_usage_error"
    assert payload["failure_codes"] == ["cli_usage_error"]


def test_with_io_failure_handling_non_json_cli_contract_error_reraises() -> None:
    def _callback(**_kwargs: object) -> None:
        raise cli_main.CLIContractError("bad contract", error_code="cli_usage_error")

    wrapped = cli_main._with_io_failure_handling(_callback)

    with pytest.raises(cli_main.CLIContractError) as exc_info:
        wrapped()

    assert exc_info.value.error_code == "cli_usage_error"


def test_with_io_failure_handling_non_json_click_exception_reraises() -> None:
    def _callback(**_kwargs: object) -> None:
        raise click.UsageError("bad flag")

    wrapped = cli_main._with_io_failure_handling(_callback)

    with pytest.raises(click.UsageError):
        wrapped()


def test_with_io_failure_handling_plain_storage_io_error_prints_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = StorageIOError(
        category="permission_denied",
        operation="write cache file",
        path="/tmp/cache",
        probable_cause="permission denied",
        remediation=["fix permissions"],
        detail="PermissionError: denied",
    )

    def _callback(**_kwargs: object) -> None:
        raise error

    wrapped = cli_main._with_io_failure_handling(_callback)

    with pytest.raises(click.exceptions.Exit) as exc_info:
        wrapped()

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 1
    assert captured.out == ""
    assert "write cache file" in captured.err
    assert "Remediation:" in captured.err


def test_with_io_failure_handling_plain_embedding_error_prints_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = EmbeddingProviderError(
        provider="local",
        operation="load model",
        detail="RuntimeError: boom",
        remediation=["reinstall model"],
    )

    def _callback(**_kwargs: object) -> None:
        raise error

    wrapped = cli_main._with_io_failure_handling(_callback)

    with pytest.raises(click.exceptions.Exit) as exc_info:
        wrapped()

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 1
    assert captured.out == ""
    assert "Embedding provider failure [local]" in captured.err
    assert "reinstall model" in captured.err
