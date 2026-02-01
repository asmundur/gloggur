from __future__ import annotations

from gloggur.models import Symbol
from gloggur.validation.docstring_validator import validate_docstrings


def _symbol(
    *,
    symbol_id: str,
    name: str,
    kind: str = "function",
    signature: str | None = None,
    docstring: str | None = None,
) -> Symbol:
    return Symbol(
        id=symbol_id,
        name=name,
        kind=kind,
        file_path="sample.py",
        start_line=1,
        end_line=2,
        signature=signature,
        docstring=docstring,
        body_hash="abc123",
        language="python",
    )


def test_validation_reports_missing_docstring() -> None:
    symbol = _symbol(symbol_id="s1", name="add", signature="def add(a, b):", docstring=None)
    reports = validate_docstrings([symbol])
    assert len(reports) == 1
    assert reports[0].warnings == ["Missing docstring"]


def test_validation_skips_private_symbols() -> None:
    symbol = _symbol(symbol_id="s2", name="_internal", signature="def _internal():", docstring=None)
    reports = validate_docstrings([symbol])
    assert reports == []


def test_validation_flags_unknown_style_and_missing_params() -> None:
    symbol = _symbol(
        symbol_id="s3",
        name="compute",
        signature="def compute(a, b):",
        docstring="Compute something.",
    )
    reports = validate_docstrings([symbol])
    assert len(reports) == 1
    warnings = reports[0].warnings
    assert "Unknown docstring style" in warnings
    assert any("Missing docstring params" in warning for warning in warnings)
