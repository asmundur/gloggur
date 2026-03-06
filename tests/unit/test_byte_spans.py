from __future__ import annotations

from pathlib import Path

import pytest

from gloggur.byte_spans import (
    LineByteSpanIndex,
    RepoPathResolutionError,
    discover_repo_root,
    resolve_repo_relative_path,
    to_repo_relative_path,
)


def test_line_byte_span_index_handles_ascii_and_lf() -> None:
    index = LineByteSpanIndex.from_bytes(b"alpha\nbeta\n")

    assert index.span_for_lines(1, 1) == (0, 6)
    assert index.span_for_lines(2, 2) == (6, 11)
    assert index.extract_text(0, 6) == "alpha\n"


def test_line_byte_span_index_handles_crlf() -> None:
    index = LineByteSpanIndex.from_bytes(b"alpha\r\nbeta\r\n")

    assert index.span_for_lines(1, 1) == (0, 7)
    assert index.span_for_lines(2, 2) == (7, 13)
    assert index.extract_bytes(0, 7) == b"alpha\r\n"


def test_line_byte_span_index_handles_multibyte_utf8_and_no_trailing_newline() -> None:
    payload = "á\nβeta".encode("utf8")
    index = LineByteSpanIndex.from_bytes(payload)

    assert index.line_count == 2
    assert index.extract_text(*index.span_for_lines(1, 1)) == "á\n"
    assert index.extract_text(*index.span_for_lines(2, 2)) == "βeta"


def test_line_byte_span_index_decodes_invalid_utf8_with_replacement() -> None:
    index = LineByteSpanIndex.from_bytes(b"ok\xff\n")

    assert index.extract_text(0, 4) == "ok\ufffd\n"


def test_line_byte_span_index_allows_zero_length_eof_slice() -> None:
    index = LineByteSpanIndex.from_bytes(b"abc")

    assert index.extract_text(3, 3) == ""


def test_line_byte_span_index_rejects_invalid_ranges() -> None:
    index = LineByteSpanIndex.from_bytes(b"abc")

    with pytest.raises(ValueError):
        index.extract_bytes(-1, 0)
    with pytest.raises(ValueError):
        index.extract_bytes(2, 1)
    with pytest.raises(IndexError):
        index.extract_bytes(0, 4)


def test_repo_path_resolution_discovers_root_and_rejects_escapes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (nested / "sample.py").write_text("print('ok')\n", encoding="utf8")

    assert discover_repo_root(nested) == repo
    assert resolve_repo_relative_path(repo, "src/pkg/sample.py") == nested / "sample.py"
    assert to_repo_relative_path(repo, str(nested / "sample.py")) == "src/pkg/sample.py"

    with pytest.raises(RepoPathResolutionError):
        resolve_repo_relative_path(repo, "/tmp/escape.py")
    with pytest.raises(RepoPathResolutionError):
        resolve_repo_relative_path(repo, "../escape.py")
