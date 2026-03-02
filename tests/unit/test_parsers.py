from __future__ import annotations

from gloggur.parsers.registry import EXTENSION_LANGUAGE, ParserRegistry
from gloggur.parsers.treesitter_parser import TreeSitterParser
from scripts.verification.fixtures import TestFixtures


def test_parser_registry_supports_known_extensions() -> None:
    """Parser registry should resolve known extensions."""
    registry = ParserRegistry()
    extensions = registry.supported_extensions()
    assert extensions == EXTENSION_LANGUAGE
    entry = registry.get_parser_for_path("sample.py")
    assert entry is not None
    assert entry.language == "python"
    assert isinstance(entry.parser, TreeSitterParser)


def test_treesitter_parser_extracts_python_symbols() -> None:
    """Tree-sitter parser should extract Python symbols."""
    source = TestFixtures.create_sample_python_file()
    parser = TreeSitterParser("python")
    symbols = parser.extract_symbols("sample.py", source)

    names = {symbol.name for symbol in symbols}
    assert "Greeter" in names
    assert "add" in names

    greeter = next(symbol for symbol in symbols if symbol.name == "Greeter")
    add = next(symbol for symbol in symbols if symbol.name == "add")
    init = next(symbol for symbol in symbols if symbol.name == "__init__")

    assert greeter.kind == "class"
    assert greeter.docstring is not None
    assert add.kind == "function"
    assert add.docstring is not None
    assert init.docstring is None


def test_parse_file_returns_parsed_file() -> None:
    """parse_file should return a ParsedFile with symbols."""
    source = TestFixtures.create_sample_python_file()
    parser = TreeSitterParser("python")
    parsed = parser.parse_file("sample.py", source)
    assert parsed.path == "sample.py"
    assert parsed.language == "python"
    assert parsed.source == source
    assert parsed.symbols

def test_treesitter_parser_extracts_python_fixtures() -> None:
    """Tree-sitter parser should extract Python fixtures."""
    source = """
import pytest

@pytest.fixture
def sample_fixture() -> int:
    return 42

@pytest.fixture(scope="session")
def session_fixture():
    yield "session"
"""
    parser = TreeSitterParser("python")
    symbols = parser.extract_symbols("sample.py", source)

    names = {symbol.name for symbol in symbols}
    assert "sample_fixture" in names
    assert "session_fixture" in names

    sample = next(symbol for symbol in symbols if symbol.name == "sample_fixture")
    session = next(symbol for symbol in symbols if symbol.name == "session_fixture")

    assert sample.kind == "fixture"
    assert session.kind == "fixture"
