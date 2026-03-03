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


def test_parser_registry_supports_extension_overrides() -> None:
    """Registry should allow extension -> adapter overrides from config."""
    registry = ParserRegistry(extension_map={".pyi": "python"})
    entry = registry.get_parser_for_path("sample.pyi")
    assert entry is not None
    assert entry.language == "python"


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


def test_treesitter_parser_extracts_python_invariants() -> None:
    """Tree-sitter parser should extract Python invariants from assert statements."""
    source = """
def test_foo():
    x = 1
    assert x == 1, "x should be 1"
    assert "foo" in ["bar", "foo"]
    assert True
"""
    parser = TreeSitterParser("python")
    symbols = parser.extract_symbols("test_source.py", source)

    test_foo_symbol = next(s for s in symbols if s.name == "test_foo")
    assert test_foo_symbol is not None
    assert len(test_foo_symbol.invariants) == 3
    assert test_foo_symbol.invariants[0] == 'x == 1, "x should be 1"'
    assert test_foo_symbol.invariants[1] == '"foo" in ["bar", "foo"]'
    assert test_foo_symbol.invariants[2] == "True"


def test_treesitter_parser_detects_serialization_boundaries() -> None:
    """Tree-sitter parser should detect serialization boundaries."""
    source = """
import json

def to_dict(self):
    return {"a": 1}

def regular_function():
    pass

def process_data(data):
    return json.loads(data)
"""
    parser = TreeSitterParser("python")
    symbols = parser.extract_symbols("test_source.py", source)

    to_dict_sym = next(s for s in symbols if s.name == "to_dict")
    regular_sym = next(s for s in symbols if s.name == "regular_function")
    process_sym = next(s for s in symbols if s.name == "process_data")

    assert to_dict_sym.is_serialization_boundary is True
    assert regular_sym.is_serialization_boundary is False
    assert process_sym.is_serialization_boundary is True


def test_treesitter_parser_parses_implicit_contracts() -> None:
    """Tree-sitter parser should parse implicit contracts from test names."""
    source = """
def test_when_logged_in_shows_dashboard():
    pass

def test_user_creation_fails_on_duplicate_email():
    pass
    
def regular_function():
    pass
"""
    parser = TreeSitterParser("python")
    symbols = parser.extract_symbols("test_source.py", source)

    logged_in_sym = next(s for s in symbols if s.name == "test_when_logged_in_shows_dashboard")
    creation_fails_sym = next(
        s for s in symbols if s.name == "test_user_creation_fails_on_duplicate_email"
    )
    regular_sym = next(s for s in symbols if s.name == "regular_function")

    assert logged_in_sym.implicit_contract == "when logged in shows dashboard"
    assert creation_fails_sym.implicit_contract == "user creation fails on duplicate email"
    assert regular_sym.implicit_contract is None


def test_treesitter_parser_projects_signals_to_legacy_fields() -> None:
    """Parser should emit normalized signals while preserving legacy field projections."""
    source = """
import json

def test_contract():
    assert 1 == 1
    return json.loads("{}")
"""
    parser = TreeSitterParser("python")
    symbols = parser.extract_symbols("test_source.py", source)
    target = next(s for s in symbols if s.name == "test_contract")
    signal_types = [signal.type for signal in target.signals]

    assert "code.invariant" in signal_types
    assert "code.call" in signal_types
    assert "boundary.serialization" in signal_types
    assert "test.implicit_contract" in signal_types
    assert target.implicit_contract == "contract"
    assert target.is_serialization_boundary is True


def test_treesitter_parser_does_not_apply_python_test_heuristics_to_typescript() -> None:
    """Non-Python parsing should not emit Python test contract signals."""
    source = """
function test_when_user_logs_in() {
  return 1;
}
"""
    parser = TreeSitterParser("typescript")
    symbols = parser.extract_symbols("sample.ts", source)
    target = next(s for s in symbols if s.name == "test_when_user_logs_in")
    assert target.implicit_contract is None


def test_treesitter_parser_go_symbols_parse_without_python_test_projection() -> None:
    """Go parser should extract symbols without Python-specific implicit contracts."""
    source = """
package main

func test_when_user_logs_in() int {
    return 1
}
"""
    parser = TreeSitterParser("go")
    symbols = parser.extract_symbols("sample.go", source)
    target = next(s for s in symbols if s.name == "test_when_user_logs_in")
    assert target.kind == "function"
    assert target.implicit_contract is None
