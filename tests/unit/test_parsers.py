from __future__ import annotations

import pytest

from gloggur.parsers.registry import EXTENSION_LANGUAGE, ParserRegistry
from gloggur.parsers.treesitter_parser import TreeSitterParser
from scripts.verification.fixtures import TestFixtures


def _symbols_by_fqname(symbols):
    return {symbol.fqname or symbol.name: symbol for symbol in symbols}


def _symbols_grouped_by_fqname(symbols):
    grouped = {}
    for symbol in symbols:
        grouped.setdefault(symbol.fqname or symbol.name, []).append(symbol)
    return grouped


def test_parser_registry_supports_known_extensions() -> None:
    """Parser registry should resolve known extensions."""
    registry = ParserRegistry()
    extensions = registry.supported_extensions()
    assert extensions == EXTENSION_LANGUAGE
    entry = registry.get_parser_for_path("sample.py")
    assert entry is not None
    assert entry.language == "python"
    assert isinstance(entry.parser, TreeSitterParser)
    c_entry = registry.get_parser_for_path("sample.h")
    assert c_entry is not None
    assert c_entry.language == "c"
    cpp_entry = registry.get_parser_for_path("sample.hpp")
    assert cpp_entry is not None
    assert cpp_entry.language == "cpp"


def test_parser_registry_supports_extension_overrides() -> None:
    """Registry should allow extension -> adapter overrides from config."""
    registry = ParserRegistry(extension_map={".pyi": "python"})
    entry = registry.get_parser_for_path("sample.pyi")
    assert entry is not None
    assert entry.language == "python"


def test_parser_registry_defers_native_parser_initialization_until_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry lookup should not initialize native C/C++ grammars until parse time."""
    calls: list[tuple[str, object]] = []
    sentinel_tree = object()

    class _FakeParser:
        def parse(self, source: bytes, old_tree=None) -> object:  # noqa: ANN001
            calls.append(("parse", source))
            return sentinel_tree

    def _fake_get_parser(language: str) -> _FakeParser:
        calls.append(("get_parser", language))
        return _FakeParser()

    monkeypatch.setattr("gloggur.parsers.treesitter_parser.get_parser", _fake_get_parser)
    entry = ParserRegistry().get_parser_for_path("sample.hpp")

    assert entry is not None
    assert entry.language == "cpp"
    assert calls == []

    tree = entry.parser.parse_with_edit(None, "int ping();\n")

    assert tree is sentinel_tree
    assert calls == [("get_parser", "cpp"), ("parse", b"int ping();\n")]


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


@pytest.mark.native_parser
def test_treesitter_parser_extracts_c_function_and_type_symbols() -> None:
    """C parser should recover function definitions/declarations and named type declarations."""
    source = """
int add(int a, int b) { return a + b; }
int declared(int value);
struct Point { int x; int y; };
enum Mode { Fast, Slow };
"""
    parser = TreeSitterParser("c")
    symbols = parser.extract_symbols("sample.c", source)
    by_fqname = _symbols_by_fqname(symbols)

    assert {"add", "declared", "Point", "Mode"}.issubset(by_fqname)
    assert by_fqname["add"].kind == "function"
    assert by_fqname["declared"].kind == "function"
    assert by_fqname["Point"].kind == "type"
    assert by_fqname["Mode"].kind == "enum"


@pytest.mark.native_parser
def test_treesitter_parser_extracts_c_callback_returning_function_pointer() -> None:
    """C parser should classify callback-return declarators as callable functions."""
    parser = TreeSitterParser("c")
    symbols = parser.extract_symbols("sample.h", "int (*make_cb(void))(int);\n")
    by_fqname = _symbols_by_fqname(symbols)

    assert "make_cb" in by_fqname
    assert by_fqname["make_cb"].kind == "function"


@pytest.mark.native_parser
def test_treesitter_parser_skips_c_function_pointer_variables() -> None:
    """C parser should not classify plain function-pointer variables as functions."""
    parser = TreeSitterParser("c")
    symbols = parser.extract_symbols("sample.h", "int (*fp)(int);\n")

    assert symbols == []


@pytest.mark.native_parser
def test_treesitter_parser_extracts_cpp_class_and_qualified_method_symbols() -> None:
    """C++ parser should recover class-body methods and qualified out-of-class method owners."""
    source = """
class Greeter {
public:
  int hello() const { return 1; }
  static int ping();
};

int Greeter::ping() { return 2; }
"""
    parser = TreeSitterParser("cpp")
    symbols = parser.extract_symbols("sample.cpp", source)
    grouped = _symbols_grouped_by_fqname(symbols)

    assert "Greeter" in grouped
    assert "Greeter.hello" in grouped
    assert "Greeter.ping" in grouped
    assert grouped["Greeter"][0].kind == "class"
    assert grouped["Greeter.hello"][0].kind == "method"
    assert grouped["Greeter.hello"][0].container_fqname == "Greeter"
    ping_symbols = grouped["Greeter.ping"]
    assert ping_symbols
    assert all(symbol.kind == "method" for symbol in ping_symbols)
    assert all(symbol.container_fqname == "Greeter" for symbol in ping_symbols)
    assert any((symbol.signature or "").startswith("static int ping();") for symbol in ping_symbols)
    assert any("Greeter::ping" in (symbol.signature or "") for symbol in ping_symbols)


@pytest.mark.native_parser
def test_treesitter_parser_extracts_cpp_namespace_template_and_operator_symbols() -> None:
    """C++ parser should preserve namespace/template owners and operator method names."""
    source = """
namespace core {
template <typename T>
class Box {
public:
  T get() const;
};

class Engine {
public:
  int start();
};
}

template <>
int core::Box<int>::get() const { return 1; }
int core::Engine::start() { return 1; }

struct Vec {
  int operator[](int idx) const;
};
int Vec::operator[](int idx) const { return idx; }
"""
    parser = TreeSitterParser("cpp")
    symbols = parser.extract_symbols("sample.cpp", source)
    grouped = _symbols_grouped_by_fqname(symbols)

    assert "core.Box" in grouped
    assert "core.Box.get" in grouped
    assert "core.Box<int>.get" in grouped
    assert "core.Engine" in grouped
    assert "core.Engine.start" in grouped
    assert "Vec.operator[]" in grouped
    assert all(symbol.name == "operator[]" for symbol in grouped["Vec.operator[]"])


@pytest.mark.native_parser
def test_treesitter_parser_recovers_cpp_macro_generated_method_symbols() -> None:
    """C++ parser should recover strict method macros and dedupe explicit duplicates."""
    parser = TreeSitterParser("cpp")
    recovered = parser.extract_symbols(
        "sample.cpp",
        (
            "class Greeter {};\n"
            "#define DECL_METHOD(Type, Name) int Type::Name() { return 0; }\n"
            "DECL_METHOD(Greeter, ping)\n"
        ),
    )
    recovered_by_fqname = _symbols_by_fqname(recovered)
    assert "Greeter.ping" in recovered_by_fqname
    assert recovered_by_fqname["Greeter.ping"].attributes["macro_generated"] is True

    deduped = parser.extract_symbols(
        "sample.cpp",
        (
            "class Greeter { public: int ping(); };\n"
            "#define DECL_METHOD(Type, Name) int Type::Name() { return 0; }\n"
            "DECL_METHOD(Greeter, ping)\n"
        ),
    )
    assert sum(1 for symbol in deduped if symbol.fqname == "Greeter.ping") == 1


def test_treesitter_parser_extracts_javascript_assignment_bound_symbols() -> None:
    """JavaScript parser should recover assignment-bound canonical fqnames and owners."""
    source = """
exports.send = function send() {};
module.exports.json = function () {};
proto.use = function (fn) {};
Router.prototype.route = function route(path) {};
app.listen = function listen() {};
const api = { send() {}, json: function () {}, end: () => {} };
const render = () => {};
const listen = function () {};
items.map(function (x) { return x; });
"""
    parser = TreeSitterParser("javascript")
    symbols = parser.extract_symbols("sample.js", source)
    by_fqname = _symbols_by_fqname(symbols)

    assert set(by_fqname) == {
        "exports.send",
        "module.exports.json",
        "proto.use",
        "Router.prototype.route",
        "app.listen",
        "api.send",
        "api.json",
        "api.end",
        "render",
        "listen",
    }

    assert by_fqname["exports.send"].kind == "function"
    assert by_fqname["exports.send"].name == "send"
    assert by_fqname["exports.send"].container_fqname == "exports"
    assert by_fqname["exports.send"].attributes["binding_style"] == "export_assignment"

    assert by_fqname["module.exports.json"].kind == "function"
    assert by_fqname["module.exports.json"].container_fqname == "module.exports"
    assert by_fqname["module.exports.json"].attributes["binding_style"] == "export_assignment"

    assert by_fqname["proto.use"].kind == "method"
    assert by_fqname["proto.use"].container_fqname == "proto"
    assert by_fqname["proto.use"].attributes["binding_style"] == "member_assignment"

    assert by_fqname["Router.prototype.route"].kind == "method"
    assert by_fqname["Router.prototype.route"].container_fqname == "Router.prototype"
    assert by_fqname["Router.prototype.route"].attributes["binding_style"] == (
        "prototype_assignment"
    )

    assert by_fqname["api.send"].kind == "method"
    assert by_fqname["api.send"].container_fqname == "api"
    assert by_fqname["api.send"].attributes["binding_style"] == "object_binding_property"
    assert by_fqname["render"].kind == "function"
    assert by_fqname["render"].container_fqname is None
    assert by_fqname["render"].attributes["binding_style"] == "variable_assignment"
    assert "map" not in {symbol.name for symbol in symbols}


def test_treesitter_parser_recovers_remaining_javascript_binding_symbols() -> None:
    """JavaScript parser should recover alias chains, literal subscripts, and descriptors."""
    source = """
res.header = res.set = function header() {};
var Router = module.exports = function Router() {};
app["all"] = function all() {};
Object.defineProperty(res, "connection", {
  value: function connection() {},
  get: function () {},
  set: function (value) {},
});
var app = exports = module.exports = {};
app.listen = function listen() {};
var api = module.exports = { send() {}, end: () => {} };
const state = { get name() {}, set name(value) {} };
const method = "all";
app[method] = function miss() {};
app["x-y"] = function missBad() {};
"""
    parser = TreeSitterParser("javascript")
    symbols = parser.extract_symbols("sample.js", source)
    grouped = _symbols_grouped_by_fqname(symbols)

    assert {
        "res.header",
        "res.set",
        "Router",
        "module.exports",
        "app.all",
        "res.connection",
        "app.listen",
        "exports.listen",
        "module.exports.listen",
        "api.send",
        "api.end",
        "module.exports.send",
        "module.exports.end",
        "state.name",
    }.issubset(grouped)
    assert "app.miss" not in grouped
    assert "app.x-y" not in grouped

    header = grouped["res.header"][0]
    assert header.kind == "method"
    assert header.name == "header"
    assert header.container_fqname == "res"
    assert header.attributes["binding_style"] == "member_assignment"

    setter_alias = grouped["res.set"][0]
    assert setter_alias.kind == "method"
    assert setter_alias.name == "set"
    assert setter_alias.container_fqname == "res"
    assert setter_alias.attributes["binding_style"] == "member_assignment"

    export_root = grouped["module.exports"][0]
    assert export_root.kind == "function"
    assert export_root.name == "Router"
    assert export_root.container_fqname is None
    assert export_root.attributes["binding_style"] == "export_root_assignment"

    local_alias = grouped["Router"][0]
    assert local_alias.kind == "function"
    assert local_alias.container_fqname is None
    assert local_alias.attributes["binding_style"] == "variable_assignment"

    subscript = grouped["app.all"][0]
    assert subscript.kind == "method"
    assert subscript.name == "all"
    assert subscript.container_fqname == "app"
    assert subscript.attributes["binding_style"] == "member_assignment"

    assert grouped["app.listen"][0].kind == "method"
    assert grouped["app.listen"][0].attributes["binding_style"] == "member_assignment"
    assert grouped["exports.listen"][0].kind == "function"
    assert grouped["exports.listen"][0].attributes["binding_style"] == "export_assignment"
    assert grouped["module.exports.listen"][0].kind == "function"
    assert grouped["module.exports.listen"][0].attributes["binding_style"] == ("export_assignment")

    descriptor_symbols = grouped["res.connection"]
    assert len(descriptor_symbols) == 3
    assert {symbol.attributes["binding_style"] for symbol in descriptor_symbols} == {
        "define_property_descriptor"
    }
    assert {symbol.attributes.get("accessor") for symbol in descriptor_symbols} == {
        None,
        "get",
        "set",
    }

    assert grouped["api.send"][0].container_fqname == "api"
    assert grouped["api.send"][0].attributes["binding_style"] == "object_binding_property"
    assert grouped["module.exports.send"][0].container_fqname == "module.exports"
    assert grouped["api.end"][0].container_fqname == "api"
    assert grouped["module.exports.end"][0].container_fqname == "module.exports"

    accessor_symbols = grouped["state.name"]
    assert len(accessor_symbols) == 2
    assert {symbol.attributes["accessor"] for symbol in accessor_symbols} == {"get", "set"}
    assert {symbol.attributes["binding_style"] for symbol in accessor_symbols} == {
        "object_binding_property"
    }


def test_treesitter_parser_extracts_typescript_assignment_bound_symbols() -> None:
    """TypeScript parser should recover typed arrow assignments and bound object helpers."""
    source = """
const add: (a: number, b: number) => number = (a, b) => a + b;
const api = { json: function (): void {}, end: (): void => {} };
api["load"] = function (): void {};
"""
    parser = TreeSitterParser("typescript")
    symbols = parser.extract_symbols("sample.ts", source)
    by_fqname = _symbols_by_fqname(symbols)

    assert set(by_fqname) == {"add", "api.json", "api.end", "api.load"}
    assert by_fqname["add"].kind == "function"
    assert by_fqname["add"].attributes["binding_style"] == "variable_assignment"
    assert by_fqname["api.json"].kind == "method"
    assert by_fqname["api.json"].container_fqname == "api"
    assert by_fqname["api.end"].attributes["binding_style"] == "object_binding_property"
    assert by_fqname["api.load"].kind == "method"
    assert by_fqname["api.load"].attributes["binding_style"] == "member_assignment"


def test_treesitter_parser_extracts_tsx_assignment_bound_symbols() -> None:
    """TSX parser should recover arrow component and bound object helper symbols."""
    source = """
const Inline = () => <span>x</span>;
const api = { render: () => <div /> };
widgets["Inline"] = () => <aside />;
"""
    parser = TreeSitterParser("tsx")
    symbols = parser.extract_symbols("sample.tsx", source)
    by_fqname = _symbols_by_fqname(symbols)

    assert set(by_fqname) == {"Inline", "api.render", "widgets.Inline"}
    assert by_fqname["Inline"].kind == "function"
    assert by_fqname["Inline"].attributes["binding_style"] == "variable_assignment"
    assert by_fqname["api.render"].kind == "method"
    assert by_fqname["api.render"].container_fqname == "api"
    assert by_fqname["widgets.Inline"].kind == "method"
    assert by_fqname["widgets.Inline"].attributes["binding_style"] == "member_assignment"
