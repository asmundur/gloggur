from __future__ import annotations

import json
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gloggur.parsers.registry import EXTENSION_LANGUAGE, ParserRegistry

if TYPE_CHECKING:
    from gloggur.config import GloggurConfig


LANGUAGE_SUPPORT_CONTRACT_SCHEMA_VERSION = "1"
PARSER_CHECK_SCHEMA_VERSION = "1"
_NATIVE_PARSER_CHECK_LANGUAGES = frozenset({"c", "cpp"})
_PARSER_CHECK_CHILD_COMMAND = (
    "from gloggur.parsers.support_contract import "
    "_run_parser_check_case_child_entrypoint; "
    "raise SystemExit(_run_parser_check_case_child_entrypoint())"
)


_LANGUAGE_CONSTRUCT_TIERS: dict[str, dict[str, str]] = {
    "python": {
        "class_definition": "baseline",
        "function_definition": "baseline",
        "decorated_method_kind": "known_gap",
    },
    "javascript": {
        "function_declaration": "baseline",
        "class_declaration": "baseline",
        "method_definition": "baseline",
        "arrow_function_assignment": "baseline",
        "function_expression_assignment": "baseline",
        "commonjs_export_assignment": "baseline",
        "assignment_alias_chain": "baseline",
        "export_root_assignment": "baseline",
        "prototype_member_assignment": "baseline",
        "string_literal_subscript_assignment": "baseline",
        "define_property_descriptor": "baseline",
        "object_binding_property": "baseline",
        "object_binding_alias_propagation": "baseline",
        "computed_identifier_subscript_assignment": "known_gap",
        "helper_runtime_mutation": "known_gap",
    },
    "typescript": {
        "function_declaration": "baseline",
        "interface_declaration": "baseline",
        "class_declaration": "baseline",
        "type_alias": "known_gap",
        "enum_declaration": "known_gap",
        "typed_arrow_function_assignment": "baseline",
    },
    "tsx": {
        "function_declaration_component": "baseline",
        "arrow_component_assignment": "baseline",
    },
    "c": {
        "function_definition": "baseline",
        "function_declaration": "baseline",
        "struct_union_declaration": "baseline",
        "enum_declaration": "baseline",
        "function_pointer_declarator": "baseline",
    },
    "cpp": {
        "class_struct_declaration": "baseline",
        "enum_declaration": "baseline",
        "class_body_method_definition": "baseline",
        "class_body_method_declaration": "baseline",
        "qualified_method_definition": "baseline",
        "namespace_qualified_container_fqname": "baseline",
        "template_operator_normalization": "baseline",
        "macro_generated_recoverable_patterns": "baseline",
        "macro_generated_complex_forms": "known_gap",
    },
    "go": {
        "function_declaration": "baseline",
        "method_declaration": "baseline",
        "named_struct_type": "known_gap",
        "named_interface_type": "known_gap",
    },
    "rust": {
        "struct_item": "baseline",
        "trait_item": "baseline",
        "impl_method_kind": "known_gap",
        "trait_method_extraction": "known_gap",
    },
    "java": {
        "class_declaration": "baseline",
        "method_declaration": "baseline",
        "record_declaration": "known_gap",
        "enum_declaration": "known_gap",
    },
}

_KNOWN_GAPS_BY_LANGUAGE: dict[str, list[str]] = {
    "python": [
        "decorated methods such as @property may be classified as function instead of method",
    ],
    "javascript": [
        (
            "computed identifier subscript assignments such as app[method] = fn are not "
            "extracted as symbols"
        ),
        "helper-driven runtime mutation such as mixin/install helpers is not extracted as symbols",
    ],
    "typescript": [
        "type alias declarations are not extracted as symbols",
        "enum declarations are not extracted as symbols",
    ],
    "tsx": [],
    "c": [],
    "cpp": [
        "macro-generated symbols outside strict placeholder patterns are not extracted",
    ],
    "go": [
        "named struct/interface type declarations are not extracted as symbols",
        "receiver methods do not include receiver type in fqname",
    ],
    "rust": [
        "impl methods are classified as function instead of method",
        "trait method signatures inside trait bodies are not extracted as symbols",
    ],
    "java": [
        "record declarations are not extracted as symbols",
        "enum declarations are not extracted as symbols",
    ],
}


@dataclass(frozen=True)
class ParserCheckCase:
    """One parser capability check case used by `gloggur parsers check`."""

    case_id: str
    language: str
    path: str
    source: str
    expected_symbols: tuple[tuple[str, str], ...]
    expected_fqnames: tuple[str, ...] = ()
    forbidden_symbols: tuple[tuple[str, str], ...] = ()
    forbidden_fqnames: tuple[str, ...] = ()
    known_gap: bool = False


_PARSER_CHECK_CASES: tuple[ParserCheckCase, ...] = (
    ParserCheckCase(
        case_id="python.function_and_class",
        language="python",
        path="sample.py",
        source=(
            "class Greeter:\n"
            "    def hello(self) -> str:\n"
            "        return 'hi'\n"
            "\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        ),
        expected_symbols=(("class", "Greeter"), ("function", "add")),
    ),
    ParserCheckCase(
        case_id="python.decorated_method_kind",
        language="python",
        path="sample.py",
        source=(
            "class Config:\n"
            "    @property\n"
            "    def name(self) -> str:\n"
            "        return 'x'\n"
        ),
        expected_symbols=(("method", "name"),),
        known_gap=True,
    ),
    ParserCheckCase(
        case_id="javascript.function_declaration",
        language="javascript",
        path="sample.js",
        source=("function main() { return 1; }\n" "class Greeter { hello() { return 'hi'; } }\n"),
        expected_symbols=(("function", "main"), ("class", "Greeter"), ("method", "hello")),
    ),
    ParserCheckCase(
        case_id="javascript.arrow_assignment",
        language="javascript",
        path="sample.js",
        source="const add = (a, b) => a + b;\n",
        expected_symbols=(("function", "add"),),
    ),
    ParserCheckCase(
        case_id="javascript.commonjs_and_member_assignments",
        language="javascript",
        path="sample.js",
        source=(
            "exports.send = function send() {};\n"
            "module.exports.json = function () {};\n"
            "proto.use = function (fn) {};\n"
            "Router.prototype.route = function route(path) {};\n"
        ),
        expected_symbols=(
            ("function", "send"),
            ("function", "json"),
            ("method", "use"),
            ("method", "route"),
        ),
    ),
    ParserCheckCase(
        case_id="javascript.assignment_alias_chain",
        language="javascript",
        path="sample.js",
        source="res.header = res.set = function header() {};\n",
        expected_symbols=(("method", "header"), ("method", "set")),
    ),
    ParserCheckCase(
        case_id="javascript.export_root_alias_chain",
        language="javascript",
        path="sample.js",
        source="var Router = module.exports = function () {};\n",
        expected_symbols=(("function", "Router"), ("function", "module.exports")),
    ),
    ParserCheckCase(
        case_id="javascript.literal_subscript_assignment",
        language="javascript",
        path="sample.js",
        source='app["all"] = function all() {};\n',
        expected_symbols=(("method", "all"),),
    ),
    ParserCheckCase(
        case_id="javascript.define_property_descriptor",
        language="javascript",
        path="sample.js",
        source=(
            'Object.defineProperty(res, "connection", { value: function connection() {} });\n'
            'Object.defineProperty(res, "path", { get: function () {} });\n'
            'Object.defineProperty(res, "host", { set: function (value) {} });\n'
        ),
        expected_symbols=(("method", "connection"), ("method", "path"), ("method", "host")),
    ),
    ParserCheckCase(
        case_id="javascript.object_binding_methods",
        language="javascript",
        path="sample.js",
        source="const api = { send() {}, json: function () {}, end: () => {} };\n",
        expected_symbols=(("method", "send"), ("method", "json"), ("method", "end")),
    ),
    ParserCheckCase(
        case_id="javascript.object_binding_alias_owners",
        language="javascript",
        path="sample.js",
        source="var api = module.exports = { send() {}, end: () => {} };\n",
        expected_symbols=(("method", "send"), ("method", "end")),
    ),
    ParserCheckCase(
        case_id="typescript.interface_and_function",
        language="typescript",
        path="sample.ts",
        source=(
            "interface Service { run(): void }\n"
            "function runTask(name: string): string { return name; }\n"
        ),
        expected_symbols=(("interface", "Service"), ("function", "runTask")),
    ),
    ParserCheckCase(
        case_id="typescript.type_alias",
        language="typescript",
        path="sample.ts",
        source="type User = { id: string };\n",
        expected_symbols=(("type", "User"),),
        known_gap=True,
    ),
    ParserCheckCase(
        case_id="tsx.function_component",
        language="tsx",
        path="sample.tsx",
        source="export function App() { return <div>Hello</div>; }\n",
        expected_symbols=(("function", "App"),),
    ),
    ParserCheckCase(
        case_id="tsx.arrow_component",
        language="tsx",
        path="sample.tsx",
        source="const Inline = () => <span>x</span>;\n",
        expected_symbols=(("function", "Inline"),),
    ),
    ParserCheckCase(
        case_id="typescript.typed_arrow_assignment",
        language="typescript",
        path="sample.ts",
        source="const add: (a: number, b: number) => number = (a, b) => a + b;\n",
        expected_symbols=(("function", "add"),),
    ),
    ParserCheckCase(
        case_id="c.functions_and_types",
        language="c",
        path="sample.c",
        source=(
            "int add(int a, int b) { return a + b; }\n"
            "int declared(int value);\n"
            "struct Point { int x; int y; };\n"
            "enum Mode { Fast, Slow };\n"
        ),
        expected_symbols=(
            ("function", "add"),
            ("function", "declared"),
            ("type", "Point"),
            ("enum", "Mode"),
        ),
        expected_fqnames=("add", "declared", "Point", "Mode"),
    ),
    ParserCheckCase(
        case_id="c.callback_returning_function_pointer",
        language="c",
        path="sample.h",
        source="int (*make_cb(void))(int);\n",
        expected_symbols=(("function", "make_cb"),),
        expected_fqnames=("make_cb",),
    ),
    ParserCheckCase(
        case_id="c.function_pointer_variable_not_callable",
        language="c",
        path="sample.h",
        source="int (*fp)(int);\n",
        expected_symbols=(),
        forbidden_symbols=(("function", "fp"),),
        forbidden_fqnames=("fp",),
    ),
    ParserCheckCase(
        case_id="cpp.class_and_qualified_methods",
        language="cpp",
        path="sample.cpp",
        source=(
            "class Greeter {\n"
            "public:\n"
            "    int hello() const { return 1; }\n"
            "    static int ping();\n"
            "};\n"
            "int Greeter::ping() { return 2; }\n"
        ),
        expected_symbols=(("class", "Greeter"), ("method", "hello"), ("method", "ping")),
        expected_fqnames=("Greeter", "Greeter.hello", "Greeter.ping"),
    ),
    ParserCheckCase(
        case_id="cpp.namespace_qualified_methods",
        language="cpp",
        path="sample.cpp",
        source=(
            "namespace core {\n"
            "class Engine {\n"
            "public:\n"
            "    int start();\n"
            "};\n"
            "}\n"
            "int core::Engine::start() { return 1; }\n"
        ),
        expected_symbols=(("class", "Engine"), ("method", "start")),
        expected_fqnames=("core.Engine", "core.Engine.start"),
    ),
    ParserCheckCase(
        case_id="cpp.template_and_operator_methods",
        language="cpp",
        path="sample.cpp",
        source=(
            "namespace core {\n"
            "template <typename T>\n"
            "class Box {\n"
            "public:\n"
            "    T get() const;\n"
            "};\n"
            "}\n"
            "template <>\n"
            "int core::Box<int>::get() const { return 1; }\n"
            "struct Vec {\n"
            "    int operator[](int idx) const;\n"
            "};\n"
            "int Vec::operator[](int idx) const { return idx; }\n"
        ),
        expected_symbols=(
            ("class", "Box"),
            ("method", "get"),
            ("class", "Vec"),
            ("method", "operator[]"),
        ),
        expected_fqnames=("core.Box", "core.Box.get", "core.Box<int>.get", "Vec.operator[]"),
    ),
    ParserCheckCase(
        case_id="cpp.macro_generated_method_recovery",
        language="cpp",
        path="sample.cpp",
        source=(
            "class Greeter {};\n"
            "#define DECL_METHOD(Type, Name) int Type::Name() { return 0; }\n"
            "DECL_METHOD(Greeter, ping)\n"
        ),
        expected_symbols=(("class", "Greeter"), ("method", "ping")),
        expected_fqnames=("Greeter", "Greeter.ping"),
    ),
    ParserCheckCase(
        case_id="go.function_and_method",
        language="go",
        path="sample.go",
        source=(
            "package main\n"
            "type svc struct{}\n"
            'func (s svc) Run() string { return "x" }\n'
            "func Helper() int { return 1 }\n"
        ),
        expected_symbols=(("method", "Run"), ("function", "Helper")),
    ),
    ParserCheckCase(
        case_id="go.named_interface",
        language="go",
        path="sample.go",
        source=("package main\n" "type Store interface { Get(id string) string }\n"),
        expected_symbols=(("interface", "Store"),),
        known_gap=True,
    ),
    ParserCheckCase(
        case_id="rust.struct_and_trait",
        language="rust",
        path="sample.rs",
        source=("pub struct Client {}\n" "trait Repo { fn fetch(&self); }\n"),
        expected_symbols=(("type", "Client"), ("trait", "Repo")),
    ),
    ParserCheckCase(
        case_id="rust.impl_method_kind",
        language="rust",
        path="sample.rs",
        source=("struct Client;\n" "impl Client {\n" "    fn parse(&self) -> i32 { 1 }\n" "}\n"),
        expected_symbols=(("method", "parse"),),
        known_gap=True,
    ),
    ParserCheckCase(
        case_id="java.class_and_method",
        language="java",
        path="App.java",
        source=("class App {\n" "    void run() {}\n" "}\n"),
        expected_symbols=(("class", "App"), ("method", "run")),
    ),
    ParserCheckCase(
        case_id="java.record_declaration",
        language="java",
        path="App.java",
        source="record User(String id) {}\n",
        expected_symbols=(("type", "User"),),
        known_gap=True,
    ),
)


def _ordered_unique_extensions(values: list[str]) -> list[str]:
    """Return normalized extension list preserving first occurrence order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _normalized_extension_map(values: dict[str, str]) -> dict[str, str]:
    """Return normalized extension->language mapping with deterministic ordering."""
    normalized_items: list[tuple[str, str]] = []
    for extension, language in values.items():
        key = extension.strip().lower()
        if not key.startswith("."):
            key = f".{key}"
        normalized_items.append((key, language))
    normalized_items.sort(key=lambda item: item[0])
    return {key: value for key, value in normalized_items}


def build_language_support_contract(*, config: GloggurConfig | None = None) -> dict[str, Any]:
    """Build the machine-readable language support contract payload."""
    supported_extensions = (
        _ordered_unique_extensions(list(config.supported_extensions))
        if config is not None
        else _ordered_unique_extensions(list(EXTENSION_LANGUAGE))
    )
    parser_extension_map = (
        _normalized_extension_map(dict(config.parser_extension_map)) if config is not None else {}
    )
    effective_extension_map = ParserRegistry(
        extension_map=parser_extension_map,
    ).supported_extensions()
    unmapped_supported_extensions = sorted(
        extension for extension in supported_extensions if extension not in effective_extension_map
    )
    enabled_languages = sorted(
        {
            language
            for extension, language in effective_extension_map.items()
            if extension in supported_extensions
        }
    )
    return {
        "schema_version": LANGUAGE_SUPPORT_CONTRACT_SCHEMA_VERSION,
        "supported_extensions": supported_extensions,
        "effective_extension_map": {
            key: value
            for key, value in sorted(
                effective_extension_map.items(),
                key=lambda item: item[0],
            )
        },
        "parser_extension_overrides": parser_extension_map,
        "enabled_languages": enabled_languages,
        "unmapped_supported_extensions": unmapped_supported_extensions,
        "construct_tiers": _LANGUAGE_CONSTRUCT_TIERS,
        "known_gaps": _KNOWN_GAPS_BY_LANGUAGE,
    }


def _parser_check_case_to_payload(case: ParserCheckCase) -> dict[str, Any]:
    """Serialize a parser capability case for subprocess transport."""
    return {
        "case_id": case.case_id,
        "language": case.language,
        "path": case.path,
        "source": case.source,
        "expected_symbols": [[kind, name] for kind, name in case.expected_symbols],
        "expected_fqnames": list(case.expected_fqnames),
        "forbidden_symbols": [[kind, name] for kind, name in case.forbidden_symbols],
        "forbidden_fqnames": list(case.forbidden_fqnames),
        "known_gap": case.known_gap,
    }


def _parser_check_case_from_payload(payload: dict[str, Any]) -> ParserCheckCase:
    """Deserialize a subprocess parser capability case payload."""
    return ParserCheckCase(
        case_id=str(payload["case_id"]),
        language=str(payload["language"]),
        path=str(payload["path"]),
        source=str(payload["source"]),
        expected_symbols=tuple(
            (str(kind), str(name)) for kind, name in payload.get("expected_symbols", [])
        ),
        expected_fqnames=tuple(str(value) for value in payload.get("expected_fqnames", [])),
        forbidden_symbols=tuple(
            (str(kind), str(name)) for kind, name in payload.get("forbidden_symbols", [])
        ),
        forbidden_fqnames=tuple(str(value) for value in payload.get("forbidden_fqnames", [])),
        known_gap=bool(payload.get("known_gap", False)),
    )


def _collect_parser_case_observation(
    case: ParserCheckCase,
    *,
    parser_registry: ParserRegistry,
) -> tuple[set[tuple[str, str]], set[str], str | None]:
    """Execute one parser case in-process and return observed symbols/fqnames/errors."""
    parser_entry = parser_registry.get_parser_for_path(case.path)
    actual_symbols: set[tuple[str, str]] = set()
    actual_fqnames: set[str] = set()
    parse_error: str | None = None
    if parser_entry is None:
        parse_error = "parser_unavailable"
    else:
        try:
            extracted = parser_entry.parser.extract_symbols(case.path, case.source)
            actual_symbols = {(symbol.kind, symbol.name) for symbol in extracted}
            actual_fqnames = {symbol.fqname or symbol.name for symbol in extracted}
        except Exception as exc:  # pragma: no cover - defensive envelope
            parse_error = f"{type(exc).__name__}: {exc}"
    return actual_symbols, actual_fqnames, parse_error


def _parser_check_observation_to_payload(
    actual_symbols: set[tuple[str, str]],
    actual_fqnames: set[str],
    parse_error: str | None,
) -> dict[str, Any]:
    """Serialize parser-case observations for subprocess transport."""
    return {
        "actual_symbols": sorted([list(symbol) for symbol in actual_symbols]),
        "actual_fqnames": sorted(actual_fqnames),
        "parse_error": parse_error,
    }


def _parser_check_observation_from_payload(
    payload: dict[str, Any],
) -> tuple[set[tuple[str, str]], set[str], str | None]:
    """Deserialize parser-case observations from subprocess JSON."""
    actual_symbols = {(str(kind), str(name)) for kind, name in payload.get("actual_symbols", [])}
    actual_fqnames = {str(value) for value in payload.get("actual_fqnames", [])}
    parse_error = payload.get("parse_error")
    return actual_symbols, actual_fqnames, None if parse_error is None else str(parse_error)


def _run_parser_check_case_child_entrypoint() -> int:
    """Run one parser capability case in a fresh Python process."""
    payload = json.loads(sys.stdin.read())
    case = _parser_check_case_from_payload(payload)
    observation = _collect_parser_case_observation(case, parser_registry=ParserRegistry())
    sys.stdout.write(json.dumps(_parser_check_observation_to_payload(*observation), sort_keys=True))
    return 0


def _format_parser_check_subprocess_failure(completed: subprocess.CompletedProcess[str]) -> str:
    """Describe a non-zero subprocess result for parser capability checks."""
    if completed.returncode < 0:
        signal_number = -completed.returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:  # pragma: no cover - defensive envelope
            signal_name = f"SIG{signal_number}"
        detail = f"child process terminated by signal {signal_name}"
    else:
        detail = f"child process exited with code {completed.returncode}"
    stderr = completed.stderr.strip()
    if stderr:
        detail = f"{detail}: {stderr.splitlines()[-1]}"
    return detail


def _run_parser_check_case_in_subprocess(
    case: ParserCheckCase,
) -> tuple[set[tuple[str, str]], set[str], str | None]:
    """Run one native parser capability case in a child Python process."""
    completed = subprocess.run(
        [sys.executable, "-c", _PARSER_CHECK_CHILD_COMMAND],
        input=json.dumps(_parser_check_case_to_payload(case)),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return set(), set(), _format_parser_check_subprocess_failure(completed)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = f"child process returned invalid JSON: {exc.msg}"
        stdout = completed.stdout.strip()
        if stdout:
            detail = f"{detail}: {stdout.splitlines()[-1]}"
        return set(), set(), detail
    return _parser_check_observation_from_payload(payload)


def run_parser_capability_check(
    *,
    parser_registry: ParserRegistry,
    config: GloggurConfig | None = None,
    _cases: tuple[ParserCheckCase, ...] | None = None,
) -> dict[str, Any]:
    """Execute built-in parser capability checks and return structured results."""
    cases: list[dict[str, Any]] = []
    required_passed = 0
    required_failed = 0
    known_gap_confirmed = 0
    known_gap_closed = 0

    parser_cases = _PARSER_CHECK_CASES if _cases is None else _cases

    for case in parser_cases:
        if case.language in _NATIVE_PARSER_CHECK_LANGUAGES:
            actual_symbols, actual_fqnames, parse_error = _run_parser_check_case_in_subprocess(case)
        else:
            actual_symbols, actual_fqnames, parse_error = _collect_parser_case_observation(
                case,
                parser_registry=parser_registry,
            )
        missing_symbols = [
            {"kind": kind, "name": name}
            for kind, name in case.expected_symbols
            if (kind, name) not in actual_symbols
        ]
        unexpected_symbols = [
            {"kind": kind, "name": name}
            for kind, name in case.forbidden_symbols
            if (kind, name) in actual_symbols
        ]
        missing_fqnames = [
            fqname for fqname in case.expected_fqnames if fqname not in actual_fqnames
        ]
        unexpected_fqnames = [
            fqname for fqname in case.forbidden_fqnames if fqname in actual_fqnames
        ]
        has_failures = bool(
            missing_symbols or missing_fqnames or unexpected_symbols or unexpected_fqnames
        )
        if case.known_gap:
            if has_failures or parse_error is not None:
                status = "gap_confirmed"
                known_gap_confirmed += 1
            else:
                status = "gap_closed"
                known_gap_closed += 1
        else:
            if has_failures or parse_error is not None:
                status = "failed"
                required_failed += 1
            else:
                status = "passed"
                required_passed += 1
        cases.append(
            {
                "id": case.case_id,
                "language": case.language,
                "path": case.path,
                "known_gap": case.known_gap,
                "status": status,
                "expected_symbols": [
                    {"kind": kind, "name": name} for kind, name in case.expected_symbols
                ],
                "expected_fqnames": list(case.expected_fqnames),
                "forbidden_symbols": [
                    {"kind": kind, "name": name} for kind, name in case.forbidden_symbols
                ],
                "forbidden_fqnames": list(case.forbidden_fqnames),
                "actual_symbol_count": len(actual_symbols),
                "missing_symbols": missing_symbols,
                "unexpected_symbols": unexpected_symbols,
                "missing_fqnames": missing_fqnames,
                "unexpected_fqnames": unexpected_fqnames,
                "parse_error": parse_error,
            }
        )

    return {
        "schema_version": PARSER_CHECK_SCHEMA_VERSION,
        "ok": required_failed == 0,
        "required_case_counts": {
            "total": required_passed + required_failed,
            "passed": required_passed,
            "failed": required_failed,
        },
        "known_gap_case_counts": {
            "total": known_gap_confirmed + known_gap_closed,
            "confirmed": known_gap_confirmed,
            "closed": known_gap_closed,
        },
        "cases": cases,
        "language_support_contract": build_language_support_contract(config=config),
    }
