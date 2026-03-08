from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gloggur.parsers.registry import EXTENSION_LANGUAGE, ParserRegistry

if TYPE_CHECKING:
    from gloggur.config import GloggurConfig


LANGUAGE_SUPPORT_CONTRACT_SCHEMA_VERSION = "1"
PARSER_CHECK_SCHEMA_VERSION = "1"


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
        "arrow_function_assignment": "known_gap",
        "function_expression_assignment": "known_gap",
    },
    "typescript": {
        "function_declaration": "baseline",
        "interface_declaration": "baseline",
        "class_declaration": "baseline",
        "type_alias": "known_gap",
        "enum_declaration": "known_gap",
        "typed_arrow_function_assignment": "known_gap",
    },
    "tsx": {
        "function_declaration_component": "baseline",
        "arrow_component_assignment": "known_gap",
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
        "arrow functions assigned to variables are not extracted as symbols",
        "function expressions assigned to variables are not extracted as symbols",
    ],
    "typescript": [
        "type alias declarations are not extracted as symbols",
        "enum declarations are not extracted as symbols",
        "typed arrow functions assigned to variables are not extracted as symbols",
    ],
    "tsx": [
        "arrow function components assigned to variables are not extracted as symbols",
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
        known_gap=True,
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
        known_gap=True,
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


def run_parser_capability_check(
    *,
    parser_registry: ParserRegistry,
    config: GloggurConfig | None = None,
) -> dict[str, Any]:
    """Execute built-in parser capability checks and return structured results."""
    cases: list[dict[str, Any]] = []
    required_passed = 0
    required_failed = 0
    known_gap_confirmed = 0
    known_gap_closed = 0

    for case in _PARSER_CHECK_CASES:
        parser_entry = parser_registry.get_parser_for_path(case.path)
        actual_symbols: set[tuple[str, str]] = set()
        parse_error: str | None = None
        if parser_entry is None:
            parse_error = "parser_unavailable"
        else:
            try:
                extracted = parser_entry.parser.extract_symbols(case.path, case.source)
                actual_symbols = {(symbol.kind, symbol.name) for symbol in extracted}
            except Exception as exc:  # pragma: no cover - defensive envelope
                parse_error = f"{type(exc).__name__}: {exc}"
        missing_symbols = [
            {"kind": kind, "name": name}
            for kind, name in case.expected_symbols
            if (kind, name) not in actual_symbols
        ]
        if case.known_gap:
            if missing_symbols or parse_error is not None:
                status = "gap_confirmed"
                known_gap_confirmed += 1
            else:
                status = "gap_closed"
                known_gap_closed += 1
        else:
            if missing_symbols or parse_error is not None:
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
                "actual_symbol_count": len(actual_symbols),
                "missing_symbols": missing_symbols,
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
