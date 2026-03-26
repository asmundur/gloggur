from __future__ import annotations

import json
import signal
import subprocess

from gloggur.config import GloggurConfig
from gloggur.parsers.registry import ParserRegistry
from gloggur.parsers.support_contract import (
    ParserCheckCase,
    build_language_support_contract,
    run_parser_capability_check,
)


def test_build_language_support_contract_reports_frontend_extension_scope() -> None:
    """Language support contract should reflect default frontend extension scope."""
    contract = build_language_support_contract()

    assert contract["schema_version"] == "1"
    assert ".c" in contract["supported_extensions"]
    assert ".cpp" in contract["supported_extensions"]
    assert ".hpp" in contract["supported_extensions"]
    assert ".jsx" in contract["supported_extensions"]
    assert ".tsx" in contract["supported_extensions"]
    assert ".html" not in contract["supported_extensions"]
    assert ".css" not in contract["supported_extensions"]
    assert "c" in contract["enabled_languages"]
    assert "cpp" in contract["enabled_languages"]
    assert "javascript" in contract["enabled_languages"]
    assert "tsx" in contract["enabled_languages"]


def test_parser_capability_check_returns_required_and_known_gap_counts() -> None:
    """Parser capability check should separate required checks from known-gap checks."""
    config = GloggurConfig()
    payload = run_parser_capability_check(
        parser_registry=ParserRegistry(),
        config=config,
    )

    assert payload["schema_version"] == "1"
    assert payload["required_case_counts"]["total"] > 0
    assert payload["required_case_counts"]["failed"] == 0
    assert payload["known_gap_case_counts"]["total"] > 0
    assert payload["known_gap_case_counts"]["confirmed"] >= 1
    assert payload["language_support_contract"]["schema_version"] == "1"


def test_parser_capability_check_promotes_js_assignment_cases_to_required_passes() -> None:
    """JS-family assignment extraction should be reported as baseline support."""
    payload = run_parser_capability_check(
        parser_registry=ParserRegistry(),
        config=GloggurConfig(),
    )

    contract = payload["language_support_contract"]
    javascript_tiers = contract["construct_tiers"]["javascript"]
    assert javascript_tiers["arrow_function_assignment"] == "baseline"
    assert javascript_tiers["function_expression_assignment"] == "baseline"
    assert javascript_tiers["commonjs_export_assignment"] == "baseline"
    assert javascript_tiers["assignment_alias_chain"] == "baseline"
    assert javascript_tiers["export_root_assignment"] == "baseline"
    assert javascript_tiers["prototype_member_assignment"] == "baseline"
    assert javascript_tiers["string_literal_subscript_assignment"] == "baseline"
    assert javascript_tiers["define_property_descriptor"] == "baseline"
    assert javascript_tiers["object_binding_property"] == "baseline"
    assert javascript_tiers["object_binding_alias_propagation"] == "baseline"
    assert javascript_tiers["computed_identifier_subscript_assignment"] == "known_gap"
    assert javascript_tiers["helper_runtime_mutation"] == "known_gap"
    assert contract["known_gaps"]["javascript"] == [
        (
            "computed identifier subscript assignments such as app[method] = fn are not "
            "extracted as symbols"
        ),
        "helper-driven runtime mutation such as mixin/install helpers is not extracted as symbols",
    ]
    assert (
        contract["construct_tiers"]["typescript"]["typed_arrow_function_assignment"] == "baseline"
    )
    assert contract["construct_tiers"]["tsx"]["arrow_component_assignment"] == "baseline"
    assert contract["construct_tiers"]["c"]["function_definition"] == "baseline"
    assert contract["construct_tiers"]["c"]["function_declaration"] == "baseline"
    assert contract["construct_tiers"]["c"]["struct_union_declaration"] == "baseline"
    assert contract["construct_tiers"]["c"]["enum_declaration"] == "baseline"
    assert contract["construct_tiers"]["c"]["function_pointer_declarator"] == "baseline"
    assert contract["construct_tiers"]["cpp"]["class_struct_declaration"] == "baseline"
    assert contract["construct_tiers"]["cpp"]["class_body_method_definition"] == "baseline"
    assert contract["construct_tiers"]["cpp"]["class_body_method_declaration"] == "baseline"
    assert contract["construct_tiers"]["cpp"]["qualified_method_definition"] == "baseline"
    assert contract["construct_tiers"]["cpp"]["namespace_qualified_container_fqname"] == "baseline"
    assert contract["construct_tiers"]["cpp"]["template_operator_normalization"] == "baseline"
    assert contract["construct_tiers"]["cpp"]["macro_generated_recoverable_patterns"] == "baseline"
    assert contract["construct_tiers"]["cpp"]["macro_generated_complex_forms"] == "known_gap"
    assert contract["known_gaps"]["c"] == []
    assert contract["known_gaps"]["cpp"] == [
        "macro-generated symbols outside strict placeholder patterns are not extracted",
    ]

    cases = {case["id"]: case for case in payload["cases"]}
    assert cases["javascript.arrow_assignment"]["known_gap"] is False
    assert cases["javascript.arrow_assignment"]["status"] == "passed"
    assert cases["javascript.commonjs_and_member_assignments"]["status"] == "passed"
    assert cases["javascript.assignment_alias_chain"]["status"] == "passed"
    assert cases["javascript.export_root_alias_chain"]["status"] == "passed"
    assert cases["javascript.literal_subscript_assignment"]["status"] == "passed"
    assert cases["javascript.define_property_descriptor"]["status"] == "passed"
    assert cases["javascript.object_binding_methods"]["status"] == "passed"
    assert cases["javascript.object_binding_alias_owners"]["status"] == "passed"
    assert cases["typescript.typed_arrow_assignment"]["status"] == "passed"
    assert cases["tsx.arrow_component"]["status"] == "passed"
    assert cases["c.functions_and_types"]["status"] == "passed"
    assert cases["c.callback_returning_function_pointer"]["status"] == "passed"
    assert cases["c.function_pointer_variable_not_callable"]["status"] == "passed"
    assert cases["cpp.class_and_qualified_methods"]["status"] == "passed"
    assert cases["cpp.namespace_qualified_methods"]["status"] == "passed"
    assert cases["cpp.template_and_operator_methods"]["status"] == "passed"
    assert cases["cpp.macro_generated_method_recovery"]["status"] == "passed"


def test_parser_capability_check_reports_native_child_crashes_as_parse_errors(
    monkeypatch,
) -> None:
    """Native parser subprocess crashes should become structured case failures."""
    case = ParserCheckCase(
        case_id="cpp.crash",
        language="cpp",
        path="sample.cpp",
        source="int ping() { return 1; }\n",
        expected_symbols=(("function", "ping"),),
    )

    def _fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=-signal.SIGBUS,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("gloggur.parsers.support_contract.subprocess.run", _fake_run)
    payload = run_parser_capability_check(
        parser_registry=ParserRegistry(),
        config=GloggurConfig(),
        _cases=(case,),
    )

    assert payload["required_case_counts"] == {"total": 1, "passed": 0, "failed": 1}
    result = payload["cases"][0]
    assert result["status"] == "failed"
    assert result["parse_error"] == "child process terminated by signal SIGBUS"


def test_parser_capability_check_preserves_native_language_missing_errors(
    monkeypatch,
) -> None:
    """Native subprocess parse errors should surface as case-level parse_error values."""
    case = ParserCheckCase(
        case_id="cpp.language_missing",
        language="cpp",
        path="sample.cpp",
        source="class Greeter {};\n",
        expected_symbols=(("class", "Greeter"),),
    )

    child_payload = {
        "actual_symbols": [],
        "actual_fqnames": [],
        "parse_error": "LanguageNotFoundError: Language 'cpp' not found",
    }

    def _fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=json.dumps(child_payload),
            stderr="",
        )

    monkeypatch.setattr("gloggur.parsers.support_contract.subprocess.run", _fake_run)
    payload = run_parser_capability_check(
        parser_registry=ParserRegistry(),
        config=GloggurConfig(),
        _cases=(case,),
    )

    assert payload["required_case_counts"] == {"total": 1, "passed": 0, "failed": 1}
    result = payload["cases"][0]
    assert result["status"] == "failed"
    assert result["parse_error"] == "LanguageNotFoundError: Language 'cpp' not found"
