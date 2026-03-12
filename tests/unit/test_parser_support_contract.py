from __future__ import annotations

from gloggur.config import GloggurConfig
from gloggur.parsers.registry import ParserRegistry
from gloggur.parsers.support_contract import (
    build_language_support_contract,
    run_parser_capability_check,
)


def test_build_language_support_contract_reports_frontend_extension_scope() -> None:
    """Language support contract should reflect default frontend extension scope."""
    contract = build_language_support_contract()

    assert contract["schema_version"] == "1"
    assert ".jsx" in contract["supported_extensions"]
    assert ".tsx" in contract["supported_extensions"]
    assert ".html" not in contract["supported_extensions"]
    assert ".css" not in contract["supported_extensions"]
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
    assert javascript_tiers["prototype_member_assignment"] == "baseline"
    assert javascript_tiers["object_binding_property"] == "baseline"
    assert contract["known_gaps"]["javascript"] == []
    assert (
        contract["construct_tiers"]["typescript"]["typed_arrow_function_assignment"] == "baseline"
    )
    assert contract["construct_tiers"]["tsx"]["arrow_component_assignment"] == "baseline"

    cases = {case["id"]: case for case in payload["cases"]}
    assert cases["javascript.arrow_assignment"]["known_gap"] is False
    assert cases["javascript.arrow_assignment"]["status"] == "passed"
    assert cases["javascript.commonjs_and_member_assignments"]["status"] == "passed"
    assert cases["javascript.object_binding_methods"]["status"] == "passed"
    assert cases["typescript.typed_arrow_assignment"]["status"] == "passed"
    assert cases["tsx.arrow_component"]["status"] == "passed"
