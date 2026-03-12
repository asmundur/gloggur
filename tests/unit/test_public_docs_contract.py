from __future__ import annotations

from pathlib import Path


def test_readme_and_agent_guide_do_not_advertise_removed_grounding_flags() -> None:
    """Public docs should avoid recommending removed search grounding flags."""
    readme_text = Path("README.md").read_text(encoding="utf8")
    agent_text = Path("docs/AGENT_INTEGRATION.md").read_text(encoding="utf8")

    removed_flags = (
        "--with-evidence-trace",
        "--validate-grounding",
        "--evidence-min-confidence",
        "--evidence-min-items",
        "--fail-on-ungrounded",
    )

    for flag in removed_flags:
        assert flag not in readme_text, f"README still advertises removed flag {flag}"

    assert "Grounded retrieve -> validate -> emit/repair flow" not in agent_text
    assert "were removed" in agent_text
    assert "search_contract_v1_removed" in agent_text


def test_readme_documents_bidirectional_graph_neighbors_and_parser_gaps() -> None:
    """README should describe current graph defaults and baseline parser support honestly."""
    text = Path("README.md").read_text(encoding="utf8")

    assert "bidirectional by default" in text
    assert "structural metadata by default" in text
    assert "--embed-graph-edges" in text
    assert "get all outgoing edges for a symbol" not in text
    assert "Language support is baseline, not uniform." in text
    assert "gloggur parsers check --json" in text


def test_quickstart_documents_current_operator_gotchas() -> None:
    """Quickstart should call out inspect scope, graph-edge defaults, repo-config warnings, and test-provider scope."""
    text = Path("docs/QUICKSTART.md").read_text(encoding="utf8")

    assert "--embed-graph-edges" in text
    assert "embedded_edge_vectors" in text
    assert "focuses on source paths by default" in text
    assert "--include-tests" in text
    assert "--include-scripts" in text
    assert "security_warning_codes" in text
    assert "untrusted_repo_config" in text
    assert "GLOGGUR_EMBEDDING_PROVIDER=test" in text
    assert "local model bootstrap" in text


def test_verification_docs_and_harness_expose_test_provider_scope() -> None:
    """Verification docs and the smoke harness should agree on deterministic provider scope."""
    verification_text = Path("docs/VERIFICATION.md").read_text(encoding="utf8")
    harness_text = Path("scripts/run_quickstart_smoke.py").read_text(encoding="utf8")

    assert "GLOGGUR_EMBEDDING_PROVIDER=test" in verification_text
    assert "local-provider bootstrap" in verification_text
    assert 'env["GLOGGUR_EMBEDDING_PROVIDER"] = "test"' in harness_text
    assert '"embedding_provider": "test"' in harness_text
    assert '"local_provider_bootstrap_validated": False' in harness_text
