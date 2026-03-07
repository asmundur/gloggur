from __future__ import annotations

import json
from pathlib import Path

from scripts.check_coverage_baseline import (
    apply_baseline_contract,
    build_baseline_payload,
    check_coverage_baseline,
    parse_coverage_report,
)


def _write_coverage_xml(path: Path, *, overall: float, modules: dict[str, float]) -> None:
    classes = "\n".join(
        (
            f'<class name="{Path(filename).name}" filename="{filename}" '
            f'line-rate="{line_rate}" branch-rate="0" />'
        )
        for filename, line_rate in modules.items()
    )
    path.write_text(
        (
            '<?xml version="1.0" ?>\n'
            f'<coverage line-rate="{overall}" lines-valid="10" lines-covered="8">\n'
            "  <packages>\n"
            '    <package name="." line-rate="0">\n'
            "      <classes>\n"
            f"        {classes}\n"
            "      </classes>\n"
            "    </package>\n"
            "  </packages>\n"
            "</coverage>\n"
        ),
        encoding="utf8",
    )


def test_parse_coverage_report_extracts_overall_and_protected_modules(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.xml"
    _write_coverage_xml(
        coverage_file,
        overall=0.834,
        modules={
            "bootstrap_launcher.py": 0.684,
            "cli/main.py": 0.842,
            "search/hybrid_search.py": 0.791,
            "storage/metadata_store.py": 0.739,
            "ignored.py": 0.999,
        },
    )

    summary = parse_coverage_report(coverage_file)

    assert summary["overall_percent"] == 83.4
    assert summary["modules"] == {
        "bootstrap_launcher.py": 68.4,
        "cli/main.py": 84.2,
        "search/hybrid_search.py": 79.1,
        "storage/metadata_store.py": 73.9,
    }


def test_apply_baseline_contract_reports_regressions_and_missing_modules() -> None:
    contract = apply_baseline_contract(
        {
            "overall_percent": 82.7,
            "modules": {
                "bootstrap_launcher.py": 67.9,
                "cli/main.py": 84.1,
                "search/hybrid_search.py": 79.0,
            },
        },
        {
            "overall_percent": 83.0,
            "modules": {
                "bootstrap_launcher.py": 68.0,
                "cli/main.py": 84.0,
                "search/hybrid_search.py": 79.0,
                "storage/metadata_store.py": 73.0,
            },
        },
    )

    assert contract["comparisons"]["overall"]["status"] == "regressed"
    module_comparisons = contract["comparisons"]["modules"]
    assert module_comparisons["bootstrap_launcher.py"]["status"] == "regressed"
    assert module_comparisons["cli/main.py"]["status"] == "pass"
    assert module_comparisons["storage/metadata_store.py"]["actual_percent"] is None
    assert "overall coverage regressed" in contract["warnings"][0]


def test_check_coverage_baseline_writes_and_validates_baseline(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.xml"
    baseline_file = tmp_path / "coverage-baseline.json"
    _write_coverage_xml(
        coverage_file,
        overall=0.84,
        modules={
            "bootstrap_launcher.py": 0.7,
            "cli/main.py": 0.85,
            "search/hybrid_search.py": 0.8,
            "storage/metadata_store.py": 0.74,
        },
    )

    written = check_coverage_baseline(coverage_file, baseline_file, write_baseline=True)
    checked = check_coverage_baseline(coverage_file, baseline_file)

    assert written["ok"] is True
    assert written["written"] is True
    assert baseline_file.exists()
    assert checked["ok"] is True
    assert checked["comparisons"]["overall"]["status"] == "pass"
    assert build_baseline_payload(
        parse_coverage_report(coverage_file), coverage_file
    ) == json.loads(baseline_file.read_text(encoding="utf8"))


def test_check_coverage_baseline_reports_invalid_baseline_and_missing_report(
    tmp_path: Path,
) -> None:
    missing_report = check_coverage_baseline(tmp_path / "missing.xml", tmp_path / "baseline.json")
    coverage_file = tmp_path / "coverage.xml"
    baseline_file = tmp_path / "baseline.json"
    _write_coverage_xml(
        coverage_file,
        overall=0.83,
        modules={
            "bootstrap_launcher.py": 0.68,
            "cli/main.py": 0.84,
            "search/hybrid_search.py": 0.79,
            "storage/metadata_store.py": 0.73,
        },
    )
    baseline_file.write_text("{not-json", encoding="utf8")
    invalid = check_coverage_baseline(coverage_file, baseline_file)

    assert missing_report["failure"]["code"] == "coverage_report_missing"
    assert invalid["failure"]["code"] == "coverage_baseline_invalid"
