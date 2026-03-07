from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

DEFAULT_COVERAGE_FILE = Path("coverage.xml")
DEFAULT_BASELINE_FILE = Path("benchmarks/coverage_baseline.json")
PROTECTED_MODULES = (
    "bootstrap_launcher.py",
    "cli/main.py",
    "search/hybrid_search.py",
    "storage/metadata_store.py",
)


def _percent(value: str | float | int) -> float:
    return round(float(value) * 100, 2)


def _require_percent(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _require_module_thresholds(payload: dict[str, object], key: str) -> dict[str, float]:
    raw_modules = payload.get(key)
    if not isinstance(raw_modules, dict):
        raise ValueError(f"{key} must be a modules object")
    normalized: dict[str, float] = {}
    for raw_name, raw_value in raw_modules.items():
        if not isinstance(raw_value, (int, float)):
            raise ValueError(f"{key} values must be numeric")
        normalized[str(raw_name)] = float(raw_value)
    return normalized


def parse_coverage_report(coverage_file: Path) -> dict[str, object]:
    if not coverage_file.exists():
        raise FileNotFoundError(str(coverage_file))
    root = ElementTree.fromstring(coverage_file.read_text(encoding="utf8"))
    modules: dict[str, float] = {}
    for class_node in root.findall(".//class"):
        filename = class_node.attrib.get("filename")
        line_rate = class_node.attrib.get("line-rate")
        if not filename or line_rate is None:
            continue
        modules[str(filename)] = _percent(line_rate)
    return {
        "overall_percent": _percent(root.attrib.get("line-rate", 0.0)),
        "modules": {module: modules[module] for module in PROTECTED_MODULES if module in modules},
    }


def build_baseline_payload(summary: dict[str, object], coverage_file: Path) -> dict[str, object]:
    modules = _require_module_thresholds(summary, "modules")
    return {
        "metric": "line_coverage_percent",
        "coverage_file": str(coverage_file),
        "overall_percent": _require_percent(summary, "overall_percent"),
        "modules": modules,
    }


def apply_baseline_contract(
    summary: dict[str, object],
    baseline_payload: dict[str, object],
) -> dict[str, object]:
    baseline_modules = _require_module_thresholds(baseline_payload, "modules")
    summary_modules = _require_module_thresholds(summary, "modules")
    actual_overall = _require_percent(summary, "overall_percent")
    minimum_overall = _require_percent(baseline_payload, "overall_percent")

    comparisons: dict[str, object] = {
        "overall": {
            "actual_percent": actual_overall,
            "minimum_percent": minimum_overall,
            "status": "pass",
        },
        "modules": {},
    }
    warnings: list[str] = []

    if actual_overall < minimum_overall:
        overall_comparison = comparisons["overall"]
        assert isinstance(overall_comparison, dict)
        overall_comparison["status"] = "regressed"
        warnings.append("overall coverage regressed " f"({actual_overall} < {minimum_overall})")

    module_comparisons: dict[str, object] = {}
    for module_path, minimum_percent in baseline_modules.items():
        actual_percent = summary_modules.get(module_path)
        status = "pass"
        if not isinstance(actual_percent, (int, float)):
            status = "regressed"
            warnings.append(f"protected module coverage missing for {module_path}")
            actual_value = None
        else:
            actual_value = float(actual_percent)
            if actual_value < float(minimum_percent):
                status = "regressed"
                warnings.append(
                    f"{module_path} regressed ({actual_value} < {float(minimum_percent)})"
                )
        module_comparisons[str(module_path)] = {
            "actual_percent": actual_value,
            "minimum_percent": float(minimum_percent),
            "status": status,
        }
    comparisons["modules"] = module_comparisons
    return {"comparisons": comparisons, "warnings": warnings}


def check_coverage_baseline(
    coverage_file: Path,
    baseline_file: Path,
    *,
    write_baseline: bool = False,
) -> dict[str, object]:
    try:
        summary = parse_coverage_report(coverage_file)
    except FileNotFoundError:
        return {
            "ok": False,
            "failure": {
                "code": "coverage_report_missing",
                "detail": f"Coverage report does not exist: {coverage_file}",
                "remediation": "Run pytest with coverage enabled and rerun the baseline check.",
            },
        }

    payload: dict[str, object] = {
        "ok": True,
        "coverage_file": str(coverage_file),
        "baseline_file": str(baseline_file),
        "coverage": summary,
    }

    if write_baseline:
        baseline_payload = build_baseline_payload(summary, coverage_file)
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        baseline_file.write_text(
            json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n", encoding="utf8"
        )
        payload["baseline"] = baseline_payload
        payload["written"] = True
        return payload

    if not baseline_file.exists():
        payload["ok"] = False
        payload["failure"] = {
            "code": "coverage_baseline_invalid",
            "detail": f"Coverage baseline file does not exist: {baseline_file}",
            "remediation": (
                "Create or restore the checked-in coverage baseline and rerun the check."
            ),
        }
        return payload

    try:
        baseline_payload = json.loads(baseline_file.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload["ok"] = False
        payload["failure"] = {
            "code": "coverage_baseline_invalid",
            "detail": f"Coverage baseline is unreadable: {exc}",
            "remediation": "Fix the baseline JSON file and rerun the check.",
        }
        return payload
    if not isinstance(baseline_payload, dict):
        payload["ok"] = False
        payload["failure"] = {
            "code": "coverage_baseline_invalid",
            "detail": "Coverage baseline payload must be a JSON object.",
            "remediation": "Replace the baseline file with a valid baseline JSON object.",
        }
        return payload

    try:
        contract = apply_baseline_contract(summary, baseline_payload)
    except (KeyError, TypeError, ValueError) as exc:
        payload["ok"] = False
        payload["failure"] = {
            "code": "coverage_baseline_invalid",
            "detail": f"Coverage baseline payload is invalid: {exc}",
            "remediation": "Regenerate or repair the baseline JSON payload and rerun the check.",
        }
        return payload

    payload["baseline"] = baseline_payload
    payload["comparisons"] = contract["comparisons"]
    payload["warnings"] = contract["warnings"]
    warnings = contract["warnings"]
    if isinstance(warnings, list) and warnings:
        payload["ok"] = False
        payload["failure"] = {
            "code": "coverage_threshold_exceeded",
            "detail": "Coverage dropped below the protected baseline.",
            "remediation": (
                "Restore overall and protected-module coverage to at least the checked-in "
                "baseline."
            ),
            "warnings": warnings,
        }
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate coverage.xml against a checked-in baseline."
    )
    parser.add_argument("--coverage-file", type=Path, default=DEFAULT_COVERAGE_FILE)
    parser.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE_FILE)
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument("--write-baseline", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = check_coverage_baseline(
        args.coverage_file,
        args.baseline_file,
        write_baseline=bool(args.write_baseline),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
