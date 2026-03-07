from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_check_coverage(args: list[str]) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "scripts/check_coverage_baseline.py", "--format", "json", *args]
    return subprocess.run(
        command,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )


def _write_coverage_xml(path: Path, *, overall: float, modules: dict[str, float]) -> None:
    classes = "\n".join(
        f'<class name="{Path(filename).name}" filename="{filename}" line-rate="{line_rate}" branch-rate="0" />'
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


def test_check_coverage_baseline_writes_baseline_file(tmp_path: Path) -> None:
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

    completed = _run_check_coverage(
        [
            "--coverage-file",
            str(coverage_file),
            "--baseline-file",
            str(baseline_file),
            "--write-baseline",
        ]
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 0, f"{completed.stderr}\n{completed.stdout}"
    assert baseline_file.exists()
    assert payload["ok"] is True
    assert payload["written"] is True


def test_check_coverage_baseline_reports_threshold_failure(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.xml"
    baseline_file = tmp_path / "coverage-baseline.json"
    _write_coverage_xml(
        coverage_file,
        overall=0.8,
        modules={
            "bootstrap_launcher.py": 0.67,
            "cli/main.py": 0.83,
            "search/hybrid_search.py": 0.78,
            "storage/metadata_store.py": 0.72,
        },
    )
    baseline_file.write_text(
        json.dumps(
            {
                "overall_percent": 83.0,
                "modules": {
                    "bootstrap_launcher.py": 68.0,
                    "cli/main.py": 84.0,
                    "search/hybrid_search.py": 79.0,
                    "storage/metadata_store.py": 73.0,
                },
            }
        ),
        encoding="utf8",
    )

    completed = _run_check_coverage(
        [
            "--coverage-file",
            str(coverage_file),
            "--baseline-file",
            str(baseline_file),
        ]
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 1, f"{completed.stderr}\n{completed.stdout}"
    assert payload["failure"]["code"] == "coverage_threshold_exceeded"
    assert payload["warnings"]
