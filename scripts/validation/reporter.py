from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TestResult:
    passed: bool
    message: str
    details: Optional[Dict[str, object]] = None


@dataclass
class _Section:
    title: str
    results: List[Dict[str, object]]


class Reporter:
    def __init__(self) -> None:
        self._sections: List[_Section] = []
        self._current: Optional[_Section] = None

    def add_section(self, title: str) -> None:
        section = _Section(title=title, results=[])
        self._sections.append(section)
        self._current = section

    def add_test_result(self, name: str, result: TestResult) -> None:
        if self._current is None:
            self.add_section("General")
        assert self._current is not None
        self._current.results.append({"name": name, "result": result})

    def generate_markdown(self) -> str:
        total, passed, failed = self._summary_counts()
        lines = ["# Validation Report", "", "## Summary", ""]
        lines.append(f"- Total: {total}")
        lines.append(f"- Passed: {passed}")
        lines.append(f"- Failed: {failed}")
        lines.append("")
        for section in self._sections:
            lines.append(f"## {section.title}")
            lines.append("")
            for item in section.results:
                result: TestResult = item["result"]
                icon = "✅" if result.passed else "❌"
                message = f"{icon} {item['name']}: {result.message}"
                lines.append(message)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def generate_json(self) -> Dict[str, object]:
        total, passed, failed = self._summary_counts()
        payload = {
            "summary": {"total": total, "passed": passed, "failed": failed},
            "sections": [],
        }
        for section in self._sections:
            results = []
            for item in section.results:
                result: TestResult = item["result"]
                results.append(
                    {
                        "name": item["name"],
                        "passed": result.passed,
                        "message": result.message,
                        "details": result.details,
                    }
                )
            payload["sections"].append({"title": section.title, "results": results})
        return payload

    def print_summary(self) -> None:
        total, passed, failed = self._summary_counts()
        green = "\033[92m"
        red = "\033[91m"
        yellow = "\033[93m"
        reset = "\033[0m"
        print(f"{yellow}Validation Summary{reset}: total={total} passed={passed} failed={failed}")
        if failed:
            for section in self._sections:
                for item in section.results:
                    result: TestResult = item["result"]
                    if not result.passed:
                        print(f"{red}FAIL{reset} {section.title} - {item['name']}: {result.message}")
        else:
            print(f"{green}All tests passed{reset}")

    def _summary_counts(self) -> tuple[int, int, int]:
        total = 0
        passed = 0
        for section in self._sections:
            for item in section.results:
                total += 1
                result: TestResult = item["result"]
                if result.passed:
                    passed += 1
        failed = total - passed
        return total, passed, failed
