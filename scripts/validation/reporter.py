from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Dict, List, Optional


@dataclass
class TestResult:
    __test__ = False

    passed: bool
    message: str
    details: Optional[Dict[str, object]] = None


@dataclass
class PerformanceMetric:
    name: str
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    throughput: Optional[float] = None
    throughput_unit: Optional[str] = None


@dataclass
class PerformanceThresholds:
    max_duration_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    min_throughput: Optional[float] = None


@dataclass
class PerformanceTrendPoint:
    label: str
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    throughput: Optional[float] = None
    throughput_unit: Optional[str] = None


@dataclass
class _Section:
    title: str
    results: List[Dict[str, object]]


class Reporter:
    def __init__(self) -> None:
        self._sections: List[_Section] = []
        self._current: Optional[_Section] = None
        self._performance: Dict[str, PerformanceMetric] = {}
        self._performance_thresholds: Dict[str, PerformanceThresholds] = {}
        self._performance_warnings: List[Dict[str, str]] = []
        self._baseline: Dict[str, PerformanceMetric] = {}
        self._performance_trends: Dict[str, List[PerformanceTrendPoint]] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

    def add_section(self, title: str) -> None:
        with self._lock:
            section = _Section(title=title, results=[])
            self._sections.append(section)
            self._current = section
            self._logger.debug("Added report section: %s", title)

    def add_test_result(self, name: str, result: TestResult) -> None:
        with self._lock:
            if self._current is None:
                self.add_section("General")
            assert self._current is not None
            self._current.results.append({"name": name, "result": result})
            self._logger.info("Recorded test result: %s (passed=%s)", name, result.passed)

    def add_test_result_to_section(self, section_title: str, name: str, result: TestResult) -> None:
        with self._lock:
            section = next((item for item in self._sections if item.title == section_title), None)
            if section is None:
                section = _Section(title=section_title, results=[])
                self._sections.append(section)
            section.results.append({"name": name, "result": result})
            self._logger.info("Recorded test result: %s (section=%s)", name, section_title)

    def add_performance_metric(
        self,
        name: str,
        *,
        duration_ms: Optional[float] = None,
        memory_mb: Optional[float] = None,
        throughput: Optional[float] = None,
        throughput_unit: Optional[str] = None,
        thresholds: Optional[PerformanceThresholds] = None,
    ) -> None:
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                throughput=throughput,
                throughput_unit=throughput_unit,
            )
            self._performance[name] = metric
            if thresholds:
                self._performance_thresholds[name] = thresholds
            self._evaluate_thresholds(name, metric)
            self._logger.debug("Added performance metric: %s", name)

    def set_performance_thresholds(
        self,
        name: str,
        *,
        max_duration_ms: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        min_throughput: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._performance_thresholds[name] = PerformanceThresholds(
                max_duration_ms=max_duration_ms,
                max_memory_mb=max_memory_mb,
                min_throughput=min_throughput,
            )
            metric = self._performance.get(name)
            if metric:
                self._evaluate_thresholds(name, metric)

    def add_performance_trend_point(
        self,
        name: str,
        label: str,
        *,
        duration_ms: Optional[float] = None,
        memory_mb: Optional[float] = None,
        throughput: Optional[float] = None,
        throughput_unit: Optional[str] = None,
    ) -> None:
        with self._lock:
            points = self._performance_trends.setdefault(name, [])
            points.append(
                PerformanceTrendPoint(
                    label=label,
                    duration_ms=duration_ms,
                    memory_mb=memory_mb,
                    throughput=throughput,
                    throughput_unit=throughput_unit,
                )
            )

    def set_baseline_metrics(self, baseline: Dict[str, PerformanceMetric]) -> None:
        with self._lock:
            self._baseline = dict(baseline)

    def load_baseline_from_payload(self, payload: Dict[str, object]) -> None:
        with self._lock:
            if not isinstance(payload, dict):
                return
            performance = payload.get("performance")
            metrics = None
            if isinstance(performance, dict):
                metrics = performance.get("metrics")
            if metrics is None and isinstance(payload.get("metrics"), dict):
                metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                return
            baseline: Dict[str, PerformanceMetric] = {}
            for name, data in metrics.items():
                if not isinstance(data, dict):
                    continue
                baseline[name] = PerformanceMetric(
                    name=name,
                    duration_ms=_coerce_float(data.get("duration_ms")),
                    memory_mb=_coerce_float(data.get("memory_mb")),
                    throughput=_coerce_float(data.get("throughput")),
                    throughput_unit=_coerce_str(data.get("throughput_unit")),
                )
            if baseline:
                self._baseline = baseline

    def add_baseline_trends(self, baseline_label: str = "baseline", current_label: str = "current") -> None:
        with self._lock:
            if not self._baseline:
                return
            for name, current in self._performance.items():
                baseline = self._baseline.get(name)
                if not baseline:
                    continue
                self.add_performance_trend_point(
                    name,
                    baseline_label,
                    duration_ms=baseline.duration_ms,
                    memory_mb=baseline.memory_mb,
                    throughput=baseline.throughput,
                    throughput_unit=baseline.throughput_unit,
                )
                self.add_performance_trend_point(
                    name,
                    current_label,
                    duration_ms=current.duration_ms,
                    memory_mb=current.memory_mb,
                    throughput=current.throughput,
                    throughput_unit=current.throughput_unit,
                )

    def generate_markdown(self) -> str:
        with self._lock:
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
            performance_section = self.render_performance_markdown()
            if performance_section:
                lines.append(performance_section)
            return "\n".join(lines).strip() + "\n"

    def render_performance_markdown(self) -> str:
        with self._lock:
            if not self._performance:
                return ""
            lines: List[str] = []
            lines.append("## Performance")
            lines.append("")
            for name, metric in sorted(self._performance.items()):
                lines.append(f"### {name}")
                lines.append("")
                metric_lines = _format_metric_lines(metric)
                if metric_lines:
                    lines.extend(metric_lines)
                comparison_lines = self._format_comparison_lines(name, metric)
                if comparison_lines:
                    lines.append("")
                    lines.append("Comparison to baseline:")
                    lines.extend(comparison_lines)
                lines.append("")
            if self._performance_warnings:
                lines.append("### Performance Warnings")
                lines.append("")
                for warning in self._performance_warnings:
                    lines.append(f"- {warning['message']}")
                lines.append("")
            trend_lines = self._render_trend_charts()
            if trend_lines:
                lines.append("### Performance Trends")
                lines.append("")
                lines.extend(trend_lines)
                lines.append("")
            return "\n".join(lines).strip()

    def generate_json(self) -> Dict[str, object]:
        with self._lock:
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
            if self._performance:
                payload["performance"] = {
                    "metrics": {name: _metric_payload(metric) for name, metric in self._performance.items()},
                    "thresholds": {
                        name: _threshold_payload(thresholds)
                        for name, thresholds in self._performance_thresholds.items()
                    },
                    "warnings": list(self._performance_warnings),
                    "baseline": {name: _metric_payload(metric) for name, metric in self._baseline.items()},
                    "comparisons": self._build_comparisons_payload(),
                    "trends": {
                        name: [_trend_payload(point) for point in points]
                        for name, points in self._performance_trends.items()
                    },
                }
            return payload

    def print_summary(self) -> None:
        with self._lock:
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
        with self._lock:
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

    def _evaluate_thresholds(self, name: str, metric: PerformanceMetric) -> None:
        thresholds = self._performance_thresholds.get(name)
        if not thresholds:
            return
        self._performance_warnings = [item for item in self._performance_warnings if item.get("metric") != name]
        if metric.duration_ms is not None and thresholds.max_duration_ms is not None:
            if metric.duration_ms > thresholds.max_duration_ms:
                current = _format_duration_ms(metric.duration_ms)
                expected = _format_duration_ms(thresholds.max_duration_ms)
                self._performance_warnings.append(
                    {"metric": name, "message": f"{name} took {current}, expected <{expected}"}
                )
        if metric.memory_mb is not None and thresholds.max_memory_mb is not None:
            if metric.memory_mb > thresholds.max_memory_mb:
                current = f"{metric.memory_mb:.2f} MB"
                expected = f"{thresholds.max_memory_mb:.2f} MB"
                self._performance_warnings.append(
                    {"metric": name, "message": f"{name} used {current}, expected <{expected}"}
                )
        if metric.throughput is not None and thresholds.min_throughput is not None:
            if metric.throughput < thresholds.min_throughput:
                unit = metric.throughput_unit or "items/s"
                current = f"{metric.throughput:.2f} {unit}"
                expected = f"{thresholds.min_throughput:.2f} {unit}"
                self._performance_warnings.append(
                    {"metric": name, "message": f"{name} throughput {current}, expected >{expected}"}
                )

    def _format_comparison_lines(self, name: str, metric: PerformanceMetric) -> List[str]:
        baseline = self._baseline.get(name)
        if not baseline:
            return []
        lines: List[str] = []
        duration_line = _format_comparison_line(
            "Duration",
            metric.duration_ms,
            baseline.duration_ms,
            higher_is_better=False,
            formatter=_format_duration_ms,
        )
        if duration_line:
            lines.append(f"- {duration_line}")
        memory_line = _format_comparison_line(
            "Memory",
            metric.memory_mb,
            baseline.memory_mb,
            higher_is_better=False,
            formatter=lambda value: f"{value:.2f} MB",
        )
        if memory_line:
            lines.append(f"- {memory_line}")
        throughput_unit = metric.throughput_unit or baseline.throughput_unit or "items/s"
        throughput_line = _format_comparison_line(
            f"Throughput ({throughput_unit})",
            metric.throughput,
            baseline.throughput,
            higher_is_better=True,
            formatter=lambda value: f"{value:.2f} {throughput_unit}",
        )
        if throughput_line:
            lines.append(f"- {throughput_line}")
        return lines

    def _build_comparisons_payload(self) -> Dict[str, object]:
        comparisons: Dict[str, object] = {}
        for name, metric in self._performance.items():
            baseline = self._baseline.get(name)
            if not baseline:
                continue
            comparisons[name] = {
                "duration_ms": _comparison_payload(metric.duration_ms, baseline.duration_ms, higher_is_better=False),
                "memory_mb": _comparison_payload(metric.memory_mb, baseline.memory_mb, higher_is_better=False),
                "throughput": _comparison_payload(metric.throughput, baseline.throughput, higher_is_better=True),
            }
        return comparisons

    def _render_trend_charts(self) -> List[str]:
        if not self._performance_trends:
            return []
        lines: List[str] = []
        for name, points in sorted(self._performance_trends.items()):
            if len(points) < 2:
                continue
            lines.extend(_render_metric_trend(name, points, metric_type="duration_ms", unit="ms"))
            lines.extend(_render_metric_trend(name, points, metric_type="memory_mb", unit="MB"))
            unit = _trend_unit(points)
            lines.extend(_render_metric_trend(name, points, metric_type="throughput", unit=unit))
        return lines


def _metric_payload(metric: PerformanceMetric) -> Dict[str, object]:
    return {
        "duration_ms": metric.duration_ms,
        "memory_mb": metric.memory_mb,
        "throughput": metric.throughput,
        "throughput_unit": metric.throughput_unit,
    }


def _threshold_payload(thresholds: PerformanceThresholds) -> Dict[str, object]:
    return {
        "max_duration_ms": thresholds.max_duration_ms,
        "max_memory_mb": thresholds.max_memory_mb,
        "min_throughput": thresholds.min_throughput,
    }


def _trend_payload(point: PerformanceTrendPoint) -> Dict[str, object]:
    return {
        "label": point.label,
        "duration_ms": point.duration_ms,
        "memory_mb": point.memory_mb,
        "throughput": point.throughput,
        "throughput_unit": point.throughput_unit,
    }


def _format_metric_lines(metric: PerformanceMetric) -> List[str]:
    lines: List[str] = []
    if metric.duration_ms is not None:
        lines.append(f"- Duration: {_format_duration_ms(metric.duration_ms)}")
    if metric.memory_mb is not None:
        lines.append(f"- Memory: {metric.memory_mb:.2f} MB")
    if metric.throughput is not None:
        unit = metric.throughput_unit or "items/s"
        lines.append(f"- Throughput: {metric.throughput:.2f} {unit}")
    return lines


def _format_duration_ms(duration_ms: float) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f}s"
    return f"{duration_ms:.0f}ms"


def _format_comparison_line(
    label: str,
    current: Optional[float],
    baseline: Optional[float],
    *,
    higher_is_better: bool,
    formatter,
) -> Optional[str]:
    if current is None or baseline is None:
        return None
    delta = current - baseline
    if baseline == 0:
        pct = None
    else:
        pct = (delta / baseline) * 100
    direction = "improved" if (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better) else "regressed"
    if delta == 0:
        direction = "unchanged"
    delta_text = formatter(abs(delta))
    pct_text = f"{abs(pct):.1f}%" if pct is not None else "n/a"
    return f"{label}: {formatter(current)} vs {formatter(baseline)} ({direction}, Δ {delta_text}, {pct_text})"


def _comparison_payload(current: Optional[float], baseline: Optional[float], *, higher_is_better: bool) -> Dict[str, object]:
    if current is None or baseline is None:
        return {"current": current, "baseline": baseline, "delta": None, "delta_pct": None, "status": "unknown"}
    delta = current - baseline
    delta_pct = None if baseline == 0 else (delta / baseline) * 100
    if delta == 0:
        status = "unchanged"
    elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
        status = "improved"
    else:
        status = "regressed"
    return {
        "current": current,
        "baseline": baseline,
        "delta": delta,
        "delta_pct": delta_pct,
        "status": status,
    }


def _render_metric_trend(
    name: str,
    points: List[PerformanceTrendPoint],
    *,
    metric_type: str,
    unit: str,
) -> List[str]:
    values: List[float] = []
    labels: List[str] = []
    for point in points:
        value = getattr(point, metric_type)
        if value is None:
            continue
        values.append(float(value))
        labels.append(point.label)
    if len(values) < 2:
        return []
    max_value = max(values)
    max_axis = max_value * 1.1 if max_value else 1.0
    title = f"{name} {metric_type.replace('_', ' ').title()}"
    lines = [
        "```mermaid",
        "xychart-beta",
        f'  title "{title}"',
        f"  x-axis [{_format_mermaid_labels(labels)}]",
        f'  y-axis "{unit}" 0 --> {max_axis:.2f}',
        f'  line "{metric_type}" [{_format_mermaid_values(values)}]',
        "```",
        "",
    ]
    return lines


def _format_mermaid_labels(labels: List[str]) -> str:
    safe = [label.replace('"', "'") for label in labels]
    return ", ".join(f'"{label}"' for label in safe)


def _format_mermaid_values(values: List[float]) -> str:
    return ", ".join(f"{value:.2f}" for value in values)


def _trend_unit(points: List[PerformanceTrendPoint]) -> str:
    for point in points:
        if point.throughput_unit:
            return point.throughput_unit
    return "items/s"


def _coerce_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)
