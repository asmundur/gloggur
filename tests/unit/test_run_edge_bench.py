from __future__ import annotations

from scripts.run_edge_bench import BenchmarkMetric, _apply_baseline_contract


def test_apply_baseline_contract_adds_thresholds_and_comparisons() -> None:
    baseline_payload = {
        "performance": {
            "metrics": {
                "cold_index_duration": {"duration_ms": 100.0},
                "unchanged_incremental_duration": {"duration_ms": 20.0},
                "search_average_latency": {"duration_ms": 5.0},
                "index_throughput": {"throughput": 200.0, "throughput_unit": "symbols/s"},
            }
        }
    }

    payload = _apply_baseline_contract(
        [
            BenchmarkMetric(name="cold_index_duration", duration_ms=110.0),
            BenchmarkMetric(name="unchanged_incremental_duration", duration_ms=18.0),
            BenchmarkMetric(name="search_average_latency", duration_ms=4.0),
            BenchmarkMetric(name="index_throughput", throughput=210.0, throughput_unit="symbols/s"),
        ],
        baseline_payload,
    )

    thresholds = payload.get("thresholds")
    assert isinstance(thresholds, dict)
    assert thresholds["cold_index_duration"]["max_duration_ms"] == 120.0
    assert thresholds["unchanged_incremental_duration"]["max_duration_ms"] == 25.0
    assert thresholds["search_average_latency"]["max_duration_ms"] == 6.0
    assert thresholds["index_throughput"]["min_throughput"] == 170.0

    comparisons = payload.get("comparisons")
    assert isinstance(comparisons, dict)
    assert comparisons["cold_index_duration"]["duration_ms"]["status"] == "regressed"
    assert comparisons["unchanged_incremental_duration"]["duration_ms"]["status"] == "improved"
    assert comparisons["index_throughput"]["throughput"]["status"] == "improved"


def test_apply_baseline_contract_emits_warning_when_threshold_exceeded() -> None:
    baseline_payload = {
        "performance": {
            "metrics": {
                "cold_index_duration": {"duration_ms": 100.0},
                "unchanged_incremental_duration": {"duration_ms": 20.0},
                "search_average_latency": {"duration_ms": 5.0},
                "index_throughput": {"throughput": 200.0, "throughput_unit": "symbols/s"},
            }
        }
    }

    payload = _apply_baseline_contract(
        [
            BenchmarkMetric(name="cold_index_duration", duration_ms=130.0),
            BenchmarkMetric(name="unchanged_incremental_duration", duration_ms=30.0),
            BenchmarkMetric(name="search_average_latency", duration_ms=7.0),
            BenchmarkMetric(name="index_throughput", throughput=100.0, throughput_unit="symbols/s"),
        ],
        baseline_payload,
    )

    warnings = payload.get("warnings")
    assert isinstance(warnings, list)
    assert len(warnings) == 4
