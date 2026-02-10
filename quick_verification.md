# Gloggur Verification Report

**Generated:** 2026-02-01 18:07:46
**Status:** PASSED (6/8 passed, 2 skipped)

## Executive Summary

- Total tests: 8
- Passed: 6
- Failed: 0
- Skipped: 2
- Recommendation: Ready with skipped tests; complete skipped phases before release.

## Phase 1: Smoke Tests

[PASS] 5/5 passed

- [PASS] Phase 1 Smoke Tests: Test 1.1: Basic Indexing: Indexed 30 files, 372 symbols in 878ms
- [PASS] Phase 1 Smoke Tests: Test 1.2: Incremental Indexing: Skipped 40/40 files; speedup 94.4% (878ms -> 49ms)
- [PASS] Phase 1 Smoke Tests: Test 1.3: Search Functionality: Search returned 5 results with valid scores and filters
- [PASS] Phase 1 Smoke Tests: Test 1.4: Docstring Audit: Audit returned 239 warning entries
- [PASS] Phase 1 Smoke Tests: Test 1.5: Status & Cache Management: Status reported 374 symbols; cache cleared successfully

## Phase 2: Embedding Providers

[PASS] 1/3 passed, 2 skipped

- [PASS] Test 2.1: Local Embeddings: Indexed 30 files and returned 3 results
- [SKIP] Test 2.2: OpenAI Embeddings: Skipped: OPENAI_API_KEY not set
- [SKIP] Test 2.3: Gemini Embeddings: Skipped: GLOGGUR_GEMINI_API_KEY or GEMINI_API_KEY not set

## Performance Summary

- Phase 1: Smoke Tests
  - metrics: {'Indexing': {'duration_ms': 878.0, 'memory_mb': None, 'throughput': 34.16856492027335, 'throughput_unit': 'files/s'}, 'Incremental Indexing': {'duration_ms': 49.0, 'memory_mb': None, 'throughput': 816.3265306122448, 'throughput_unit': 'files/s'}, 'Search': {'duration_ms': 2.0, 'memory_mb': None, 'throughput': 2500.0, 'throughput_unit': 'results/s'}, 'Phase 1 Total': {'duration_ms': 5831.95545792114, 'memory_mb': None, 'throughput': 0.8573453683033274, 'throughput_unit': 'tests/s'}}
  - thresholds: {'Indexing': {'max_duration_ms': 30000.0, 'max_memory_mb': None, 'min_throughput': None}, 'Incremental Indexing': {'max_duration_ms': 10000.0, 'max_memory_mb': None, 'min_throughput': None}, 'Search': {'max_duration_ms': 5000.0, 'max_memory_mb': None, 'min_throughput': None}}
  - warnings: []
  - baseline: {}
  - comparisons: {}
  - trends: {}

## Issues & Recommendations

- No issues reported.
