# Verification

## What Is Tested in CI

CI runs `pytest` as the single required test gate.

Smoke coverage lives in:

- `tests/integration/test_smoke.py`

That keeps core behavior checks with the rest of the test suite under normal test naming conventions.

## Running Tests Locally

```bash
pip install -e ".[dev]"
pytest
```

## Non-Test Verification Probes

The scripts below are for manual verification and investigation (not the primary CI gate):

```bash
# Run all non-test verification phases
python scripts/run_suite.py

# Run specific phases
python scripts/run_provider_probe.py   # Embedding providers
python scripts/run_edge_bench.py       # Edge cases + performance
```

`run_suite.py` now includes phases 2-4 only:

- Phase 2: Embedding Providers
- Phase 3: Edge Cases
- Phase 4: Performance Benchmarks
