# Verification

## What Is Tested in CI

CI runs `pytest` as the primary required test gate.

The verification workflow also includes non-pytest required gates on Python `3.13`:

- `python scripts/run_static_quality_gates.py --format json`
- `python scripts/check_error_catalog_contract.py --format json`
- `python scripts/run_smoke.py --format json`
- `python scripts/run_packaging_smoke.py --format json`
- `python scripts/run_artifact_smoke.py --format json`
- `python scripts/run_edge_bench.py --benchmark-only --baseline-file benchmarks/performance_baseline.json --format json`

Verification harness scripts in this repository force
`GLOGGUR_EMBEDDING_PROVIDER=test` for deterministic offline behavior.
Deprecated `GLOGGUR_LOCAL_FALLBACK` now fails closed with
`local_fallback_env_unsupported`.

The workflow also includes a lane-policy audit gate:

- each matrix lane emits a JSON lane report artifact (`verification-lane-<python-version>.json`)
- `lane-audit` runs after matrix completion and fails non-zero on missing lanes, policy drift, or required-lane failures

Smoke coverage lives in:

- `tests/integration/test_smoke.py`

That keeps core behavior checks with the rest of the test suite under normal test naming conventions.

## Running Tests Locally

```bash
pip install -e ".[dev]"
pytest
```

Pytest defaults:

- parallel workers enabled (`-n auto --dist=loadscope`)
- coverage enabled with terminal + XML reports

Useful overrides:

```bash
# Force serial execution
pytest -n 0

# Generate HTML coverage report on demand
pytest --cov-report=html
```

## Non-Test Verification Probes

The scripts below are for manual verification and investigation (not the primary CI gate):

```bash
# Run full smoke workflow with stage-specific diagnostics
python scripts/run_smoke.py --format json

# Run fail-closed static quality gates for the verification control plane
python scripts/run_static_quality_gates.py --format json

# Run full packaging/distribution smoke (build -> install -> upgrade -> CLI checks)
python scripts/run_packaging_smoke.py --format json

# Optional faster build-only variant
python scripts/run_packaging_smoke.py --format json --skip-install-smoke

# Run artifact publish -> validate -> restore smoke
python scripts/run_artifact_smoke.py --format json

# Run deterministic performance regression benchmarks against the checked-in baseline
python scripts/run_edge_bench.py --benchmark-only --baseline-file benchmarks/performance_baseline.json --format json

# Verify the published error-code catalog matches live source contracts
python scripts/check_error_catalog_contract.py --format json

# Run minimal reference agent loop eval harness (10 built-in cases)
python scripts/run_reference_agent_eval.py --mode eval --format json --min-pass-rate 0.8

# Validate quickstart docs contract against source failure-code references
python scripts/check_quickstart_contract.py --format json

# Validate the consolidated CLI error-code catalog against source code maps
python scripts/check_error_code_catalog.py --format json

# Run the documented quickstart sequence on a fixture repository
python scripts/run_quickstart_smoke.py --format json

# Audit downloaded lane-report artifacts locally
python scripts/audit_verification_lanes.py --reports-dir verification-lane-artifacts --format json

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

`run_smoke.py` validates this ordered workflow in one command:

1. clean index build
2. watch incremental update
3. resume contract via `status --json`
4. retrieval via `search --json`
5. inspect summary via `inspect --json`

On failure it exits non-zero and emits a stable stage `failure.code` in JSON output.

`run_packaging_smoke.py` validates packaging contracts with deterministic stage codes:

- `packaging_build_failed`
- `packaging_install_failed`
- `packaging_upgrade_failed`
- `packaging_help_failed`
- `packaging_status_failed`

`run_artifact_smoke.py` validates downstream artifact reuse with deterministic stage codes:

1. source index build
2. artifact publish
3. artifact validate
4. artifact restore into a fresh cache directory
5. restored `status --json`
6. restored `search --json`

Failure codes:

- `artifact_smoke_index_failed`
- `artifact_smoke_publish_failed`
- `artifact_smoke_validate_failed`
- `artifact_smoke_restore_failed`
- `artifact_smoke_status_failed`
- `artifact_smoke_search_failed`

`run_edge_bench.py` validates phase-4 benchmark regression policy against the
checked-in baseline at `benchmarks/performance_baseline.json`.

The baseline-backed gate fails with `performance_threshold_exceeded` when any of
these drift limits are exceeded:

- cold index duration: more than 20% slower than baseline
- unchanged incremental duration: more than 25% slower than baseline
- search average latency: more than 20% slower than baseline
- index throughput: more than 15% below baseline

To refresh the baseline intentionally after an accepted performance tradeoff:

```bash
python scripts/run_edge_bench.py \
  --benchmark-only \
  --baseline-file benchmarks/performance_baseline.json \
  --write-baseline \
  --format json
```
