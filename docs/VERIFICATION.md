# Verification Suite

## Overview

The verification suite exercises Gloggur's indexing, search, and storage workflows to catch regressions early. Run it before releases, after large refactors, or when changing embedding providers.

The suite is split into phases so you can run the light checks quickly and the heavier checks when needed.

## Quick Start

```bash
# Install gloggur
pip install -e .

# Run full verification
python scripts/run_suite.py
```

## Running Individual Phases

```bash
# Phase 1: Smoke tests
python scripts/run_smoke.py

# Phase 2: Embedding providers
python scripts/run_provider_probe.py

# Phase 3 & 4: Edge cases and performance
python scripts/run_edge_bench.py
```

## CI/CD Integration

The GitHub Actions workflow runs verification automatically on pushes to `main`/`develop`, on pull requests targeting `main`, and nightly at 02:00 UTC.

- The workflow runs against Python 3.10, 3.11, and 3.12.
- Verification reports are uploaded as build artifacts.
- Pull requests receive a comment with the latest report.

### Required Secrets

Some phases need API keys to call external embedding providers. Add secrets in the repository settings:

- `GEMINI_API_KEY` (optional, for Gemini tests)
- `OPENAI_API_KEY` (optional, for OpenAI tests)

If a key is missing, the related tests should skip or report a failure depending on the phase configuration.

## Interpreting Results

- A passing phase indicates schema checks, cache integrity, and expected behaviors succeeded.
- A failure includes a message and details in the report for fast diagnosis.
- The full report aggregates all phases with timings and outputs.

## Troubleshooting

- **API key errors**: Confirm the secret exists in GitHub and matches the provider used in your config.
- **Performance issues**: Run Phase 3 & 4 locally to reproduce; consider clearing cache before re-testing.
- **Cache/database failures**: Delete the cache directory configured by `GLOGGUR_CACHE_DIR` and rerun the suite.

## Extending the Suite

- Add new tests to the relevant phase script.
- Use the `scripts/verification` helpers (runner, fixtures, reporters, checks) for consistent output.
- Update `run_suite.py` to include new phases or change ordering.

## Development Workflow

For local guardrails, optionally install the pre-push hook:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-push
```

The hook runs the full suite in quick mode before pushing.
