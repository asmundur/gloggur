# Validation Suite

## Overview

The validation suite exercises Gloggur's indexing, search, and storage workflows to catch regressions early. Run it before releases, after large refactors, or when changing embedding providers.

The suite is split into phases so you can run the light checks quickly and the heavier checks when needed.

## Quick Start

```bash
# Install gloggur
pip install -e .

# Run full validation
python scripts/validate_all.py
```

## Running Individual Phases

```bash
# Phase 1: Smoke tests
python scripts/validate_phase1.py

# Phase 2: Embedding providers
python scripts/validate_phase2.py

# Phase 3 & 4: Edge cases and performance
python scripts/validate_phase3_4.py
```

## CI/CD Integration

The GitHub Actions workflow runs validation automatically on pushes to `main`/`develop`, on pull requests targeting `main`, and nightly at 02:00 UTC.

- The workflow runs against Python 3.10, 3.11, and 3.12.
- Validation reports are uploaded as build artifacts.
- Pull requests receive a comment with the latest report.

### Required Secrets

Some phases need API keys to call external embedding providers. Add secrets in the repository settings:

- `GEMINI_API_KEY` (optional, for Gemini tests)
- `OPENAI_API_KEY` (optional, for OpenAI tests)

If a key is missing, the related tests should skip or report a failure depending on the phase configuration.

## Interpreting Results

- A passing phase indicates schema validation, cache integrity, and expected behaviors succeeded.
- A failure includes a message and details in the report for fast diagnosis.
- The full report aggregates all phases with timings and outputs.

## Troubleshooting

- **API key errors**: Confirm the secret exists in GitHub and matches the provider used in your config.
- **Performance issues**: Run Phase 3 & 4 locally to reproduce; consider clearing cache before re-testing.
- **Cache/database failures**: Delete the cache directory configured by `GLOGGUR_CACHE_DIR` and rerun validation.

## Extending the Suite

- Add new tests to the relevant phase script.
- Use the `scripts/validation` helpers (runner, fixtures, reporters, validators) for consistent output.
- Update `validate_all.py` to include new phases or change ordering.

## Development Workflow

For local guardrails, optionally install the pre-push hook:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-push
```

The hook runs the full suite in quick mode before pushing.
