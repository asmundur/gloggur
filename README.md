# Gloggur

Gloggur is a symbol-level, incremental codebase indexer for semantic search and precedent retrieval. It parses code into symbols, generates embeddings, stores them locally, and exposes a JSON-friendly CLI for integration with AI agents.

## Features

- Tree-sitter parsing for multi-language symbol extraction
- Incremental indexing with SHA-256 hashing
- Pluggable embedding backends (local models, OpenAI, or Gemini)
- FAISS vector search with SQLite metadata
- Docstring audit with semantic similarity scoring
- JSON output for CLI automation

## Installation

Preferred:

```bash
pipx install gloggur
```

Alternatives:

```bash
pip install gloggur
```

Optional:

```bash
pipx install "gloggur[openai]"
pipx install "gloggur[gemini]"
pipx install "gloggur[local]"
```

For local development worktrees, use the repo wrapper/bootstrap flow:

```bash
scripts/bootstrap_gloggur_env.sh
scripts/gloggur status --json
```

`scripts/bootstrap_gloggur_env.sh` now runs index freshness checks automatically
when `scripts/gloggur` is available (`status --json`, optional `index . --json`,
then a final `status --json` verification), then runs the canonical startup
readiness probe:

```bash
python scripts/check_startup_readiness.py --format json
```

The readiness probe verifies, in order:
- `scripts/gloggur status --json`
- `scripts/gloggur watch status --json`

It fails non-zero with deterministic startup codes when either probe fails or
the watch runtime state is contradictory (`startup_status_probe_failed`,
`startup_watch_status_probe_failed`, `startup_watch_payload_invalid`,
`startup_watch_state_contradictory`).

`scripts/gloggur` now runs a preflight check before launching the CLI:
- prefers `.venv/bin/python` when healthy
- otherwise falls back to system Python with repo-root `PYTHONPATH`
- returns structured `--json` failures with `operation=preflight`
- is the canonical command for worktree-local agent/dev flows (no PATH assumptions)

Optional fast cache hydration from another workspace:

```bash
# fastest (same machine): symlink the cache
scripts/bootstrap_gloggur_env.sh --seed-cache-from /path/to/other/workspace --seed-cache-mode symlink

# isolated copy if you do not want shared cache state
scripts/bootstrap_gloggur_env.sh --seed-cache-from /path/to/other/workspace --seed-cache-mode copy
```

Use `copy` mode when symlink targets may be read-only in your execution environment.

Offline-friendly full bootstrap from another workspace (venv + cache):

```bash
scripts/bootstrap_gloggur_env.sh \
  --seed-venv-from /path/to/other/workspace \
  --seed-venv-mode symlink \
  --seed-cache-from /path/to/other/workspace \
  --seed-cache-mode symlink
```

## Quickstart

For a deterministic first-run path (install, provider setup, index/watch/search/inspect, troubleshooting codes), use `docs/QUICKSTART.md`.

## Task Tracking

This repository now uses [Beads (`bd`)](https://github.com/steveyegge/beads) for new task tracking.

- Start with `bd ready` to see unblocked work.
- Claim work with `bd update <id> --claim`.
- Create new tasks with `bd create --title "..." -p 2`.
- Use `bd show <id>` for the full task record.

`TODOs.md` and `DONEs.md` remain in the repo during the transition as the historical Markdown backlog and verification log. Existing items stay there for now, but new work created after the Beads rollout should be tracked in `bd`.

Transition and retirement policy:
- Keep `TODOs.md` and `DONEs.md` until every historical open Markdown task is either imported and verified in Beads or explicitly archived/cancelled with provenance.
- Treat `DONEs.md` as a historical archive during coexistence; do not delete it as part of the Beads rollout.
- Retire Markdown tracking only after one full release cycle with no new work landing in Markdown, no Beads command panics during normal serialized use, stable `.githooks` integration, and DB/export parity in `.beads/issues.jsonl`.
- If Beads export or sync becomes unreliable again, keep Markdown tracking active and reopen the retirement decision instead of forcing a cutover.

Beads operational note:
- Run `bd` commands serially in this repo. Parallel `bd` invocations against the embedded Dolt store have triggered nil-pointer panics during local verification.

For the consolidated machine-readable CLI error and resume code catalog, use `docs/ERROR_CODES.md`.

Index a repository:

```bash
gloggur index . --json
```

Search for similar symbols:

```bash
gloggur search "streaming parser" --top-k 5 --json
```

Stream results as line-delimited JSON:

```bash
gloggur search "streaming parser" --top-k 50 --json --stream
```

Inspect docstrings (semantic similarity scoring):

```bash
gloggur inspect . --json
```

By default, directory inspect focuses on high-signal runtime paths (`src/` + non-test/script files)
and excludes `tests/` + `scripts/`. Opt in to full-audit mode with:

```bash
gloggur inspect . --json --include-tests --include-scripts
```

Inspect JSON now includes grouped summaries under `warning_summary`:
- `by_warning_type`
- `by_path_class` / `reports_by_path_class`
- `top_files`
- `inspect_payload_schema_version` for stable automation parsing contracts
- `inspect_payload_schema_policy` with explicit bump rules:
  - increment `inspect_payload_schema_version` for breaking changes (remove/rename fields, type/semantic changes)
  - additive optional fields/reason-codes are allowed without a version bump
- deterministic partial-failure contract fields:
  - `failed_reasons`, `failure_codes`, `failure_guidance`
  - `allow_partial` and `allow_partial_applied` so partial-success runs are explicit
  - fail-closed `index`, `inspect`, and foreground `watch start` exits now also include
    a top-level `error` object whose `code` mirrors the primary failure code

The consolidated published error-code catalog lives in `docs/ERROR_CODES.md`.

`gloggur inspect` skips unchanged files by default. Use `--force` to reinspect everything.

Skipped files now reuse cached audit reports, exposed via `cached_files_reused`,
`cached_reports_reused`, and `cached_warning_reports_reused`, so repeated
`inspect --json` runs preserve both actionable warnings and clean semantic-score
metadata without resurrecting stale findings from previously fixed files.

Check status:

```bash
gloggur status --json
```

Enable save-triggered indexing (one-time setup):

```bash
gloggur watch init . --json
gloggur watch start --daemon --json
```

Check or stop watcher:

```bash
gloggur watch status --json
gloggur watch stop --json
```

`watch start --json` preflight failures now emit stable machine-readable codes:
- `watch_mode_conflict` (both `--foreground` and `--daemon`)
- `watch_path_missing` (configured watch path does not exist)
- `watch_mode_invalid` (unsupported configured watch mode)

Clear cache:

```bash
gloggur clear-cache --json
```

Publish a reusable cache artifact for CI/CD handoff (local/file destinations):

```bash
gloggur artifact publish --json --destination ./artifacts/
```

`artifact publish --json` fails closed with stable codes for common preflight issues:
- `artifact_source_uninitialized`
- `artifact_destination_unsupported`
- `artifact_destination_exists`
- `artifact_destination_inside_source`

Publish through an external uploader command (for provider CLIs, artifact stores, or presigned upload wrappers):

```bash
gloggur artifact publish --json \
  --destination https://artifacts.example.com/gloggur/cache.tgz \
  --uploader-command 'scripts/upload_artifact {artifact_path} {destination}'
```

Supported uploader placeholders:
- `{artifact_path}` or `{artifact}`: staged archive file to upload
- `{destination}`: raw `--destination` value
- `{artifact_name}`: generated artifact filename
- `{archive_sha256}`
- `{archive_bytes}`
- `{manifest_sha256}`

Uploader-command failures emit stable machine-readable codes:
- `artifact_uploader_command_invalid`
- `artifact_uploader_failed`
- `artifact_uploader_timeout`

Publish directly to an HTTP(S) endpoint (presigned URL style PUT):

```bash
gloggur artifact publish --json \
  --destination https://artifacts.example.com/gloggur/cache.tgz
```

Direct HTTP upload sends:
- request method: `PUT`
- content type: `application/gzip`
- metadata headers:
  - `X-Gloggur-Archive-Sha256`
  - `X-Gloggur-Archive-Bytes`
  - `X-Gloggur-Manifest-Sha256`

Direct HTTP upload failures emit stable machine-readable codes:
- `artifact_http_upload_failed`
- `artifact_http_upload_timeout`

Validate a published artifact before handing it to a downstream runner:

```bash
gloggur artifact validate --json --artifact ./artifacts/gloggur-index-20260227T000000Z.tar.gz
```

Restore a validated artifact into a cache directory for downstream reuse:

```bash
gloggur artifact restore --json \
  --artifact ./artifacts/gloggur-index-20260227T000000Z.tar.gz \
  --destination ./.gloggur-cache-restored
```

`artifact validate --json` / `artifact restore --json` fail closed with stable codes for common integrity and restore issues:
- `artifact_path_missing`
- `artifact_manifest_missing`
- `artifact_manifest_invalid`
- `artifact_manifest_schema_unsupported`
- `artifact_manifest_file_mismatch`
- `artifact_manifest_totals_mismatch`
- `artifact_restore_destination_exists`
- `artifact_restore_destination_not_directory`

Cache compatibility is automatic:
- If cache schema changes, Gloggur rebuilds `.gloggur-cache/index.db` automatically.
- If embedding provider/model changes, the next `gloggur index ...` run rebuilds cache and vectors automatically.
- `gloggur status --json` and `gloggur search --json` expose session-resume fields:
  - `resume_decision` (`resume_ok` or `reindex_required`)
  - `resume_reason_codes` (machine-readable mismatch causes)
  - `expected_resume_fingerprint` / `cached_resume_fingerprint` for deterministic compatibility checks
  - `last_success_resume_fingerprint` / `last_success_resume_at` to compare current state against the last successful reusable index state
  - `tool_version` / `last_success_tool_version` to detect version drift; when marker drift is detected, compatibility is `reindex_required` (`tool_version_changed`)

Session resume decision flow:

```bash
gloggur status --json
# if resume_decision == "resume_ok": reuse cache
# if resume_decision == "reindex_required": run `gloggur index . --json` first
```

Controlled tool-version drift override (offline/air-gapped fallback):

```bash
gloggur status --json --allow-tool-version-drift
gloggur search "query" --json --allow-tool-version-drift
```

You can also set an environment override for non-interactive runs:

```bash
export GLOGGUR_ALLOW_TOOL_VERSION_DRIFT=true
gloggur status --json
gloggur search "query" --json
```

Accepted env values are strict: `1|true|yes|on|0|false|no|off` (case-insensitive). Invalid values fail non-zero with `allow_tool_version_drift_env_invalid`.

When this override is used, JSON output remains explicit and machine-readable:
- `tool_version_drift_detected: true`
- `allow_tool_version_drift: true`
- `tool_version_drift_override_applied: true`
- `resume_reason_codes` includes `tool_version_changed_override`

Evidence trace + grounding validation (opt-in):

```bash
gloggur search "add function" --json \
  --with-evidence-trace \
  --validate-grounding \
  --evidence-min-confidence 0.6 \
  --evidence-min-items 1
```

When validation must block ungrounded output, enforce non-zero exit:

```bash
gloggur search "query" --json --validate-grounding --fail-on-ungrounded
```

Grounding failure path is explicit and machine-readable:
- `validation.reason_code`: `grounding_evidence_missing` or `grounding_confidence_below_threshold`
- `error.code`: `search_grounding_validation_failed`
- `failure_codes`: includes `search_grounding_validation_failed`

Minimal reference agent loop + eval harness:

```bash
# Single run (retrieve -> validate -> bounded retry -> stop)
python scripts/run_reference_agent_eval.py --mode run --query "add numbers token" --format json

# Deterministic 10-case eval suite (fails non-zero below threshold)
python scripts/run_reference_agent_eval.py --mode eval --format json --min-pass-rate 0.8
```

Harness logs include deterministic agent steps: `decide`, `act`, `validate`, `stop`.

Concurrency behavior:
- Readers (`status`, `search`) are safe to run concurrently.
- Cache writers (`index`, `clear-cache`, watch updates) are serialized via `.gloggur-cache/.cache-write.lock`.
- Writer lock acquisition is bounded (default `5000ms` total wait) and fails fast with structured JSON on timeout.

## On-Save Indexing (Watch Mode)

Watch mode keeps the index updated in the background as files are saved.

- Filesystem events are debounced (`watch_debounce_ms`, default `300`) to coalesce save bursts.
- Content hashing avoids re-parsing/re-embedding unchanged files.
- Deleted/changed symbol vectors are removed before upsert to prevent stale search hits.
- Runtime state is written to `.gloggur-cache/watch_state.json` by default.

## Verification

Core behavior checks run in `pytest` (including smoke tests):

```bash
.venv/bin/pytest
```

Default pytest settings run tests in parallel (`-n auto --dist=loadscope`).
Use serial mode when debugging order-sensitive behavior:

```bash
.venv/bin/pytest -n 0
```

Coverage reports include terminal + XML by default. Generate HTML on demand:

```bash
.venv/bin/pytest --cov-report=html
```

Coverage is intentionally targeted at the runtime package path (`src/gloggur`) so
reports reflect production modules instead of the repo-root bootstrap shim.

The first required static-quality gate now runs through one fail-closed command on
the required Python `3.13` lane:

```bash
python scripts/run_static_quality_gates.py --format json
```

This gate currently covers the verification control plane (`verification.yml`,
lane-audit tooling, and its workflow-policy tests) and explicitly avoids shadow
worktree noise by excluding `.claude/` and `.gloggur-cache` from formatter/linter
discovery.

Additional verification probes are available for provider/edge/performance checks:

```bash
# Run fail-closed static quality gates for the CI verification control plane
python scripts/run_static_quality_gates.py --format json

# Run the full index -> watch -> resume -> search -> inspect smoke workflow
python scripts/run_smoke.py --format json

# Run full packaging/distribution smoke (build -> install -> upgrade -> CLI checks)
python scripts/run_packaging_smoke.py --format json

# Optional faster build-only variant
python scripts/run_packaging_smoke.py --format json --skip-install-smoke

# Run artifact publish -> validate -> restore -> downstream search smoke
python scripts/run_artifact_smoke.py --format json

# Verify the published error-code catalog matches live source contracts
python scripts/check_error_catalog_contract.py --format json

# Run deterministic benchmark regression checks against the checked-in baseline
python scripts/run_edge_bench.py --benchmark-only --baseline-file benchmarks/performance_baseline.json --format json

# Run non-test verification phases (providers, edge cases, performance)
python scripts/run_suite.py

# Run specific phases
python scripts/run_provider_probe.py  # Embedding providers
python scripts/run_edge_bench.py  # Edge cases & performance
```

`scripts/run_smoke.py` fails fast with deterministic stage codes:
`smoke_index_failed`, `smoke_watch_incremental_failed`, `smoke_resume_status_failed`,
`smoke_search_failed`, and `smoke_inspect_failed`.

`scripts/run_packaging_smoke.py` emits deterministic stage codes:
`packaging_build_failed`, `packaging_install_failed`, `packaging_upgrade_failed`,
`packaging_help_failed`, and `packaging_status_failed`.

`scripts/run_artifact_smoke.py` emits deterministic stage codes:
`artifact_smoke_index_failed`, `artifact_smoke_publish_failed`,
`artifact_smoke_validate_failed`, `artifact_smoke_restore_failed`,
`artifact_smoke_status_failed`, and `artifact_smoke_search_failed`.

`scripts/run_edge_bench.py` now benchmarks a deterministic generated fixture by
default instead of the mutable repo checkout. The checked-in benchmark baseline
lives at `benchmarks/performance_baseline.json`.

To refresh the baseline intentionally:

```bash
python scripts/run_edge_bench.py \
  --benchmark-only \
  --baseline-file benchmarks/performance_baseline.json \
  --write-baseline \
  --format json
```

The performance regression policy fails with `performance_threshold_exceeded`
when any metric drifts beyond the allowed threshold:
- cold index duration: more than 20% slower than baseline
- unchanged incremental duration: more than 25% slower than baseline
- search average latency: more than 20% slower than baseline
- index throughput: more than 15% below baseline

Provider verification checklist (OpenAI + Gemini):

```bash
# Deterministic mocked provider selection + index/search flow
.venv/bin/python -m pytest \
  tests/integration/test_provider_cli_integration.py \
  tests/unit/test_embeddings.py \
  tests/unit/test_embeddings_factory.py \
  tests/unit/test_run_provider_probe.py -q

# Optional live provider smoke probe (requires API keys)
python scripts/run_provider_probe.py --format markdown
```

Provider verification now fails closed on malformed SDK payloads:
- missing vectors,
- count mismatches between requested inputs and returned embeddings,
- non-numeric vector entries,
- inconsistent vector dimensions.

See `docs/VERIFICATION.md` for detailed documentation.
See `docs/AGENT_INTEGRATION.md` for agent integration guidance.

## Python Support Policy

Verification lanes in `.github/workflows/verification.yml` are split into required and provisional tiers:

- Required (blocking): `3.10`, `3.11`, `3.12`, `3.13`
- Provisional (non-blocking): `3.14`

`3.14` remains provisional while dependency/runtime compatibility stabilizes across the full stack.
Graduation criteria for `3.14` to required:

- at least two consecutive green CI runs on representative Python-touching PRs
- no open `3.14`-specific compatibility bugs in project issue tracking
- dependency install + pytest lane remains stable without temporary workarounds

Lane evidence is now explicit and fail-closed in CI:

- each matrix lane writes `verification-lane-<python-version>.json` and uploads it as an artifact
- a follow-on `lane-audit` job downloads all lane artifacts and fails non-zero when:
  - any expected lane report is missing
  - required/provisional policy classification drifts
  - any required lane reports non-success status

## Configuration

Create `.gloggur.yaml` or `.gloggur.json` in your repository:

```yaml
embedding_provider: gemini
local_embedding_model: microsoft/codebert-base
openai_embedding_model: text-embedding-3-large
gemini_embedding_model: gemini-embedding-001
cache_dir: .gloggur-cache
watch_enabled: false
watch_path: .
watch_debounce_ms: 300
watch_mode: daemon
watch_state_file: .gloggur-cache/watch_state.json
watch_pid_file: .gloggur-cache/watch.pid
watch_log_file: .gloggur-cache/watch.log
docstring_semantic_threshold: 0.10
docstring_semantic_max_chars: 4000
docstring_semantic_min_code_chars: 30
docstring_semantic_kind_thresholds:
  class: 0.05
  interface: 0.05
supported_extensions:
  - .py
  - .js
  - .jsx
  - .ts
  - .tsx
  - .rs
  - .go
  - .java
excluded_dirs:
  - .git
  - node_modules
  - venv
  - .venv
  - .gloggur-cache
  - dist
  - build
  - htmlcov
```

Semantic threshold rationale:
- `docstring_semantic_threshold=0.10` is calibrated for `microsoft/codebert-base` to reduce false-positive warning noise compared with `0.20`.
- `docstring_semantic_kind_thresholds` keeps `class`/`interface` checks more lenient (`0.05`) because abstract descriptions are typically higher-level than implementation bodies.
- `docstring_semantic_min_code_chars=30` avoids scoring trivially short implementations where cosine similarity is unstable.

Environment variables:

- `GLOGGUR_EMBEDDING_PROVIDER`
- `GLOGGUR_LOCAL_MODEL`
- `GLOGGUR_OPENAI_MODEL`
- `OPENAI_API_KEY`
- `GLOGGUR_GEMINI_MODEL`
- `GLOGGUR_GEMINI_API_KEY`
- `GLOGGUR_CACHE_DIR`
- `GLOGGUR_WATCH_ENABLED`
- `GLOGGUR_WATCH_PATH`
- `GLOGGUR_WATCH_DEBOUNCE_MS`
- `GLOGGUR_WATCH_MODE`
- `GLOGGUR_WATCH_STATE_FILE`
- `GLOGGUR_WATCH_PID_FILE`
- `GLOGGUR_WATCH_LOG_FILE`
- `GLOGGUR_DOCSTRING_SEMANTIC_MIN_CHARS`
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

Gloggur also auto-loads a repo-local `.env` file at runtime; exported process environment variables still take precedence over `.env` values.

## Output Schema

Search results are returned as JSON:

```json
{
  "query": "...",
  "results": [
    {
      "symbol": "function_name",
      "kind": "function",
      "file": "path/to/file.py",
      "line": 42,
      "signature": "def function_name(arg1, arg2)",
      "docstring": "...",
      "similarity_score": 0.95,
      "context": "surrounding code snippet"
    }
  ],
  "metadata": {
    "total_results": 10,
    "search_time_ms": 45
  }
}
```

## Development

```bash
scripts/bootstrap_gloggur_env.sh
scripts/gloggur status --json
.venv/bin/pytest
```

If you want bare `gloggur` on PATH in a local shell session, activate `.venv` first:

```bash
source .venv/bin/activate
gloggur status --json
```

If bootstrap/preflight fails with `--json`, error payloads include:
- `error_code`: `missing_venv`, `missing_python`, `missing_package`, `broken_environment`
- `message`
- `remediation` steps
- `detected_environment` details
