# Agent Integration Guide

This repository ships Gloggur, a symbol-level indexer intended for coding agents. If you are an agent working on this project, you **must** use Gloggur to understand and navigate the codebase.

## Bootstrap and preflight

Before first use in a fresh worktree:

```bash
scripts/bootstrap_gloggur_env.sh
```

Bootstrap now verifies index freshness automatically when `scripts/gloggur` is
present (status -> optional index -> status verification), then runs the
canonical startup readiness probe:

```bash
python scripts/check_startup_readiness.py --format json
```

This probe validates both `scripts/gloggur status --json` and
`scripts/gloggur watch status --json`, and fails non-zero with deterministic
startup codes when the worktree is not actually ready.

Optional cache hydration for faster first run:

```bash
scripts/bootstrap_gloggur_env.sh --seed-cache-from /path/to/other/workspace --seed-cache-mode symlink
```

If symlink targets are read-only in your runtime, switch to `--seed-cache-mode copy`.

If network/package installs are unavailable, seed both `.venv` and cache from a healthy workspace:

```bash
scripts/bootstrap_gloggur_env.sh \
  --seed-venv-from /path/to/other/workspace \
  --seed-venv-mode symlink \
  --seed-cache-from /path/to/other/workspace \
  --seed-cache-mode symlink
```

Agent wrapper entrypoint:

```bash
scripts/gloggur <command> --json
```

When `bd setup codex` is installed, the global `gloggur` launcher also works from
external repositories by delegating to repo `scripts/gloggur` while preserving the
caller working directory.
Bootstrap refreshes that global launcher by default, replacing stale legacy
wrappers that fail with "run inside a gloggur git worktree".

For in-repo automation, verification harnesses, and benchmark runs, always prefer
repo-local invocations (`scripts/gloggur ...` or `python -m gloggur.cli.main ...`)
instead of depending on ambient `PATH` launcher state.

`scripts/gloggur` preflight behavior:
- use repo `.venv` when healthy
- fallback to system Python + repo `PYTHONPATH` when `.venv` is missing/broken
- if startup is impossible and `--json` is enabled, emit one envelope object:
  - `ok=false`
  - `error_code` (`missing_venv`, `missing_python`, `missing_package`, `broken_environment`, etc.)
  - `error`
  - `stage` (`bootstrap|dispatch|search`)
  - `compatibility` (legacy remediation/detected-environment payload)

Optional bootstrap env vars no longer hard-fail when unset/unwritable:
- `BOOTSTRAP_GLOGGUR_LOG_FILE`
- `BOOTSTRAP_GLOGGUR_STATE_FILE`

Set `GLOGGUR_BOOTSTRAP_STRICT=1` to enforce hard-fail behavior for degraded bootstrap capabilities.

Global-wrapper launch failures under `--json` return deterministic payloads with:
- top-level envelope fields above plus compatibility details
- `error_code`: `wrapper_launch_target_missing` or `wrapper_install_root_invalid`

## Task tracking with Beads

New task tracking in this repo uses [Beads (`bd`)](https://github.com/steveyegge/beads).

- Check for available work with `bd ready`.
- Claim work with `bd update <id> --claim`.
- Create new tasks with `bd create --title "..." -p 2`.
- Inspect task details with `bd show <id>`.
- Close completed work with `bd close <id>`.
- Run `bd prime` when you need the current Beads workflow guidance optimized for agents.

Transition policy:
- `TODOs.md` and `DONEs.md` remain in the repo as the historical Markdown backlog and verification record.
- Existing Markdown tasks stay where they are until explicitly migrated or retired.
- New tasks created after the Beads rollout should go into `bd`.
- Retire Markdown tracking only after every historical open Markdown task is imported and verified in Beads or explicitly archived/cancelled with provenance, one release cycle passes with no new Markdown tasks, `.beads/issues.jsonl` stays in parity with the live Beads DB, and the hook integration remains stable.
- If Beads sync/export reliability regresses, keep Markdown tracking active and reopen the retirement decision.

Operational note:
- Run `bd` commands serially in this repo. Parallel `bd` invocations against the embedded Dolt store have reproduced tracker panics during local verification.

## Quick workflow

For the single-path onboarding flow with provider setup and troubleshooting codes, see `docs/QUICKSTART.md`.

1. **Create or refresh the index**:
   ```bash
   scripts/gloggur index . --json
   ```
   Optional one-time setup for background save-triggered indexing:
   ```bash
   scripts/gloggur watch init . --json
   scripts/gloggur watch start --daemon --json
   ```
2. **Locate relevant code** with semantic search:
   ```bash
   scripts/gloggur search "<query>" --top-k 5 --json
   scripts/gloggur search "rg -S -g '*.py' AuthToken" --json
   ```
   If the cache is not reusable, `search --json` exits non-zero with
   `error.code=search_cache_not_ready` and a top-level `metadata` object that
   includes resume/build-state details.
3. **Confirm coverage**:
   ```bash
   scripts/gloggur status --json
   ```
   Watcher runtime health:
   ```bash
   scripts/gloggur watch status --json
   ```
   One-command startup-readiness verification:
   ```bash
   python scripts/check_startup_readiness.py --format json
   ```
4. **Inspect docstrings (optional but recommended for documentation changes)**:
   ```bash
   scripts/gloggur inspect . --json
   ```
   Inspection skips unchanged files by default; add `--force` to reinspect everything.

## Support Bundles

Use the support CLI when a field tester needs a sendable diagnostics bundle after Glöggur fails.

Capture a traced run around the failing command:

```bash
scripts/gloggur support run -- search "add numbers token" --json
```

- Everything after `--` must be a normal Glöggur subcommand.
- Non-zero child exits automatically create a support bundle unless `--no-bundle-on-failure` is set.
- Add `--note "what went wrong"` when the tester can explain the failure in plain language.

Create a manual snapshot without rerunning the failing command:

```bash
scripts/gloggur support collect --json --note "manual support snapshot"
```

Support bundle paths:
- sessions: `.gloggur/support/sessions/<session-id>/`
- bundles: `.gloggur/support/bundles/gloggur-support-<session-id>.tar.gz`

Bundle policy:
- sanitized by default: secrets and local absolute paths are redacted
- runtime logs/status/config snapshots are included automatically
- cache/index databases stay out of the bundle unless `--include-cache` is passed

Subject repo coverage note:
- For subject repo analysis, test/coverage collection should run in the subject repo's own environment/toolchain (its own venv/runner), not in this Gloggur repo environment.
- After coverage is produced there, run `gloggur coverage import ...` from the subject repo cwd (or pass explicit coverage artifact paths).

Pytest defaults for this repo:

- tests run in parallel (`-n auto --dist=loadscope`)
- serial override for debugging: `pytest -n 0`
- HTML coverage report is opt-in: `pytest --cov-report=html`

## Tips for effective searches

- Search by **concepts**, not just filenames (e.g., "incremental hashing", "embedding provider", "tree-sitter parser").
- Use `--top-k` to widen or narrow results based on the task.
- Use `--stream` if you are integrating results into a tool chain.
- For identifier-heavy lookups where tests are noisy, use `--ranking-mode source-first`.
- Grep/ripgrep style queries are accepted as input compatibility (`rg Foo`, `rg -g "*.py" Foo`, `grep -R foo_bar src/`).
- Quoted short grep patterns are preserved in router debug (`parsed_query.pattern_quoted=true` for queries like `rg "id"`).
- Use `--file` for exact file or directory-prefix scoping (`exact_or_prefix` with
  normalized `src/` vs `./src/` behavior); check
  `metadata.file_filter_warning_codes` for `file_filter_no_match`.
- Use `--context-radius` (default `12`, range `1..200`) when agents need more or
  less surrounding implementation detail per hit.
- Symbol-backed hits can include `symbol_def` and `symbol_ref` tags; if `.gloggur/index/symbols.db`
  is missing, search still runs but symbol-tagged hits are absent until reindex.
- Search hits now expose repo-relative `path` plus additive `start_byte` / `end_byte`
  fields so agents can round-trip exact source text with `gloggur extract`.
- Missing/corrupt symbol index stays non-fatal: use `--debug-router` and inspect
  `debug.backend_errors.symbol` for deterministic diagnostics.

Search -> extract reference flow:

```bash
scripts/gloggur search "where is Foo defined" --json
scripts/gloggur extract sample.py 0 42 --json
```

- `hits[].path` is repo-relative and safe to feed directly into `extract`.
- `start_byte` is inclusive and `end_byte` is exclusive everywhere.
- `extract` rejects absolute or escaping paths and does not require a fresh index if you
  already have a valid path/span.

Grounded retrieve -> validate -> emit/repair flow (recommended for agent outputs):

```bash
scripts/gloggur search "<query>" --json \
  --with-evidence-trace \
  --validate-grounding \
  --evidence-min-confidence 0.6 \
  --evidence-min-items 1
```

- If `validation.passed` is `true`: emit response and cite `evidence_trace` items.
- If `validation.passed` is `false`: repair by broadening query and/or increasing `--top-k`.
- To hard-block ungrounded responses in automation, add `--fail-on-ungrounded` and branch on non-zero exit plus `error.code=search_grounding_validation_failed`.

Minimal reference loop/eval harness:

```bash
# Single query loop with structured step logs
python scripts/run_reference_agent_eval.py --mode run --query "<query>" --format json

# Built-in tiny eval suite (10 deterministic cases)
python scripts/run_reference_agent_eval.py --mode eval --format json --min-pass-rate 0.8
```

- Loop steps are always emitted as structured logs: `decide`, `act`, `validate`, `stop`.
- Eval mode exits non-zero when pass rate falls below threshold (`agent_eval_threshold_failed`).

## Configuration

You can tailor indexing and embedding behavior via `.gloggur.yaml` or `.gloggur.json` at the repo root. See the README for supported keys and environment variables.
Gloggur also auto-loads a repo-local `.env` file; exported process environment variables take precedence over `.env`.

## Cache handling

Gloggur stores its cache in `.gloggur-cache`. This directory is **local-only** and should never be committed.
`gloggur status --json` includes `schema_version`, `needs_reindex`, `build_state`, `raw_total_symbols`, and `total_symbols` so agents can distinguish raw on-disk rows from reusable search state without manual cleanup.
Watch mode writes runtime files (`watch_state.json`, `watch.pid`, `watch.log`) under `.gloggur-cache` by default.

Cross-workspace note:
- `gloggur search` operates on the current workspace cache context.
- Before searching in a new workspace, run `gloggur index <workspace> --json` there (or set `GLOGGUR_CACHE_DIR` to that workspace cache).

## Concurrency contract

Gloggur supports concurrent reader/writer usage with explicit guarantees:

- Safe concurrently:
  - multiple readers (`status`, `search`)
  - one writer (`index`, `clear-cache`, watch batch updates) plus any number of readers
- Not allowed concurrently:
  - multiple cache writers in parallel for the same `.gloggur-cache`
  - writers are serialized by a cache write lock at `.gloggur-cache/.cache-write.lock`

Writer lock behavior:

- lock acquisition is non-blocking with bounded retry/backoff
- default policy: `5000ms` total wait, exponential delays (`25ms` initial, capped at `250ms`)
- timeout exits non-zero with structured JSON error (`operation: acquire cache write lock`, `category: cache_lock_held`)
- lock failures can include `holder_pid`, `holder_started_at`, `holder_age_ms`, and `holder_command`
- tunables:
  - `GLOGGUR_CACHE_LOCK_TIMEOUT_MS`
  - `GLOGGUR_CACHE_LOCK_INITIAL_BACKOFF_MS`
  - `GLOGGUR_CACHE_LOCK_MAX_BACKOFF_MS`
  - `GLOGGUR_CACHE_LOCK_BACKOFF_MULTIPLIER`

SQLite behavior under contention:

- cache DB uses `journal_mode=WAL` and `busy_timeout=5000ms`
- read/write contention should not hang indefinitely
- lock contention surfaces as bounded fail/retry outcomes

Publication consistency:

- index writes are staged under `.gloggur-cache/.builds/<build_id>/`
- the last committed cache remains readable until the staged build publishes successfully
- `status --json` exposes in-progress or interrupted writers through `build_state`
- if a first build is interrupted, `status` reports `resume_reason_codes` including `index_interrupted` / `missing_index_metadata`, and `search --json` fails non-zero with `search_cache_not_ready`
- if a rebuild is interrupted after a healthy cache already exists, readers keep using the last committed cache while `build_state.state="interrupted"` signals the abandoned staged build
