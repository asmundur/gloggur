# Agent Integration Guide

This repository ships Gloggur, a symbol-level indexer intended for coding agents. If you are an agent working on this project, you **must** use Gloggur to understand and navigate the codebase.

## Bootstrap and preflight

Before first use in a fresh worktree:

```bash
scripts/bootstrap_gloggur_env.sh
```

Bootstrap now verifies index freshness automatically when `scripts/gloggur` is
present (status -> optional index -> status verification), then runs the
canonical startup readiness probe. In this repo it also bootstraps Beads task
reads for fresh clones by creating a local Dolt database from the tracked JSONL
export when needed:

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
- Fresh clones are `bootstrap_required`: they ship tracked `.beads/issues.jsonl`
  and `.beads/clone-contract.json`, but not the local `.beads/dolt` database.
- Run `scripts/bootstrap_gloggur_env.sh` before any `bd ... --json` read in a
  fresh clone.
- Manual repair path:
  ```bash
  bd init -p bd --json
  bd import -i .beads/issues.jsonl --json
  bd status --json
  ```
- Machine consumers should read `.beads/clone-contract.json` to discover the
  clone contract instead of inferring readiness from `.beads/metadata.json`.

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
   `index` now auto-resumes a compatible interrupted staged build instead of
   discarding it and starting over. For a pre-manifest legacy staged cache that
   was stranded before this recovery flow shipped, explicitly adopt it once:
   ```bash
   scripts/gloggur index . --json --adopt-interrupted-build /path/to/staged-build
   ```
   Optional one-time repo setup for betatester support bundles:
   ```bash
   scripts/gloggur init . --betatester-support --yes --json
   ```
   `gloggur init .` is optional and repo-local. It does not enable watch mode.
   It now plans repo access and, with `--yes`, applies only Glöggur-local permission
   fixes while surfacing any remaining repo or macOS privacy blockers honestly.
   Optional one-time setup for background save-triggered indexing:
   ```bash
   scripts/gloggur watch init . --json
   scripts/gloggur watch start --daemon --json
   ```
   `watch init` remains watch-specific.
2. **Locate relevant code** with `find` or `search`:
   ```bash
   scripts/gloggur find "<query>"
   scripts/gloggur find "<query>" --json
   scripts/gloggur find make_response src/flask/app.py tests/test_app.py --json
   scripts/gloggur find "pkg.module.Handler.handle" --search-mode by_fqname --json
   scripts/gloggur find "src/services" --search-mode by_path --json
   scripts/gloggur find rg -S -g '*.py' AuthToken src/ --about "cache warmup startup state" --json
   scripts/gloggur search "<query>" --top-k 5 --json
   scripts/gloggur search "rg -S -g '*.py' AuthToken" --json
   ```
   Prefer `find` for the first retrieval pass in agent loops. Drop to `search --json`
   when you need exact byte offsets, full router/debug metadata, or the full
   ContextPack v2 contract.
   If the cache is not reusable, `search --json` exits non-zero with
   `error.code=search_cache_not_ready` and a top-level `metadata` object that
   includes resume/build-state details. Interrupted-build recovery is surfaced
   additively through `interrupted_resume_available` and `resume_candidate`;
   search still stays unavailable until the staged build publishes.
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
   Directory inspections focus on source paths by default; add `--include-tests`
   and `--include-scripts` when you need full repo coverage. Inspection skips
   unchanged files by default; add `--force` to reinspect everything.
   To diagnose unsupported extension skips explicitly:
   ```bash
   scripts/gloggur inspect . --json --warn-on-skipped-extensions
   ```

5. **Validate parser capability contract (optional, useful before language-heavy work)**:
   ```bash
   scripts/gloggur parsers check --json
   ```
   This reports the live support contract plus confirmed known gaps, which is
   useful before relying on JS/TS/TSX/Go/Rust/Java symbol fidelity for a task.

## Support Bundles

Use the support CLI when a field tester needs a sendable diagnostics bundle
after anything odd, including hangs or commands that are still running.

One time per repo, enable richer support tracing:

```bash
scripts/gloggur init . --betatester-support --yes --json
```

Normal Glöggur commands still work without repo init. This only enables richer
support capture for later `support collect` runs.

If you need to inspect or re-run the access flow separately:

```bash
scripts/gloggur access plan . --json
scripts/grant_gloggur_access.sh . --yes --json
```

Create a bundle after the problem happens:

```bash
scripts/gloggur support collect --json --note "manual support snapshot"
```

Support bundle paths:
- sessions: `.gloggur/support/sessions/<session-id>/`
- bundles: `.gloggur/support/bundles/gloggur-support-<session-id>.tar.gz`

Bundle policy:
- sanitized by default: secrets and local absolute paths are redacted
- runtime logs/status/config snapshots are included automatically
- in betatester-support repos, recent command traces and active running-command traces are included automatically
- when a Glöggur command is still running, `support collect` requests a live Python stack dump when supported
- cache/index databases are auto-included only when active or recent evidence points at index/cache trouble; otherwise they stay out of the bundle unless `--include-cache` is passed

Fallback behavior:
- without `--betatester-support`, `support collect` still creates a smaller current-state snapshot
- `support run` remains available for advanced/internal repros, but it is not the primary field-tester workflow

Subject repo coverage note:
- For subject repo analysis, test/coverage collection should run in the subject repo's own environment/toolchain (its own venv/runner), not in this Gloggur repo environment.
- After coverage is produced there, run `gloggur coverage import ...` from the subject repo cwd (or pass explicit coverage artifact paths).

Pytest defaults for this repo:

- tests run in parallel (`-n auto --dist=loadscope`)
- serial override for debugging: `pytest -n 0`
- HTML coverage report is opt-in: `pytest --cov-report=html`

## Tips for effective searches

- Use `find` first when you want the shortest viable answer for an agent loop.
- Use `find --json` or `find --stream` when you need a slim, low-token machine-readable result set with exact byte offsets.
- By default, decisive `find --json` and `find --stream` responses emit one top hit. Pass `--top-k` or `--max-snippets` explicitly when you want a wider hit list on decisive outcomes.
- `find` accepts grep-like token sequences directly. One or more trailing existing file/directory operands are treated as implicit scope filters.
- Mixed existing/missing trailing scope operands now fail closed with a structured error; fix the paths or make scope explicit with `--file` / `--path-prefix`.
- Use `find --about "<semantic description>"` when you already know the lexical pattern but want bounded semantic disambiguation. It reranks the lexical candidates from the same invocation and only falls back to a single semantic code rescue when lexical lookup is empty or auxiliary-only.
- If a literal query token collides with a Glöggur option name such as `--about`, quote it or place it after `--`.
- Use `search --json` when you need exact extraction offsets, resume/build-state metadata, or full router/debug payloads.
- Search by **concepts**, not just filenames (e.g., "incremental hashing", "embedding provider", "tree-sitter parser").
- Use `--top-k` to widen or narrow results based on the task.
- Use `--stream` if you are integrating results into a tool chain.
- For identifier-heavy lookups where tests are noisy, use `--ranking-mode source-first`.
- Grep/ripgrep style queries are accepted as input compatibility (`rg Foo`, `rg -g "*.py" Foo`, `grep -R foo_bar src/`).
- Quoted short grep patterns are preserved in router debug (`parsed_query.pattern_quoted=true` for queries like `rg "id"`).
- Router contract boundary: callers provide intent-only routing fields; grep/ripgrep flags are parsed into internal execution hints and surfaced in `debug.constraints`.
- Use `--file` for exact file or directory-prefix scoping (`exact_or_prefix` with
  normalized `src/` vs `./src/` behavior); check
  `metadata.file_filter_warning_codes` for `file_filter_no_match`.
- Use `--context-radius` (default `12`, range `1..200`) when agents need more or
  less surrounding implementation detail per hit.
- Symbol-backed hits can include `symbol_def` and `symbol_ref` tags; if `.gloggur/index/symbols.db`
  is missing, search still runs but symbol-tagged hits are absent until reindex.
- Search hits now expose repo-relative `path` plus additive `start_byte` / `end_byte`
  fields so agents can round-trip exact source text with `gloggur extract`.
- `find --json` responses now include additive `decision.target` when the next step is obvious. Prefer that machine-readable open/scope target over inferring from the hit list.
- Ambiguous `find --json` responses can still include `decision.suggested_next_command`; prefer that direct narrowing step over broad follow-up probes.
- Missing/corrupt symbol index stays non-fatal: use `--debug-router` and inspect
  `debug.backend_errors.symbol` for deterministic diagnostics.
- Semantic fallback failures also stay structured: when lexical hits survive, `find`/`search` report backend errors instead of surfacing a raw traceback.

Search -> extract reference flow:

```bash
scripts/gloggur search "where is Foo defined" --json
scripts/gloggur extract sample.py 0 42 --json
```

- `hits[].path` is repo-relative and safe to feed directly into `extract`.
- `start_byte` is inclusive and `end_byte` is exclusive everywhere.
- `extract` rejects absolute or escaping paths and does not require a fresh index if you
  already have a valid path/span.

Grounding note for agent outputs:

- Legacy `search` grounding flags from the pre-ContextPack flow were removed.
  `--with-evidence-trace`, `--validate-grounding`, and `--fail-on-ungrounded`
  now fail closed with `error.code=search_contract_v1_removed`.
- For caller-side grounding today, branch on `summary.strategy`,
  `summary.reason`, `summary.next_action`, and hit count. When you need router
  confidence detail, rerun with `--debug-router` and inspect backend thresholds
  in the returned payload before deciding whether to emit, repair, or block.

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
By default, `*.min.js` files are skipped during index/watch; set `include_minified_js: true` (or `GLOGGUR_INCLUDE_MINIFIED_JS=true`) when you need vendored/minified JavaScript indexed.

## Cache handling

Gloggur stores its cache in `.gloggur-cache`. This directory is **local-only** and should never be committed.
`gloggur status --json` includes `schema_version`, `needs_reindex`, `build_state`, `raw_total_symbols`, and `total_symbols` so agents can distinguish raw on-disk rows from reusable search state without manual cleanup.
Interrupted-build recovery metadata is also surfaced additively through
`interrupted_resume_available` plus `resume_candidate` (`build_id`, `source`,
compatibility, and staged counts).
Successful embedding batches are deduplicated in the durable ledger
`.gloggur-cache/embedding-ledger.db`, keyed by canonical embedding profile plus
content hash, so retries and resumed builds do not repay for already-returned
vectors.
`gloggur clear-cache --json` preserves that ledger by default; pass
`--purge-embedding-ledger` only when you intentionally want to discard cached
embedding vectors and force future re-embedding.
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
- startup discovers compatible interrupted staged builds before cleanup; matching
  manifest-backed builds are resumed in place and unrelated stale stages are the
  only ones deleted automatically
- legacy interrupted staged cache directories can be adopted once with
  `gloggur index . --json --adopt-interrupted-build <stage-dir>` so compatible
  pre-update work is not thrown away
- `status --json` exposes in-progress or interrupted writers through `build_state`; stale dead-PID in-progress markers are reclassified to interrupted semantics and surfaced via `stale_build_state`
- `status --json` and `search --json` also surface `interrupted_resume_available`
  plus `resume_candidate` when an interrupted staged build is compatible with the
  current workspace/profile/schema
- if a first build is interrupted, `status` reports `resume_reason_codes`
  including `index_interrupted` / `missing_index_metadata`, and `search --json`
  still fails non-zero with `search_cache_not_ready`; the remediation now
  explicitly says that resume is available when the staged build can continue
- if a rebuild is interrupted after a healthy cache already exists, readers keep using the last committed cache while `build_state.state="interrupted"` signals the abandoned staged build
- embedding results are also written into the durable ledger immediately after
  provider success, so a crash between provider return and staged-cache commit
  can still reuse those vectors on the next run
