# Agent Integration Guide

This repository ships Gloggur, a symbol-level indexer intended for coding agents. If you are an agent working on this project, you **must** use Gloggur to understand and navigate the codebase.

## Bootstrap and preflight

Before first use in a fresh worktree:

```bash
scripts/bootstrap_gloggur_env.sh
```

Bootstrap now verifies index freshness automatically when `scripts/gloggur` is
present (status -> optional index -> status verification).

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

`scripts/gloggur` preflight behavior:
- use repo `.venv` when healthy
- fallback to system Python + repo `PYTHONPATH` when `.venv` is missing/broken
- if startup is impossible, return deterministic JSON (`operation: preflight`) with:
  - `error_code`: `missing_venv`, `missing_python`, `missing_package`, or `broken_environment`
  - `remediation` guidance and detected runtime details

## Quick workflow

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
   ```
3. **Confirm coverage**:
   ```bash
   scripts/gloggur status --json
   ```
   Watcher runtime health:
   ```bash
   scripts/gloggur watch status --json
   ```
4. **Inspect docstrings (optional but recommended for documentation changes)**:
   ```bash
   scripts/gloggur inspect . --json
   ```
   Inspection skips unchanged files by default; add `--force` to reinspect everything.

Pytest defaults for this repo:

- tests run in parallel (`-n auto --dist=loadscope`)
- serial override for debugging: `pytest -n 0`
- HTML coverage report is opt-in: `pytest --cov-report=html`

## Tips for effective searches

- Search by **concepts**, not just filenames (e.g., "incremental hashing", "embedding provider", "tree-sitter parser").
- Use `--top-k` to widen or narrow results based on the task.
- Use `--stream` if you are integrating results into a tool chain.

## Configuration

You can tailor indexing and embedding behavior via `.gloggur.yaml` or `.gloggur.json` at the repo root. See the README for supported keys and environment variables.

## Cache handling

Gloggur stores its cache in `.gloggur-cache`. This directory is **local-only** and should never be committed.
`gloggur status --json` includes `schema_version` and `needs_reindex` so agents can detect stale cache state without manual cleanup.
Watch mode writes runtime files (`watch_state.json`, `watch.pid`, `watch.log`) under `.gloggur-cache` by default.

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
- timeout exits non-zero with structured JSON error (`operation: acquire cache write lock`)
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

- index metadata is cleared before a rebuild starts
- metadata/profile are published only after vector artifacts are saved
- if a writer is interrupted, `status`/`search` report `needs_reindex=true` instead of advertising a healthy partial state
