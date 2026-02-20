# Agent Integration Guide

This repository ships Gloggur, a symbol-level indexer intended for coding agents. If you are an agent working on this project, you **must** use Gloggur to understand and navigate the codebase.

## Bootstrap and preflight

Before first use in a fresh worktree:

```bash
scripts/bootstrap_gloggur_env.sh
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
   gloggur index . --json
   ```
   Optional one-time setup for background save-triggered indexing:
   ```bash
   gloggur watch init . --json
   gloggur watch start --daemon --json
   ```
2. **Locate relevant code** with semantic search:
   ```bash
   gloggur search "<query>" --top-k 5 --json
   ```
3. **Confirm coverage**:
   ```bash
   gloggur status --json
   ```
   Watcher runtime health:
   ```bash
   gloggur watch status --json
   ```
4. **Inspect docstrings (optional but recommended for documentation changes)**:
   ```bash
   gloggur inspect . --json
   ```
   Inspection skips unchanged files by default; add `--force` to reinspect everything.

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
