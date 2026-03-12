# Quickstart

This guide is the single-path operator flow for first use.

## Install and Bootstrap

```bash
scripts/bootstrap_gloggur_env.sh
scripts/gloggur status --json
```

If `status --json` returns `"resume_decision": "reindex_required"`, run:

```bash
scripts/gloggur index . --json
```

Default indexing embeds symbol chunks only. Graph edges are still extracted and
stored for traversal, but `embedded_edge_vectors` stays `0` unless you opt in
with `scripts/gloggur index . --json --embed-graph-edges` or
`GLOGGUR_EMBED_GRAPH_EDGES=true`.

`status --json` now separates raw on-disk rows from reusable search state:

- `build_state`: active/incomplete build metadata when a writer is building or was interrupted
- `raw_total_symbols`: raw symbol rows observed on disk
- `total_symbols`: searchable symbol count, forced to `0` whenever `resume_decision != "resume_ok"`
- `index_stats`: persisted symbol/chunk/graph totals plus embedded vector totals

Workspace note:
- cache/state defaults to `.gloggur-cache` in the current workspace
- when working from a different cwd/repo, run `gloggur index <that-workspace> --json` first
- or set `GLOGGUR_CACHE_DIR` explicitly

## Provider Setup

Local provider is the default and requires a working local sentence-transformers model:

```bash
export GLOGGUR_EMBEDDING_PROVIDER=local
# optional override:
export GLOGGUR_LOCAL_MODEL="microsoft/codebert-base"
```

### OpenAI

#### Via OpenRouter (preferred)

```bash
export GLOGGUR_EMBEDDING_PROVIDER=openai
export OPENROUTER_API_KEY="<your-openrouter-key>"
export GLOGGUR_OPENAI_MODEL="text-embedding-3-large"
# optional endpoint override (defaults to https://openrouter.ai/api/v1 when OPENROUTER_API_KEY is set)
export GLOGGUR_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
# optional attribution headers sent only on OpenRouter endpoints
export GLOGGUR_OPENROUTER_SITE_URL="https://example.com"
export GLOGGUR_OPENROUTER_APP_NAME="gloggur"
```

#### Direct (backward-compatible)

```bash
export GLOGGUR_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY="<your-openai-key>"
export GLOGGUR_OPENAI_MODEL="text-embedding-3-large"
# optional custom OpenAI-compatible endpoint
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

If credentials or model setup are wrong, commands fail non-zero with:

- `error.type: embedding_provider_error`
- `error.provider: openai`

### Gemini

```bash
export GLOGGUR_EMBEDDING_PROVIDER=gemini
export GEMINI_API_KEY="<your-gemini-key>"
# optional alternate env key name:
export GOOGLE_API_KEY="<your-gemini-key>"
export GLOGGUR_GEMINI_MODEL="gemini-embedding-001"
```

If credentials or model setup are wrong, commands fail non-zero with:

- `error.type: embedding_provider_error`
- `error.provider: gemini`

## First Run (Index, Watch, Search, Inspect)

Run commands from repo root:

```bash
scripts/gloggur index . --json
# opt in if you explicitly want semantic edge vectors too
scripts/gloggur index . --json --embed-graph-edges
scripts/gloggur watch init . --json
scripts/gloggur watch start --daemon --json
scripts/gloggur watch status --json
scripts/gloggur search "add numbers token" --top-k 5 --json
scripts/gloggur inspect . --json
scripts/gloggur watch stop --json
```

Optional repo-local setup for betatester support bundles:

```bash
scripts/gloggur init . --betatester-support --json
```

`gloggur init .` writes a minimal repo-local Glöggur config scaffold. Add
`--betatester-support` when you want later `support collect` runs to include
recent and active command traces. `gloggur watch init . --json` stays
watch-specific and only configures watch mode.

`search --json` emits ContextPack v2 fields (`schema_version`, `query`, `summary`, `hits[]`) only when the cache is ready.
When the cache is not reusable, `search --json` exits non-zero with:

- `error.type: search_unavailable`
- `error.code: search_cache_not_ready`
- `error.category: cache_not_ready`
- top-level `metadata` including resume fields, `build_state`, and `search_integrity`

## Current gotchas

- `scripts/gloggur inspect . --json` focuses on source paths by default; add `--include-tests` and `--include-scripts` when you want a broader repo audit.
- Parser support is baseline rather than uniform. Run `scripts/gloggur parsers check --json` before depending on symbol fidelity for TypeScript type aliases or enums, named Go types, Rust impl methods, or Java records/enums.
- After `scripts/gloggur init . --json` or `scripts/gloggur watch init . --json`, later commands may include `security_warning_codes: ["untrusted_repo_config"]` because the repo-local config is auto-discovered and treated as untrusted by default.
- This repo's `scripts/run_quickstart_smoke.py` harness and most deterministic CI checks use `GLOGGUR_EMBEDDING_PROVIDER=test`; they do not validate first-run local model bootstrap.

## If Something Goes Wrong

Use the support tool when a field tester sees anything odd: failures, hangs,
wrong results, or unusually slow commands.

1. One time per repo, enable betatester support tracing:

```bash
scripts/gloggur init . --betatester-support --json
```

This is optional. Normal `index`, `search`, `status`, and `watch` commands still
work without repo init.

2. After anything weird, collect a bundle:

```bash
scripts/gloggur support collect --json --note "index hung for several minutes"
```

3. Send the generated `.tar.gz` file from `.gloggur/support/bundles/`.

What `support collect` does for you:
- always creates a support session under `.gloggur/support/sessions/`
- always copies Glöggur logs, watch state, bootstrap state, current diagnostics, and config summaries
- in betatester-support repos, also bundles recent completed command traces and traces from Glöggur commands that are still running
- when a Glöggur command is still running, requests a live Python stack dump from that process when the platform supports it
- redacts obvious secrets and local absolute paths by default
- auto-includes raw cache/index artifacts when active or recent evidence points at index/cache trouble; otherwise keeps the bundle sanitized

If betatester support was not enabled first, `support collect` still creates a
smaller snapshot bundle from current logs/state, but it cannot recover past
live-command traces that were never recorded.

`support run` still exists for advanced/internal repro workflows, but field
testers should use `support collect`.

## Troubleshooting by Failure Code

| Failure code | Meaning | Action |
| --- | --- | --- |
| `embedding_provider_error` | Provider client could not initialize or authenticate. | Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` plus model/env settings, then retry. |
| `search_cache_not_ready` | Search was attempted against a cache that is still building, interrupted, or otherwise not reusable. | Run `scripts/gloggur index . --json`, then inspect `status --json` for `build_state` and resume details. |
| `watch_mode_conflict` | Both `--foreground` and `--daemon` were passed. | Use exactly one watch mode flag. |
| `watch_path_missing` | Configured watch path does not exist. | Run `scripts/gloggur watch init <existing-path> --json`. |
| `search_contract_v1_removed` | Deprecated v1-only flags or payload assumptions were used with search JSON v2. | Remove v1 flags and parse ContextPack v2 (`summary`, `hits[]`). |
| `search_router_backends_failed` | Router could not produce usable evidence from enabled backends. | Reindex and rerun with `--debug-router`; adjust mode/filters/time budget. |

For `--json` failures, the top-level contract is:
- `ok=false`
- `error_code`
- `error`
- `stage` (`bootstrap|dispatch|search`)

## Command Reference

- CLI behavior and config keys: `README.md`
- Application and command state diagrams: `docs/state-diagrams/README.md`
- Search JSON migration: `docs/SEARCH_JSON_V2_MIGRATION.md`
- Verification probes and smoke harnesses: `docs/VERIFICATION.md`
- Agent integration conventions: `docs/AGENT_INTEGRATION.md`
