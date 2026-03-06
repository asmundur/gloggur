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
scripts/gloggur watch init . --json
scripts/gloggur watch start --daemon --json
scripts/gloggur watch status --json
scripts/gloggur search "add numbers token" --top-k 5 --json
scripts/gloggur inspect . --json
scripts/gloggur watch stop --json
```

`search --json` emits ContextPack v2 fields (`schema_version`, `query`, `summary`, `hits[]`). Legacy `results` + `metadata` top-level keys are not emitted.

## If Something Goes Wrong

Use the support tool when a field tester needs to send you enough trace data to reproduce a Glöggur failure.

1. Rerun the failing Glöggur command through the support wrapper.

```bash
scripts/gloggur support run -- search "add numbers token" --json
```

Everything after `--` is the normal Glöggur command. Replace `search "add numbers token" --json` with the command that failed for you.

2. If you want to add a short human note, include `--note`.

```bash
scripts/gloggur support run --note "search failed right after indexing" -- search "add numbers token" --json
```

3. If the failure already happened and you just want a snapshot of the current Glöggur state, collect a support bundle directly.

```bash
scripts/gloggur support collect --json --note "manual snapshot after search failure"
```

4. Send the generated `.tar.gz` file from `.gloggur/support/bundles/`.

What the tool does for you:
- creates a support session under `.gloggur/support/sessions/`
- copies Glöggur logs, watch state, bootstrap state, and config summaries
- redacts obvious secrets and local absolute paths by default
- creates a compressed support bundle you can attach to a bug report

Use `--include-cache` only when the smaller sanitized bundle is not enough and you explicitly want to include the runtime cache and index databases.

## Troubleshooting by Failure Code

| Failure code | Meaning | Action |
| --- | --- | --- |
| `embedding_provider_error` | Provider client could not initialize or authenticate. | Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` plus model/env settings, then retry. |
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
- Search JSON migration: `docs/SEARCH_JSON_V2_MIGRATION.md`
- Verification probes and smoke harnesses: `docs/VERIFICATION.md`
- Agent integration conventions: `docs/AGENT_INTEGRATION.md`
