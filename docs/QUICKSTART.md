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

```bash
export GLOGGUR_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY="<your-openai-key>"
export GLOGGUR_OPENAI_MODEL="text-embedding-3-large"
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

## Troubleshooting by Failure Code

| Failure code | Meaning | Action |
| --- | --- | --- |
| `embedding_provider_error` | Provider client could not initialize or authenticate. | Set the provider API key and model env vars, then retry. |
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
