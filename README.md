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
then a final `status --json` verification).

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

`gloggur inspect` skips unchanged files by default. Use `--force` to reinspect everything.

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

Clear cache:

```bash
gloggur clear-cache --json
```

Cache compatibility is automatic:
- If cache schema changes, Gloggur rebuilds `.gloggur-cache/index.db` automatically.
- If embedding provider/model changes, the next `gloggur index ...` run rebuilds cache and vectors automatically.

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

Additional verification probes are available for provider/edge/performance checks:

```bash
# Run non-test verification phases (providers, edge cases, performance)
python scripts/run_suite.py

# Run specific phases
python scripts/run_provider_probe.py  # Embedding providers
python scripts/run_edge_bench.py  # Edge cases & performance
```

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
docstring_semantic_threshold: 0.2
docstring_semantic_max_chars: 4000
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
