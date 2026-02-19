# DONEs Workflow

`DONEs.md` is the immutable log of completed work.

## Usage Rules

1. Only move tasks here after acceptance criteria are actually met.
2. Keep the same task ID used in `TODOs.md`.
3. Include completion date (`YYYY-MM-DD`), evidence, and verification commands.
4. Record behavioral impact (what changed for users/agents).
5. If follow-up work is discovered during closure, create a new item in `TODOs.md` and link it.
6. Do not rewrite history; append corrections as follow-up entries.

## Completion Entry Template

Copy this section for completed tasks:

```md
## <ID> - <Title>

**Completed On**: YYYY-MM-DD
**Completed By**: <name or agent>
**Source**: moved from TODOs.md

**Delivered**
- <implemented change 1>
- <implemented change 2>

**Verification**
- Commands run:
  - `<command>`
  - `<command>`
- Results:
  - <pass/fail summary>

**Evidence**
- Files: `/abs/path/one`, `/abs/path/two`
- PR/commit/issues: <links or identifiers>

**Follow-ups**
- None
```

## D1 - Document Semantic Search Embedding Inputs

**Completed On**: 2026-02-12
**Completed By**: codex
**Source**: moved from TODOs.md

**Delivered**
- Traced and documented the actual symbol-level chunking pipeline used for semantic search embeddings.
- Added a concrete Markdown artifact for `gloggur/embeddings/base.py` with exact `chunk_text` payloads and all symbol annotations (`id`, `name`, `kind`, lines, `signature`, `docstring`, `body_hash`, `language`, embedding preview).
- Recorded runtime provider details observed in this workspace (`local:microsoft/codebert-base` profile running in deterministic local fallback mode).

**Verification**
- Commands run:
  - `scripts/gloggur index . --json`
  - `scripts/gloggur status --json`
  - `scripts/gloggur search "embedding" --top-k 5 --json`
  - `scripts/gloggur inspect gloggur/embeddings/base.py --json`
  - `.venv/bin/python - <<'PY' ... extract_symbols + Indexer._symbol_text + embed_batch ... PY`
  - `.venv/bin/python - <<'PY' ... create_embedding_provider + get_dimension ... PY`
- Results:
  - Index/status/search commands succeeded.
  - Inspect command succeeded for the example file.
  - Extracted symbol/chunk output matched the example file and documented chunk text.
  - Provider probe confirmed local provider fallback mode and 256-dim vectors for this workspace.

**Evidence**
- Files: `/Users/auzi/vinnustofa/gloggur/docs/semantic-search-embedding-breakdown.md`
- PR/commit/issues: local working tree changes

**Follow-ups**
- None

## D2 - Automate Semantic Embedding Breakdown Report

**Completed On**: 2026-02-12
**Completed By**: codex
**Source**: moved from TODOs.md

**Delivered**
- Added `/Users/auzi/vinnustofa/gloggur/generate_embedding_breakdown.py` to generate semantic-embedding breakdown Markdown for any supported source file.
- Script reports symbol-level chunk text (the exact text sent to embedding), parsed annotations, embedding previews, and provider runtime details (including local fallback marker when present).
- Added CLI options for output path, config path, embedding-provider override, preview length, and optional full vector inclusion.

**Verification**
- Commands run:
  - `scripts/gloggur index . --json`
  - `scripts/gloggur search "symbol text embedding chunk" --top-k 5 --json`
  - `.venv/bin/python -m ruff check generate_embedding_breakdown.py`
  - `.venv/bin/python generate_embedding_breakdown.py gloggur/embeddings/base.py --output /tmp/semantic-search-embedding-breakdown-auto.md`
  - `scripts/gloggur inspect generate_embedding_breakdown.py --json`
- Results:
  - Index refreshed successfully after adding the new script file.
  - Script lint checks passed.
  - Script generated markdown successfully at `/tmp/semantic-search-embedding-breakdown-auto.md`.
  - Generated report contains per-symbol chunks/annotations and runtime embedding metadata.

**Evidence**
- Files: `/Users/auzi/vinnustofa/gloggur/generate_embedding_breakdown.py`
- Files: `/private/tmp/semantic-search-embedding-breakdown-auto.md`
- PR/commit/issues: local working tree changes

**Follow-ups**
- None
