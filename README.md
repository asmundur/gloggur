# Gloggur

**Gloggur** is a self-contained indexing and retrieval engine for codebases. It is built as a *search, discovery and exploration* tool rather than a general agent framework. Gloggur parses supported source files into language-agnostic symbols and chunks, builds a local metadata and vector store, and exposes a simple CLI that emits machine-readable JSON for integration with agents or dashboards.

The current design emphasises **lexical candidate generation** backed by **semantic reranking**.  Gloggur still supports full vector search, but it treats grep/ripgrep queries and fully‑qualified names as first‑class citizens.  Search results are chunk‑aware, deterministic and enriched with parser metadata so that you can not only *find* code but also *explore* its relationships.

## Core concepts

### Chunk‑first indexing

Gloggur extracts baseline symbol coverage using Tree-sitter parsers for Python (`.py`), JavaScript (`.js`, `.jsx`), TypeScript (`.ts`, `.tsx`), C (`.c`, `.h`), C++ (`.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh`, `.hxx`), Rust (`.rs`), Go (`.go`), and Java (`.java`). Each symbol's definition and docstring are split into one or more **chunks**, and each chunk is assigned a stable, hashed ID. The indexer persists symbols, chunks and edges into a local SQLite/FAISS cache. Incremental indexing updates only modified files, keeping search fast without reprocessing the entire project.

### Search & discovery

The CLI exposes two retrieval surfaces:

* **find** – a terse, agent-first entrypoint that returns a short decision plus the best hits in plain text by default, or a slim `find_v1` JSON/NDJSON contract for tool pipelines.
* **search** – the full-fidelity retrieval surface with ContextPack v2 output and the complete operational metadata/debug contract.

`search` supports multiple modes:

* **Auto** (default) – routes queries to either lexical search (ripgrep), exact name search or semantic search based on heuristics.
* **by_fqname** – search by fully‑qualified symbol name (e.g. `pkg.module.Class.method`).
* **by_path** – search by file path or directory prefix.
* **grep / rg** – pipe a raw grep/ripgrep query and receive ranked results.
* **semantic** – perform full vector search over the chunk index.

Regardless of mode, search produces a candidate set and optionally reranks it using embeddings.  Each hit includes `chunk_id`, `symbol_id`, file path and line range, along with a snippet of code and any matching tags (such as `symbol_def`/`symbol_ref`).

### Exploration

Gloggur goes beyond search by providing a **reference graph**.  The `graph` command exposes:

* `neighbors <symbol-id>` – inspect incoming and outgoing edges for a symbol (bidirectional by default).
* `incoming <symbol-id>` – list callers or importers.
* `outgoing <symbol-id>` – list callees or referenced definitions.
* `search <natural-language>` – retrieve symbols connected through graph edges that best match a description.

Edges have deterministic IDs and types (`CONTAINS`, `DEFINES`, `IMPORTS`, `CALLS`, `REFERENCES`, `TESTS`) to help you traverse and visualise how code elements relate.  This is especially useful when planning refactors or porting subsystems to another language.
Reference-graph edges are always extracted and stored, but they remain structural metadata by default. Index runs embed symbol chunks eagerly and only embed edge text when you opt in with `gloggur index . --json --embed-graph-edges` or `embed_graph_edges: true`.

### Semantic reranking

Embeddings are a core part of Gloggur. They are used to rerank lexical results and power semantic search when appropriate. The engine supports OpenAI, Gemini or any hugging-face compatible local model via configuration. The main `search` surface can degrade to lexical-only retrieval when semantic embeddings are unavailable; `graph search` still requires an embedding provider.

### Docstring audit

The `inspect` command analyses docstrings and implementation to identify mismatches. It flags missing descriptions, inaccurate parameter documentation and other quality issues. Directory audits focus on source paths by default; add the include flags when you also want `tests/` or `scripts/`. Combined with search and graph, this helps ensure rewritten or refactored code preserves behaviour.

## Quickstart

Install via `pipx` (recommended):

```bash
pipx install gloggur
# optional: pipx install "gloggur[openai]"  # or [gemini], [local]
```

Index your repository from the project root:

```bash
gloggur index . --json
# opt in if you explicitly want semantic edge vectors too
gloggur index . --json --embed-graph-edges
gloggur index . --json --verbose
```

Run searches in different modes:

```bash
# agent-first default
gloggur find "how to decode auth token"

# slim structured output for agents
gloggur find "pkg.module.Handler.handle" --json

# natural grep-like narrowing with trailing scope paths
gloggur find make_response src/flask/app.py tests/test_app.py

# identifier-first lookup through find
gloggur find "pkg.module.Handler.handle" --search-mode by_fqname --json

# direct path lookup through find
gloggur find "src/services" --search-mode by_path --json

# combine a ripgrep-style lexical query with bounded semantic disambiguation
gloggur find rg -S -g '*.py' AuthToken src/ --about "cache warmup startup state" --json

# full-fidelity JSON
gloggur search "how to decode auth token" --json

# fully‑qualified name
gloggur search "pkg.module.Handler.handle" --search-mode by_fqname --json

# path / directory prefix
gloggur search "src/services" --search-mode by_path --json

# grep / ripgrep pass‑through (regex)
gloggur search "rg -i \"retry\" src/" --json
```

Explore relationships:

```bash
gloggur graph neighbors <symbol-id> --json
gloggur graph incoming <symbol-id> --edge-type CALLS --json
gloggur graph search "who initializes the cache" --json
```

Audit docstrings:

```bash
gloggur inspect . --json
```

## Configuration

Place a `.gloggur.yaml` or `.gloggur.json` file in your project root to customise embedding providers, cache location, watch settings and supported file extensions. By default, minified JavaScript (`*.min.js`) is excluded from index/watch runs to avoid noisy vendor artifacts, and graph edges are stored structurally without semantic embeddings; set `include_minified_js: true` when you explicitly need those files indexed, or `embed_graph_edges: true` (or `GLOGGUR_EMBED_GRAPH_EDGES=true`) when you explicitly want edge vectors. Environment variables can override any option.  Here is a minimal example:

```yaml
embedding_provider: openai       # or 'gemini' or 'local'
cache_dir: .gloggur-cache
watch_enabled: false
embed_graph_edges: false         # set true to restore edge embeddings
include_minified_js: false       # set true to include `*.min.js` files
supported_extensions:
  - .py
  - .js
  - .jsx
  - .ts
  - .tsx
  - .c
  - .h
  - .cpp
  - .cc
  - .cxx
  - .hpp
  - .hh
  - .hxx
  - .rs
  - .go
  - .java
excluded_dirs:
  - .git
  - node_modules
  - .gloggur-cache
```

## Output schema

Commands emit JSON structures that are easy to consume programmatically.

`find --json` returns a slim contract intended for agent loops:

* `schema_version` / `contract_version`
* `query`
* `about` – optional semantic description when `--about` is supplied
* `decision` – status, strategy, query kind, next action, bounded assist mode, an additive `target` for the machine-readable next open/scope step, and an additive `suggested_next_command` when narrowing is needed
* `hits[]` – rank, path, start/end lines, start/end bytes, score, tags, and a trimmed snippet

By default, `find --json` and `find --stream` now emit a single top hit when
`decision.status=decisive`. Pass `--top-k` or `--max-snippets` explicitly when
you want wider result sets even on decisive outcomes.

`find` accepts grep-like token sequences directly. If one or more trailing
positional tokens are existing files or directories, they are treated like
implicit `--file` / `--path-prefix` scope filters automatically. Mixed
existing/missing trailing scope operands now fail closed with a structured
error instead of silently broadening the query. If you need a literal query
token that collides with a Glöggur option name, quote it or place it after `--`.

Use `search --json` when you need the full ContextPack v2 contract. Search results include:

* `schema_version` – version of the JSON schema.
* `query` – the original query string.
* `summary` – optional summary of top matches.
* `hits[]` – list of candidate chunks with fields:

  * `path` – repo-relative path suitable for `gloggur extract`.
  * `span.start_line` / `span.end_line` – inclusive logical line span.
  * `start_byte` / `end_byte` – raw file byte offsets (`end_byte` exclusive).
  * `snippet`, `score`, and `tags` (`literal_match`, `semantic_match`, `symbol_def`, `symbol_ref`).
* `debug` – when `--debug-router` is provided, includes routing decisions, candidate counts, backend errors, and routed lexical/semantic query fields.

`find --json` includes byte offsets for exact agent round-trips while still omitting bulkier success-only metadata such as resume fingerprints and search-integrity payloads. Routed semantic fallback failures are downgraded into backend-error metadata when lexical hits remain available, so agents receive a structured result instead of a raw traceback. Reach for `search --json` when you need the full operational health/debug contract.

Byte-range extraction is available without a fresh index once you already have a hit path/span:

```bash
gloggur search "Foo" --json
gloggur extract sample.py 0 42
gloggur extract sample.py 0 42 --json
```

`extract` requires repo-relative paths under the active workspace root and reads exact raw bytes before decoding with UTF-8 replacement.

`index --json` and `status --json` also include `index_stats`, which separates structural graph counts from embedded vector counts (`graph_edge_count`, `embedded_symbol_vectors`, `embedded_edge_vectors`, `embedded_vector_count`).

`index --json --verbose` adds `verbose.lines.index`, which reports physical source-line totals plus duplicated and unique embedded-line coverage for symbol chunks and optional graph-edge vectors. Inspect-command embeddings remain out of scope for that section.

Graph commands return `edge_id`, `edge_type`, `from_id`, `to_id`, `file_path`, `line` and `confidence`.  Inspect commands return audit findings per symbol with categories such as missing docstring, parameter mismatch and summary quality.

Application and command lifecycle maps live in
[`docs/state-diagrams/README.md`](docs/state-diagrams/README.md).

## Language Support Contract

Gloggur exposes a machine-readable support contract for extensions/languages and parser tiers:

```bash
gloggur status --json
gloggur adapters list --json
```

Parser capability checks against a built-in corpus:

```bash
gloggur parsers check --json
```

To surface unsupported-extension skip diagnostics during indexing/inspect runs:

```bash
gloggur index . --json --warn-on-skipped-extensions
gloggur inspect . --json --warn-on-skipped-extensions
```

## Current gotchas

- Language support is baseline, not uniform. Current known construct gaps include JavaScript computed identifier subscripts and helper-driven runtime mutation, TypeScript type aliases and enums, strict C++ macro recovery limits outside supported placeholder patterns, Go named struct/interface declarations, Rust impl/trait method forms, and Java record/enum declarations. Use `gloggur status --json` or `gloggur parsers check --json` when symbol fidelity matters for a workflow.
- `gloggur inspect . --json` audits source paths by default. Add `--include-tests` and `--include-scripts` when you need a wider repo audit.
- `gloggur init . --betatester-support --yes --json` enables repo-local support tracing and applies Glöggur-local access fixes. Use `gloggur access plan <path>` or `scripts/grant_gloggur_access.sh <path>` when you want to inspect or re-run the access flow separately. `gloggur watch init . --json` stays watch-specific. Either init command writes repo-local config, so later commands in that workspace may report `security_warning_codes=["untrusted_repo_config"]` because auto-discovered repo config is treated as untrusted by default.
- This repo's quickstart smoke and most deterministic CI verification use `GLOGGUR_EMBEDDING_PROVIDER=test`; they do not validate first-run local model bootstrap.

## Current status

Gloggur's chunk-first architecture and graph retrieval surfaces landed in early March 2026. The semantic reranking path is functional but the full-corpus semantic search is still being tuned; hybrid search is the recommended mode until embedding pipelines are finalised. Additional work on ranking heuristics and evaluation harnesses is underway, and future releases will continue to enhance search accuracy and exploration capabilities.
