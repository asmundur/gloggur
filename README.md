# Gloggur

**Gloggur** is a self‑contained indexing and retrieval engine for codebases.  It is built as a *search, discovery and exploration* tool rather than a general agent framework.  Gloggur parses your repository into language‑agnostic symbols and chunks, builds a local metadata and vector store, and exposes a simple CLI that emits machine‑readable JSON for integration with agents or dashboards.

The current design emphasises **lexical candidate generation** backed by **semantic reranking**.  Gloggur still supports full vector search, but it treats grep/ripgrep queries and fully‑qualified names as first‑class citizens.  Search results are chunk‑aware, deterministic and enriched with parser metadata so that you can not only *find* code but also *explore* its relationships.

## Core concepts

### Chunk‑first indexing

Gloggur extracts symbols using Tree‑sitter parsers for Python (`.py`), JavaScript (`.js`, `.jsx`), TypeScript (`.ts`, `.tsx`), Rust (`.rs`), Go (`.go`), and Java (`.java`). Each symbol’s definition and docstring are split into one or more **chunks**, and each chunk is assigned a stable, hashed ID. The indexer persists symbols, chunks and edges into a local SQLite/FAISS cache. Incremental indexing updates only modified files, keeping search fast without reprocessing the entire project.

### Search & discovery

The CLI provides a single `search` command with multiple modes:

* **Auto** (default) – routes queries to either lexical search (ripgrep), exact name search or semantic search based on heuristics.
* **by_fqname** – search by fully‑qualified symbol name (e.g. `pkg.module.Class.method`).
* **by_path** – search by file path or directory prefix.
* **grep / rg** – pipe a raw grep/ripgrep query and receive ranked results.
* **semantic** – perform full vector search over the chunk index.

Regardless of mode, search produces a candidate set and optionally reranks it using embeddings.  Each hit includes `chunk_id`, `symbol_id`, file path and line range, along with a snippet of code and any matching tags (such as `symbol_def`/`symbol_ref`).

### Exploration

Gloggur goes beyond search by providing a **reference graph**.  The `graph` command exposes:

* `neighbors <symbol-id>` – get all outgoing edges for a symbol.
* `incoming <symbol-id>` – list callers or importers.
* `outgoing <symbol-id>` – list callees or referenced definitions.
* `search <natural-language>` – retrieve symbols connected through graph edges that best match a description.

Edges have deterministic IDs and types (`CONTAINS`, `DEFINES`, `IMPORTS`, `CALLS`, `REFERENCES`, `TESTS`) to help you traverse and visualise how code elements relate.  This is especially useful when planning refactors or porting subsystems to another language.

### Semantic reranking

Embeddings are a core part of Gloggur.  They are used to rerank lexical results and power semantic search when appropriate.  The engine supports OpenAI, Gemini or any hugging‑face compatible local model via configuration.  If embeddings aren’t available, the system falls back to pure lexical ranking and still returns meaningful results.

### Docstring audit

The `inspect` command analyses docstrings and implementation to identify mismatches.  It flags missing descriptions, inaccurate parameter documentation and other quality issues.  Combined with search and graph, this helps ensure rewritten or refactored code preserves behaviour.

## Quickstart

Install via `pipx` (recommended):

```bash
pipx install gloggur
# optional: pipx install "gloggur[openai]"  # or [gemini], [local]
```

Index your repository from the project root:

```bash
gloggur index . --json
```

Run searches in different modes:

```bash
# natural language
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

Place a `.gloggur.yaml` or `.gloggur.json` file in your project root to customise embedding providers, cache location, watch settings and supported file extensions. By default, minified JavaScript (`*.min.js`) is excluded from index/watch runs to avoid noisy vendor artifacts; set `include_minified_js: true` when you explicitly need those files indexed. Environment variables can override any option.  Here is a minimal example:

```yaml
embedding_provider: openai       # or 'gemini' or 'local'
cache_dir: .gloggur-cache
watch_enabled: false
include_minified_js: false       # set true to include `*.min.js` files
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
  - .gloggur-cache
```

## Output schema

Commands emit JSON structures that are easy to consume programmatically.  Search results include:

* `schema_version` – version of the JSON schema.
* `query` – the original query string.
* `summary` – optional summary of top matches.
* `hits[]` – list of candidate chunks with fields:

  * `path` – repo-relative path suitable for `gloggur extract`.
  * `span.start_line` / `span.end_line` – inclusive logical line span.
  * `start_byte` / `end_byte` – raw file byte offsets (`end_byte` exclusive).
  * `snippet`, `score`, and `tags` (`literal_match`, `semantic_match`, `symbol_def`, `symbol_ref`).
* `debug` – when `--debug-router` is provided, includes routing decisions, candidate counts and backend errors.

Byte-range extraction is available without a fresh index once you already have a hit path/span:

```bash
gloggur search "Foo" --json
gloggur extract sample.py 0 42
gloggur extract sample.py 0 42 --json
```

`extract` requires repo-relative paths under the active workspace root and reads exact raw bytes before decoding with UTF-8 replacement.

Graph commands return `edge_id`, `edge_type`, `from_id`, `to_id`, `file_path`, `line` and `confidence`.  Inspect commands return audit findings per symbol with categories such as missing docstring, parameter mismatch and summary quality.

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

## Current status

Gloggur’s chunk‑first architecture and graph retrieval surfaces landed in early March 2026.  The semantic reranking path is functional but the full‑corpus semantic search is still being tuned; hybrid search is the recommended mode until embedding pipelines are finalised.  Additional work on ranking heuristics and evaluation harnesses is underway, and future releases will continue to enhance search accuracy and exploration capabilities.
