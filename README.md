# Gloggur

Gloggur is a symbol-level codebase indexer and semantic search tool designed to help agents and developers answer questions about a codebase. It parses source files into symbols, embeds them with language models, stores them locally, and exposes a CLI that returns JSON for easy integration with other tools. Gloggur builds and maintains a local cache of symbol vectors that can be generated on demand and updated incrementally, and it supports pluggable embedding providers (local, OpenAI and Gemini). Its search engine is built on FAISS with a SQLite metadata store and can also audit docstrings to spot discrepancies between documentation and implementation.

## Key features

- Multi-language symbol extraction - uses Tree-sitter parsers to recognise functions, classes, interfaces and other symbols across Python, JavaScript/TypeScript, Go, Rust, Java and more.
- Efficient indexing - initial indexing builds a vector database for your project, and subsequent runs update only changed files so searches stay fast.
- Pluggable embeddings - choose from OpenAI, Gemini or a local hugging-face compatible model. Embedding providers are configured in `.gloggur.yaml` or via environment variables.
- Semantic search with FAISS - quickly retrieve the most similar symbols and view surrounding context; optionally filter to a file or directory prefix and control the amount of context returned.
- Docstring audit - inspect symbols and score how well their docstrings match their implementation.
- On-save indexing (watch mode) - run a background watcher that re-indexes changed files, making search results up-to-date.
- Structured JSON output - all commands return machine-readable JSON envelopes to make it easy to build agents, dashboards and CI jobs around Gloggur.

## Installation

The recommended way to install Gloggur is via `pipx`:

```bash
pipx install gloggur
```

You can also install directly with `pip`:

```bash
pip install gloggur
```

To use specific embeddings providers, install optional extras:

```bash
# OpenAI embeddings
pipx install "gloggur[openai]"

# Gemini embeddings
pipx install "gloggur[gemini]"

# Local model (e.g. CodeBERT)
pipx install "gloggur[local]"
```

For development within the repository clone, run the bootstrap script to create a virtual environment and ensure caches are refreshed:

```bash
scripts/bootstrap_gloggur_env.sh
scripts/gloggur status --json
```

This script checks readiness and can seed the cache/venv from another workspace.

## Quickstart

Index your repository:

```bash
# from the root of your project
gloggur index . --json
```

This parses source files, generates embeddings and stores them in `.gloggur-cache`.

Search for symbols:

```bash
gloggur search "streaming parser" --top-k 5 --json
```

Use `--ranking-mode source-first` to prefer source definitions over test mocks. You can filter results to a path prefix with `--file src/` and adjust the amount of surrounding context with `--context-radius 15`.

Inspect docstrings:

```bash
gloggur inspect . --json  # audit selected paths for docstring fidelity
```

Add `--include-tests` or `--include-scripts` to broaden the inspection scope.

Enable watch mode:

```bash
gloggur watch init . --json   # one-time setup
gloggur watch start --daemon --json  # re-index on file changes
```

Use `gloggur watch status --json` to check the watcher and `gloggur watch stop --json` to stop it.

## Configuration

Gloggur reads configuration from `.gloggur.yaml` or `.gloggur.json` in your project root. A typical configuration might look like:

```yaml
embedding_provider: gemini        # or 'openai' or 'local'
local_embedding_model: microsoft/codebert-base
cache_dir: .gloggur-cache
watch_enabled: false              # set to true to enable background indexing
watch_debounce_ms: 300            # debounce time for file system events
supported_extensions:
  - .py
  - .js
  - .ts
  - .rs
  - .go
  - .java
excluded_dirs:
  - .git
  - node_modules
  - .gloggur-cache
```

Environment variables override configuration settings and are documented in the [full README](https://github.com/asmundur/gloggur/blob/main/README.md) for advanced use cases. Gloggur also loads a `.env` file if present.

## Output schema

Search results are returned as structured JSON with a `query` field, a list of `results` and a `metadata` section describing the search. Each result includes the symbol name, kind, file path, line number, signature, docstring (if present), similarity score, ranking score and a code snippet. This format is stable and suitable for consumption by agents or dashboards.

## Further reading

This document highlights the core concepts and workflows. For detailed error codes, adapter configuration, test harnesses and advanced CLI commands such as artifact publishing or evidence-trace validation, refer to the full documentation in the `docs/` directory of this repository.
