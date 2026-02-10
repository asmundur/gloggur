# Gloggur

Gloggur is a symbol-level, incremental codebase indexer for semantic search and precedent retrieval. It parses code into symbols, generates embeddings, stores them locally, and exposes a JSON-friendly CLI for integration with AI agents.

## Features

- Tree-sitter parsing for multi-language symbol extraction
- Incremental indexing with SHA-256 hashing
- Pluggable embedding backends (local models or OpenAI)
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

Clear cache:

```bash
gloggur clear-cache --json
```

## Verification

Gloggur includes a verification suite to verify functionality:

```bash
# Run all verification tests
python scripts/run_suite.py

# Run specific phases
python scripts/run_smoke.py  # Smoke tests
python scripts/run_provider_probe.py  # Embedding providers
python scripts/run_edge_bench.py  # Edge cases & performance
```

See `docs/VERIFICATION.md` for detailed documentation.
See `docs/AGENT_INTEGRATION.md` for agent integration guidance.

## Configuration

Create `.gloggur.yaml` or `.gloggur.json` in your repository:

```yaml
embedding_provider: gemini
local_embedding_model: microsoft/codebert-base
openai_embedding_model: text-embedding-3-large
gemini_embedding_model: gemini-embedding-001
cache_dir: .gloggur-cache
docstring_semantic_threshold: 0.2
docstring_semantic_max_chars: 4000
supported_extensions:
  - .py
  - .ts
excluded_dirs:
  - node_modules
  - .venv
```

Environment variables:

- `GLOGGUR_EMBEDDING_PROVIDER`
- `GLOGGUR_LOCAL_MODEL`
- `GLOGGUR_OPENAI_MODEL`
- `GLOGGUR_GEMINI_MODEL`
- `GLOGGUR_GEMINI_API_KEY`
- `GLOGGUR_CACHE_DIR`
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
pip install -e ".[dev]"
pytest
```
