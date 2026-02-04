# Gloggur

Gloggur is a symbol-level, incremental codebase indexer for semantic search and precedent retrieval. It parses code into symbols, generates embeddings, stores them locally, and exposes a JSON-friendly CLI for integration with AI agents.

## Features

- Tree-sitter parsing for multi-language symbol extraction
- Incremental indexing with SHA-256 hashing
- Pluggable embedding backends (local models or OpenAI)
- FAISS vector search with SQLite metadata
- Docstring validation with semantic similarity scoring
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

Validate docstrings (semantic similarity scoring):

```bash
gloggur validate . --json
```

`gloggur validate` skips unchanged files by default. Use `--force` to revalidate everything.

Check status:

```bash
gloggur status --json
```

Clear cache:

```bash
gloggur clear-cache --json
```

## Validation

Gloggur includes a validation suite to verify functionality:

```bash
# Run all validation tests
python scripts/validate_all.py

# Run specific phases
python scripts/validate_phase1.py  # Smoke tests
python scripts/validate_phase2.py  # Embedding providers
python scripts/validate_phase3_4.py  # Edge cases & performance
```

See `docs/VALIDATION.md` for detailed documentation.
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
