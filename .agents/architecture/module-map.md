# Module Map

Gloggur is a CLI-first Python package rooted at `src/gloggur`.

## Entrypoints

- `scripts/gloggur`: Repo-local launcher that selects a usable Python interpreter and runs `gloggur.bootstrap_launcher`.
- `src/gloggur/bootstrap_launcher.py`: Preflight and launch planning for the CLI wrapper.
- `src/gloggur/cli/main.py`: Click command surface for index, status, find, search, graph, inspect, watch, support, adapters, coverage, access, and related commands.
- `src/gloggur/__main__.py`: Module entrypoint for `python -m gloggur`.

## Core Data Model

- `src/gloggur/models.py`: Pydantic models for symbols, chunks, file metadata, graph edges, index metadata, and audit metadata.
- `src/gloggur/config.py`: Configuration loading, environment override handling, extension policy validation, and embedding/provider settings.
- `src/gloggur/byte_spans.py`: Repo-relative path normalization and line/byte span helpers for exact extraction.
- `src/gloggur/io_failures.py`: Stable storage I/O error classification and fail-loud messages.

## Indexing Pipeline

- `src/gloggur/indexer/indexer.py`: Main index build orchestration, parser execution, chunking, graph edge construction, embedding, persistence, and resume semantics.
- `src/gloggur/indexer/cache.py`: Cache schema, build manifests, staged build state, integrity markers, and recovery behavior.
- `src/gloggur/indexer/concurrency.py`: Cache write lock acquisition, retry behavior, and lock metadata.
- `src/gloggur/indexer/embedding_ledger.py`: Embedding text hashing and reuse ledger.
- `src/gloggur/indexer/shared.py`: Shared parsed-file and timing types used across index workers.

## Parsing And Language Support

- `src/gloggur/parsers/registry.py`: Parser adapter resolution and extension-to-parser mapping.
- `src/gloggur/parsers/treesitter_parser.py`: Tree-sitter parser implementation for supported source languages.
- `src/gloggur/parsers/support_contract.py`: Published parser support contract and capability checks.
- `src/gloggur/parsers/signal_processors.py`: Language-specific signal extraction helpers.
- `src/gloggur/parsers/coverage.py`: Coverage parser support.

## Search And Graph

- `src/gloggur/search/hybrid_search.py`: Hybrid retrieval over lexical, exact, path, and semantic evidence.
- `src/gloggur/search/router/`: Query hints, routing configuration, backend execution, path priors, telemetry, and router types.
- `src/gloggur/search/evidence.py`: Evidence trace construction and validation.
- `src/gloggur/graph/extractor.py`: Static graph edge extraction from parsed symbols and source references.
- `src/gloggur/graph/service.py`: Graph traversal and graph search service logic.

## Storage And Embeddings

- `src/gloggur/storage/metadata_store.py`: SQLite metadata persistence.
- `src/gloggur/storage/vector_store.py`: FAISS vector persistence and vector/cache integrity checks.
- `src/gloggur/storage/backends.py`: Storage backend adapter construction.
- `src/gloggur/embeddings/`: Local, OpenAI-compatible, Gemini, and deterministic test embedding providers.

## Supporting Features

- `src/gloggur/watch/service.py`: Watch-mode daemon state, file event batching, and incremental indexing triggers.
- `src/gloggur/audit/`: Docstring and code-text audit helpers used by `inspect`.
- `src/gloggur/access.py`: Repo-local access planning and grant persistence for support workflows.
- `src/gloggur/support.py` and `src/gloggur/support_runtime.py`: Support bundle collection and command trace persistence.
- `src/gloggur/adapters/`: Runtime extension contracts and adapter registry helpers.
- `src/gloggur/runtime/hosts.py`: Runtime host adapter support.
