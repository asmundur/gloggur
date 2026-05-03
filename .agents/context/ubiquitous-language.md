# Ubiquitous Language

This file captures repository-specific terms agents should use consistently when working on Gloggur.

## Product Terms

- **Gloggur**: A CLI-first indexing and retrieval engine for codebases.
- **Index**: The persisted local search state built from source files, symbols, chunks, graph edges, metadata rows, and vectors.
- **Cache**: The workspace-local `.gloggur-cache` directory that stores metadata, vector data, build state, support traces, and integrity markers.
- **Support bundle**: A redacted diagnostic archive created by `scripts/gloggur support collect --json`.

## Source Model

- **Symbol**: A parsed source-level unit such as a function, class, method, interface, struct, trait, or comparable language construct.
- **Chunk**: A stable searchable span derived from a symbol definition, docstring, or related source text.
- **Signal**: Lightweight parser metadata attached to a symbol or chunk to describe source features.
- **Edge**: A deterministic graph relationship such as `CONTAINS`, `DEFINES`, `IMPORTS`, `CALLS`, `REFERENCES`, or `TESTS`.
- **Language support contract**: The machine-readable status/parser contract that describes supported extensions, parser mappings, construct tiers, and known gaps.

## Retrieval Terms

- **find**: The terse agent-first retrieval command that emits a decision plus compact hits.
- **search**: The full-fidelity retrieval command that emits ContextPack v2 results and operational metadata.
- **Hybrid search**: Lexical candidate generation with optional semantic reranking.
- **Semantic search**: Vector-backed retrieval over indexed chunks; available only when cache integrity and embeddings are ready.
- **Search router**: The query classifier and backend coordinator that chooses lexical, exact, path, or semantic retrieval paths.

## Operational Terms

- **Bootstrap**: Environment or scaffold hydration. In repo workflow, `/bootstrap` hydrates scaffold values; `scripts/bootstrap_gloggur_env.sh` prepares local tooling.
- **Resume candidate**: An interrupted index build that can be safely resumed.
- **Resume blocked**: A fail-closed state where interrupted build data cannot be safely adopted without explicit operator action.
- **Beads**: The `bd` task tracker. New durable work is tracked there; historical Markdown backlog files remain only as migration evidence.
