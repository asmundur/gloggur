# Adapter Architecture

Gloggur supports adapter-driven extension points for parser, coverage import, embedding providers, storage backends, and runtime hosts.

## Discoverability

```bash
gloggur adapters list --json
```

The payload includes:

- active adapter ids (`embedding_provider`, `storage_backend`, `runtime_host`)
- discoverable adapters by category (`parsers`, `coverage_importers`, `embedding_providers`, `storage_backends`, `runtime_hosts`)

## Config

Use additive config keys:

```yaml
adapters:
  parsers:
    my_parser: mypkg.parsers:create_parser
  coverage_importers:
    my_coverage: mypkg.coverage:create_importer
  embedding_providers:
    my_embedder: mypkg.embed:create_provider
  storage:
    my_storage: mypkg.storage:create_backend
  runtime:
    my_runtime: mypkg.runtime:create_host

storage:
  backend: sqlite_faiss

runtime:
  host: python_local

parser_extension_map:
  .pyi: python
```

`adapters.<category>.<adapter_id>` values are optional module overrides (`module:callable`). If omitted, Glöggur resolves built-in adapters first, then installed Python entry points.

## Entry Point Groups

- `gloggur.parsers`
- `gloggur.coverage_importers`
- `gloggur.embedding_providers`
- `gloggur.storage_backends`
- `gloggur.runtime_hosts`

## Compatibility

- Existing commands and defaults remain intact.
- `coverage import-python` is preserved.
- New generic coverage path: `coverage import --importer <id>`.
