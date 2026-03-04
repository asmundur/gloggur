# Search JSON v2 Migration

`gloggur search --json` now emits **ContextPack v2** as a hard break.

## New payload shape

```json
{
  "schema_version": 2,
  "query": "...",
  "summary": { "...": "..." },
  "hits": [
    {
      "path": "src/example.py",
      "span": { "start_line": 10, "end_line": 14 },
      "snippet": "...",
      "score": 0.91,
      "tags": ["literal_match"]
    }
  ]
}
```

Optional:

- `debug` is included only when `--debug-router` is enabled.

## Removed top-level keys

- `results`
- `metadata`

These keys are no longer emitted by the CLI.

## Field mapping (v1 -> v2)

- `metadata.total_results` -> `len(hits)`
- `results[i].file` -> `hits[i].path`
- `results[i].line` -> `hits[i].span.start_line`
- `results[i].line_end` -> `hits[i].span.end_line`
- `results[i].context` -> `hits[i].snippet`
- `results[i].similarity_score` -> `hits[i].score`
- `metadata.search_time_ms` -> `debug.timings.total_ms` (when `--debug-router` is set)

## Behavioral contracts preserved

- deterministic ordering and tie-breaks
- bounded pack size and snippet limits
- bounded numeric scores (`0.0` to `1.0`)
- stable non-zero failure codes for invalid options/runtime errors
- fail-closed behavior when reindex is required

## v1 flags removed from search

These now fail with `search_contract_v1_removed`:

- `--with-evidence-trace`
- `--validate-grounding`
- `--fail-on-ungrounded`

## Recommended upgrade steps

1. Update parsers to consume `schema_version`, `summary`, and `hits`.
2. Enable `--debug-router` in CI/probes that require timing metadata.
3. Remove all logic that assumes top-level `results`/`metadata`.
