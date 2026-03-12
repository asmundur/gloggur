# Find Command State

This diagram documents `gloggur find`, which reuses the shared search-health
gate but emits the slimmer `find_v1` decision contract.

| State | Transitions |
| --- | --- |
| `preflight_validation` | Resolves query parts, validates `--about`, and checks option compatibility. |
| `search_cache_not_ready` | Shared cache-health error path reused from the search execution layer. |
| `find_v1` | Successful routed retrieval projected into the slim `find_v1` contract. |
| `suppressed` / `no_match` / `decisive` / `ambiguous` | Exact `decision.status` values emitted by the command. |
| `suggested_next_command` | Additive narrowing hint emitted for some `ambiguous` outcomes. |
| `emitted` | Final text, JSON, or NDJSON output step. |

```mermaid
stateDiagram-v2
    [*] --> preflight_validation

    state "preflight_validation" as preflight_validation
    state "search_cache_not_ready" as search_cache_not_ready
    state "find_v1" as find_v1
    state "suppressed" as suppressed
    state "no_match" as no_match
    state "decisive" as decisive
    state "ambiguous" as ambiguous
    state "suggested_next_command" as suggested_next_command
    state "emitted" as emitted

    preflight_validation --> search_cache_not_ready: health.needs_reindex=true
    preflight_validation --> find_v1: query resolved and cache reusable

    find_v1 --> suppressed: decision.status=suppressed
    find_v1 --> no_match: decision.status=no_match
    find_v1 --> decisive: decision.status=decisive
    find_v1 --> ambiguous: decision.status=ambiguous

    ambiguous --> suggested_next_command: decision.suggested_next_command
    suggested_next_command --> emitted: JSON or text includes hint
    suppressed --> emitted
    no_match --> emitted
    decisive --> emitted
    ambiguous --> emitted: no narrowing hint available
    search_cache_not_ready --> emitted: error payload + Exit(1)
```

## Notes

- `find` keeps the shared cache-ready gate but changes the success surface from
  `contextpack_v2` to `find_v1`.
- `stream` is an output-mode switch rather than a decision state, so it is
  treated as part of `emitted` here.
- `suggested_next_command` is optional and only appears on narrowing-friendly
  ambiguous paths.
