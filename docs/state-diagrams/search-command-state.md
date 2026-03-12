# Search Command State

This diagram documents `gloggur search` from option validation through the
shared cache-health gate, routed retrieval, and final payload emission.

| State | Transitions |
| --- | --- |
| `preflight_validation` | Validates CLI options before any cache or router work begins. |
| `search_cache_not_ready` | Error state emitted when shared health reports `needs_reindex=true`. |
| `routed_retrieval` | Router and backend execution path when the cache is reusable. |
| `search_router_backends_failed` | Error state when the router cannot produce usable evidence. |
| `contextpack_v2` | Successful full-fidelity payload path with `summary` and `hits[]`. |
| `stream` | NDJSON emission path when `--stream --json` is used. |
| `emitted` | Final output step for text, JSON, or NDJSON responses. |

```mermaid
stateDiagram-v2
    [*] --> preflight_validation

    state "preflight_validation" as preflight_validation
    state "search_cache_not_ready" as search_cache_not_ready
    state "routed_retrieval" as routed_retrieval
    state "search_router_backends_failed" as search_router_backends_failed
    state "contextpack_v2" as contextpack_v2
    state "stream" as stream
    state "emitted" as emitted

    preflight_validation --> search_cache_not_ready: health.needs_reindex=true
    preflight_validation --> routed_retrieval: options valid and cache reusable

    routed_retrieval --> search_router_backends_failed: no usable evidence
    routed_retrieval --> contextpack_v2: contract_version=contextpack_v2

    contextpack_v2 --> stream: --stream and --json
    contextpack_v2 --> emitted: text or JSON payload
    search_cache_not_ready --> emitted: error payload + Exit(1)
    search_router_backends_failed --> emitted: failure payload + Exit(1)
    stream --> emitted: NDJSON hits[]
```

## Notes

- Removed v1-only grounding flags fail during `preflight_validation`; they do
  not create a separate search lifecycle.
- `search_cache_not_ready` carries the shared resume/build-state metadata so the
  caller can diagnose why search was gated.
- `contextpack_v2` is the success contract regardless of search mode.
