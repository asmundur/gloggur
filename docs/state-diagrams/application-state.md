# Application State

This diagram maps the shared repository-health contract surfaced by
`status --json` and reused by the search entrypoints. It focuses on
`resume_decision`, normalized `build_state`, additive interrupted-build
recovery metadata, and whether semantic retrieval is currently allowed.

| State | Transitions |
| --- | --- |
| `status_cli` | Evaluates cache metadata, profile drift, build state, and integrity markers. |
| `building` | Entered when persisted `build_state.state=building` still points at a live PID. |
| `stale_build_state` | Entered when a recorded `building` PID is dead and the payload is normalized before reuse decisions. |
| `interrupted` | Entered when persisted `build_state.state=interrupted` exists or `stale_build_state` is normalized into an interrupted cleanup case. |
| `interrupted_resume_available` | Additive metadata state when an interrupted staged build has a compatible `resume_candidate`; this does not itself make search ready. |
| `resume_ok` | Reached when the committed canonical cache metadata and resume fingerprints are reusable, even if an abandoned staged rebuild also exists. |
| `reindex_required` | Reached when metadata, build-state, profile, reset, or integrity checks say the cache is not reusable. |
| `semantic_search_allowed=true` | Reached only when `resume_ok` and both search-integrity markers pass. |
| `semantic_search_allowed=false` | Reached whenever reindex or integrity warnings make semantic retrieval unavailable. |

```mermaid
stateDiagram-v2
    [*] --> status_cli

    state "status_cli" as status_cli
    state "building" as building
    state "stale_build_state" as stale_build_state
    state "interrupted" as interrupted
    state "interrupted_resume_available" as interrupted_resume_available
    state "resume_ok" as resume_ok
    state "reindex_required" as reindex_required
    state "semantic_search_allowed=true" as semantic_search_allowed_true
    state "semantic_search_allowed=false" as semantic_search_allowed_false

    status_cli --> building: build_state.state=building
    status_cli --> interrupted: build_state.state=interrupted
    building --> stale_build_state: pid is no longer running
    stale_build_state --> interrupted: normalize state + cleanup_pending=true

    status_cli --> resume_ok: resume_decision=resume_ok
    status_cli --> reindex_required: resume_decision=reindex_required
    building --> reindex_required: resume_reason_codes+=build_in_progress
    interrupted --> interrupted_resume_available: resume_candidate.compatibility.compatible
    stale_build_state --> interrupted_resume_available: resume_candidate.compatibility.compatible
    interrupted --> reindex_required: metadata missing or canonical cache incompatible
    stale_build_state --> reindex_required: metadata missing or canonical cache incompatible
    interrupted --> resume_ok: last committed cache still reusable
    stale_build_state --> resume_ok: last committed cache still reusable

    resume_ok --> semantic_search_allowed_true: search_integrity markers passed
    resume_ok --> semantic_search_allowed_false: search_integrity missing or failed
    reindex_required --> semantic_search_allowed_false
    interrupted_resume_available --> semantic_search_allowed_false: first build still unpublished
```

## Notes

- `stale_build_state` is not stored as-is; it is a normalized health-classifier
  result produced when a saved `building` PID is dead.
- `interrupted_resume_available` is surfaced additively as
  `interrupted_resume_available=true` plus `resume_candidate` metadata with
  `build_id`, `source`, compatibility, and staged counts.
- `semantic_search_allowed=false` is broader than `reindex_required`; it also
  covers integrity failures on otherwise readable caches.
- A healthy committed cache can remain `resume_ok` and searchable while an
  abandoned staged rebuild is recoverable in parallel.
- See [../ERROR_CODES.md](../ERROR_CODES.md) for the error catalog that pairs
  with `resume_reason_codes` and search integrity warnings.
