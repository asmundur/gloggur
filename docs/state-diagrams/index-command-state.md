# Index Command State

This diagram captures the indexed build flow for `gloggur index`, including the
staged-build lifecycle, stage-level `completed` / `failed` / `not_run`
statuses, and interruption cleanup.

| State | Transitions |
| --- | --- |
| `building` | Build-state sidecar is written before staged work proceeds. |
| `bootstrap_model` through `commit_metadata` | Ordered stages recorded in the JSON `stages[]` payload. |
| `failed` | Stage-local outcome when categorized failure reasons are recorded. |
| `interrupted` | Build-state sidecar is rewritten on signal handling or pre-commit failure cleanup. |
| `not_run` | `commit_metadata` terminal status when the build cannot be published cleanly. |
| `completed` | Final successful publish path after staged data is promoted and build state is cleared. |

```mermaid
stateDiagram-v2
    [*] --> building

    state "building" as building
    state "bootstrap_model" as bootstrap_model
    state "scan_source" as scan_source
    state "extract_symbols" as extract_symbols
    state "embed_chunks" as embed_chunks
    state "persist_cache" as persist_cache
    state "validate_integrity" as validate_integrity
    state "update_symbol_index" as update_symbol_index
    state "commit_metadata" as commit_metadata
    state "failed" as failed
    state "interrupted" as interrupted
    state "not_run" as not_run
    state "completed" as completed

    building --> bootstrap_model
    bootstrap_model --> scan_source
    scan_source --> extract_symbols
    extract_symbols --> embed_chunks
    embed_chunks --> persist_cache
    persist_cache --> validate_integrity
    validate_integrity --> update_symbol_index
    update_symbol_index --> commit_metadata
    commit_metadata --> completed: publish_staged_build + clear_build_state

    extract_symbols --> failed: stage_status=failed
    embed_chunks --> failed: stage_status=failed
    persist_cache --> failed: stage_status=failed
    validate_integrity --> failed: stage_status=failed
    update_symbol_index --> failed: failed>0

    bootstrap_model --> interrupted: SIGINT or SIGTERM
    scan_source --> interrupted: SIGINT or SIGTERM
    extract_symbols --> interrupted: SIGINT or SIGTERM
    embed_chunks --> interrupted: SIGINT or SIGTERM
    persist_cache --> interrupted: SIGINT or SIGTERM
    validate_integrity --> interrupted: SIGINT or SIGTERM
    update_symbol_index --> interrupted: SIGINT or SIGTERM

    failed --> interrupted: write build_state.state=interrupted
    failed --> not_run: commit_metadata status=not_run
    interrupted --> not_run: cleanup_pending=true
```

## Notes

- Repository and single-file indexing share the same top-level stage order even
  when some counters differ.
- `commit_metadata` is the publish boundary. Success there clears build state
  and persists last-success resume markers.
- `allow_partial` affects the command exit code, not the stage-state model.
