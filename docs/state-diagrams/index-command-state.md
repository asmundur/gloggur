# Index Command State

This diagram captures the indexed build flow for `gloggur index`, including
startup resume discovery, staged-build reuse/adoption, stage-level
`completed` / `failed` / `not_run` statuses, and interruption cleanup.

| State | Transitions |
| --- | --- |
| `discover_resume` | Writer lock is held, then interrupted staged builds are inspected before cleanup. |
| `resume_existing` | Compatible manifest-backed staged build is reused in place with its existing build id. |
| `legacy_adopted` | `--adopt-interrupted-build` copies a legacy staged cache under `.builds/` and writes a resume manifest. |
| `fresh_stage` | No compatible interrupted build is available, so a new staged build is prepared from the canonical cache. |
| `building` | Build-state sidecar is written before staged work proceeds. |
| `bootstrap_model` through `commit_metadata` | Ordered stages recorded in the JSON `stages[]` payload. |
| `failed` | Stage-local outcome when categorized failure reasons are recorded. |
| `interrupted` | Build-state sidecar and staged `resume-manifest.json` are rewritten on signal handling or pre-commit failure cleanup. |
| `not_run` | `commit_metadata` terminal status when the build cannot be published cleanly. |
| `completed` | Final successful publish path after staged data is promoted and build state is cleared. |

```mermaid
stateDiagram-v2
    [*] --> discover_resume

    state "discover_resume" as discover_resume
    state "resume_existing" as resume_existing
    state "legacy_adopted" as legacy_adopted
    state "fresh_stage" as fresh_stage
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

    discover_resume --> legacy_adopted: explicit --adopt-interrupted-build
    discover_resume --> resume_existing: manifest candidate compatible
    discover_resume --> fresh_stage: no compatible candidate

    legacy_adopted --> building: write resume manifest + keep adopted stage
    resume_existing --> building: reuse staged cache + build_id
    fresh_stage --> building: prepare_staged_build

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

    failed --> interrupted: write build_state.state=interrupted + resume manifest
    failed --> not_run: commit_metadata status=not_run
    interrupted --> not_run: cleanup_pending=true
    interrupted --> discover_resume: next index run
```

## Notes

- Repository and single-file indexing share the same top-level stage order even
  when some counters differ.
- Successful embedding batches are written to the durable embedding ledger
  immediately during `embed_chunks`, so resumed/retried builds can reuse vectors
  even if the process dies before file-level staged-cache persistence.
- `commit_metadata` is the publish boundary. Success there clears build state
  and persists last-success resume markers.
- The embedding ledger lives outside staged-build promotion, so `clear-cache`
  preserves it unless `--purge-embedding-ledger` is requested.
- `allow_partial` affects the command exit code, not the stage-state model.
