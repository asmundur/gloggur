# Simple Command Reference

These commands and groups intentionally do not get separate lifecycle diagrams.
Their stable public surface is primarily success/failure plus machine-readable
error codes, not a long-lived state machine.

| Command or group | Kind | Stable surface |
| --- | --- | --- |
| `init` | Config-mutating | Writes repo-local support/runtime config and returns `initialized=true` or a typed failure. |
| `status` | Read-only | Wrapper over the shared health model in [application-state.md](application-state.md); the command itself is request/response. |
| `extract` | Read-only | Byte-range extraction with path/range validation and deterministic success/error payloads. |
| `inspect` | Read-only | Audit run with success/failure summaries and warning/error codes, but no separate long-lived lifecycle. |
| `clear-cache` | Maintenance-oriented | Cache reset operation with a simple success/failure boundary. |
| `guidance` | Read-only | Context-generation wrapper without persistent command-state transitions. |
| `adapters list` | Read-only | Registry inspection with a single response payload. |
| `parsers check` | Read-only | Capability report that may exit non-zero, but does not maintain runtime state. |
| `coverage *` | Maintenance-oriented | Import/ingest commands that validate inputs, mutate coverage mappings, and then terminate. |
| `graph *` | Read-only | Query-style graph traversal/search commands with request/response semantics only. |
| `support *` | Maintenance-oriented | Bundle/run helpers whose stable interface is artifact creation plus error codes, not a reusable lifecycle state machine. |

## Notes

- `watch init` participates in the watch feature but is still configuration
  setup rather than a long-lived runtime state; the runtime is documented in
  [watch-command-state.md](watch-command-state.md).
- The error catalog remains centralized in [../ERROR_CODES.md](../ERROR_CODES.md).
