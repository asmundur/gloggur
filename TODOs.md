# TODOs Workflow

`TODOs.md` is the source of truth for work that is not finished yet.

## Usage Rules

1. Add a task here before starting substantial work.
2. Keep task IDs stable (`R1`, `R2`, `R3`, `F1`, etc.).
3. Use explicit completion criteria (tests, behavior, docs).
4. Update status on every working session (`planned`, `in_progress`, `blocked`, `ready_for_review`).
5. When completed, move the full item to `DONEs.md` with the same ID and completion date.
6. Do not delete historical tasks; if obsolete, mark as `cancelled` with reason.

## Task Template

Copy this section for new tasks:

```md
## <ID> - <Title>

**Status**: planned | in_progress | blocked | ready_for_review | cancelled
**Priority**: P0 | P1 | P2
**Owner**: <name or agent>

**Problem**
- <what is broken or missing>

**Goal**
- <what success looks like>

**Scope**
- <in-scope work item 1>
- <in-scope work item 2>

**Out of Scope**
- <explicitly excluded item>

**Acceptance Criteria**
- <observable success criterion 1>
- <observable success criterion 2>

**Tests Required**
- <unit tests>
- <integration tests>

**Links**
- PR/commit/issues/docs: <links or paths>
```

# Reliability Backlog

These tasks track reliability hardening for cache/index operations after the schema/profile auto-rebuild work.

## R1 - Harden OS-Level Failure Handling (Permissions, Read-only FS, Disk Full)

**Problem**
- Core commands (`index`, `search`, `inspect`, `status`, `clear-cache`) currently assume normal filesystem behavior.
- Real environments can fail with permission denied, read-only mounts, quota errors, or disk full errors.
- When this happens, we need deterministic failure semantics and highly actionable operator guidance.

**Goal**
- Ensure OS-level failures fail loudly, with consistent error classification and remediation guidance.

**Scope**
- Add explicit exception handling around all cache/vector file writes and deletes.
- Classify common `OSError`/`sqlite3` failures into structured categories:
  - `permission_denied`
  - `read_only_filesystem`
  - `disk_full_or_quota`
  - `path_not_writable`
  - `unknown_io_error`
- Ensure JSON command outputs include machine-readable failure payloads where relevant.
- Standardize human-readable stderr messaging to include:
  - failing path
  - operation attempted
  - probable cause
  - short remediation steps
- Ensure non-zero exits for hard failures.

**Acceptance Criteria**
- For each category above, command exits non-zero and prints an actionable message.
- Error output never hides the original exception detail.
- Messages are stable enough for CI parsing (no random formatting drift).
- Existing successful flows remain unchanged.

**Tests Required**
- Unit tests that simulate `OSError`/`sqlite3.OperationalError` via monkeypatch/mocks for each category.
- Fast integration tests using temp dirs with restricted permissions (where portable).
- Assertions on:
  - exit code
  - JSON/error payload fields
  - stderr content and remediation hints

---

## R2 - Recover Gracefully from Corrupted SQLite Cache Files

**Problem**
- Current schema auto-reset handles many incompatibilities, but severe DB corruption requires explicit recovery guarantees.
- Corruption can occur from abrupt process termination, external tampering, or partial writes.

**Goal**
- Guarantee deterministic, automatic recovery from corrupted cache DB state without manual cleanup.

**Scope**
- During cache initialization, run lightweight integrity checks (`PRAGMA integrity_check`) when opening existing DB.
- If corruption is detected:
  - emit explicit one-line corruption notice
  - quarantine/rename corrupted DB artifact (optional `.corrupt.<timestamp>` suffix) when possible
  - rebuild a fresh DB automatically
- Ensure WAL/SHM sidecar files are handled safely in rebuild path.
- Add a single source of truth for corruption-detection/rebuild behavior in `CacheManager`.

**Acceptance Criteria**
- Corrupted DB does not cause undefined behavior or silent partial reads.
- Commands either:
  - recover automatically and continue, or
  - fail loudly with clear remediation if recovery is impossible.
- Corruption handling is idempotent across repeated runs.

**Tests Required**
- Unit test that writes invalid bytes to `index.db` and verifies automatic rebuild.
- Unit test that simulates `sqlite3.DatabaseError` during open/integrity-check and verifies deterministic path.
- Integration test that seeds a broken DB then runs `status` and `index` to confirm self-heal path.

---

## R3 - Concurrency and Race-Condition Hardening for Cache/Vector Operations

**Problem**
- Concurrent command execution (multiple `index/search/status` processes) can cause lock contention, stale assumptions, and race windows.
- Current behavior is mostly single-process friendly; explicit concurrency guarantees are not documented or tested.

**Goal**
- Make concurrent operations predictable, safe, and test-verified under contention.

**Scope**
- Define and document concurrency contract:
  - what is safe concurrently
  - expected behavior under write-write and read-write contention
  - retry/backoff policy and timeout behavior
- Configure SQLite pragmas and transaction mode intentionally (for example WAL + busy timeout where appropriate).
- Ensure vector-store writes are consistent with DB writes in concurrent index runs.
- Prevent partial state publication (metadata/profile should not report "healthy" while index write is incomplete).

**Acceptance Criteria**
- Concurrent invocations do not produce corrupt cache state.
- Lock contention results in bounded retry/fail behavior, not hangs.
- Status/profile markers remain internally consistent after concurrent runs.
- Behavior is documented in `docs/AGENT_INTEGRATION.md` or dedicated reliability docs.

**Tests Required**
- Fast concurrency integration test:
  - start two index runs against same cache/repo
  - verify both complete with deterministic result or one fails with explicit lock message
  - verify final cache is valid and searchable
- Unit tests for retry/backoff helpers and lock timeout handling.
- Regression test asserting no deadlock/hang (with strict timeout) under simulated contention.

---

## Ordering / Priority

1. R2 (corruption recovery) - highest operational risk, easiest to misdiagnose.
2. R1 (OS-level failure handling) - critical for CI/dev ergonomics and safe operations.
3. R3 (concurrency hardening) - important for robustness, likely broader design work.
