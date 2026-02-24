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

**Status**: ready_for_review

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

**Status**: ready_for_review

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

## Ordering / Priority

1. R4 (bootstrap/preflight reliability) - ready for user decision.
2. R2 (corruption recovery) - highest operational risk, easiest to misdiagnose.
3. R1 (OS-level failure handling) - critical for CI/dev ergonomics and safe operations.
4. R3 (concurrency hardening) - important for robustness, likely broader design work.
5. F1 (watch mode) - product capability work after reliability hardening.

---

## R4 - Perfect Reliability: Bootstrap, Preflight, and Self-Healing CLI Execution

**Status**: ready_for_review
**Priority**: P0
**Owner**: codex

**Problem**
- Operator workflows depend on `gloggur` or `scripts/gloggur`, but bootstrap assumptions are fragile across fresh worktrees.
- A missing `.venv` (or missing dependencies in `.venv`) causes immediate command failure before users can run `status`, `index`, or `search`.
- Failures are currently binary ("works" or "command not found/no such file"), with limited recovery guidance.

**Goal**
- Make command execution reliable in fresh and long-lived worktrees by adding deterministic preflight checks and self-healing fallback behavior.

**Scope**
- Add a bootstrap preflight command path used by `scripts/gloggur` that validates:
  - Python interpreter availability
  - virtualenv presence/health
  - required package importability (`gloggur`)
- Implement safe fallback order with explicit logs:
  - use repo `.venv` when healthy
  - otherwise use system `python -m gloggur` when available
  - otherwise fail with one actionable setup block
- Add a single canonical setup helper (`scripts/bootstrap_gloggur_env.sh`) that can create/repair `.venv` and install required extras for local dev.
- Add optional cache hydration to bootstrap helper so a fresh worktree can reuse an existing `.gloggur-cache` via symlink or copy for faster startup.
- Add optional virtualenv hydration to bootstrap helper so a fresh worktree can reuse an existing `.venv` via symlink or copy when package installs are unavailable.
- Ensure all early failures return structured JSON when `--json` is set, including:
  - `error_code` (`missing_venv`, `missing_python`, `missing_package`, `broken_environment`)
  - remediation steps
  - detected environment details
- Document expected bootstrap behavior and recovery flow in `README.md` and `docs/AGENT_INTEGRATION.md`.

**Out of Scope**
- Installing global system packages automatically without explicit user action.
- Replacing the Python packaging strategy (`pipx` vs `pip` vs editable installs).

**Acceptance Criteria**
- In a fresh clone with no `.venv`, `scripts/gloggur status --json` either succeeds via fallback or fails with deterministic JSON + clear remediation.
- In a broken `.venv` (missing `gloggur`), tool does not crash with raw traceback by default; it provides guided recovery.
- In healthy env, startup overhead remains low (preflight <200ms on warm path).
- Agent-required workflows in `AGENTS.md` run successfully after one documented bootstrap command, without assuming bare `gloggur` is on PATH.
- `scripts/bootstrap_gloggur_env.sh --seed-cache-from <workspace>` supports deterministic cache hydration with `--seed-cache-mode symlink|copy`.
- `scripts/bootstrap_gloggur_env.sh --seed-venv-from <workspace>` supports deterministic virtualenv hydration with `--seed-venv-mode symlink|copy`.

**Tests Required**
- Unit tests for preflight detection matrix:
  - missing `.venv`
  - missing interpreter
  - missing package
  - healthy environment
- Integration tests for wrapper behavior:
  - fallback to system python path
  - deterministic failure payload with `--json`
  - human-readable stderr guidance without `--json`
- Integration tests for bootstrap helper cache hydration:
  - `--seed-cache-mode symlink`
  - `--seed-cache-mode copy`
- Integration tests for bootstrap helper virtualenv hydration:
  - `--seed-venv-mode symlink`
- Regression test to ensure wrapper exit codes are stable across failure classes.

**Verification Evidence**
- Commands run:
  - `/Users/auzi/vinnustofa/gloggur/.venv/bin/python -m pytest tests/unit/test_bootstrap_launcher.py tests/integration/test_bootstrap_wrapper.py tests/integration/test_bootstrap_env_script.py -q`
  - `scripts/bootstrap_gloggur_env.sh --recreate --seed-venv-from /Users/auzi/vinnustofa/gloggur --seed-venv-mode symlink --seed-cache-from /Users/auzi/vinnustofa/gloggur --seed-cache-mode copy`
  - `scripts/gloggur status --json`
  - `scripts/gloggur index . --json`
  - `GLOGGUR_PREFLIGHT_DRY_RUN=1 GLOGGUR_PREFLIGHT_VENV_PYTHON=/tmp/does-not-exist/bin/python GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS=$(command -v python3) GLOGGUR_PREFLIGHT_PROBE_MODULE=gloggur.bootstrap_launcher scripts/gloggur status --json`
  - `python3 -m gloggur.bootstrap_launcher status --json`
  - `python3 -m compileall gloggur tests/integration/test_bootstrap_wrapper.py tests/integration/test_bootstrap_env_script.py tests/unit/test_bootstrap_launcher.py`
  - `scripts/gloggur inspect . --json`
- Results:
  - Bootstrap/preflight unit+integration tests passed (`15 passed`).
  - Bootstrap command completed in one run with deterministic venv/cache seeding and no network dependency.
  - Wrapper status, index, and direct launcher status all succeeded with JSON output in this worktree.
  - Dry-run preflight selected deterministic system fallback with structured payload and measured warm-path timings (`second_ms=53`, under 200ms).
  - Compileall and inspect checks succeeded.

---

## F1 - Configurable On-Save Incremental Indexing (Watch Mode)

**Status**: blocked
**Priority**: P1
**Owner**: codex

**Problem**
- Incremental indexing currently requires explicit command runs and does not update automatically on file save.
- Reindexing changed files can leave stale vectors/results when symbol IDs shift, because vector entries are append-only today.

**Goal**
- Add first-class watch mode with background lifecycle commands and correct vector upsert/removal semantics.

**Scope**
- Add `watch` CLI commands (`init`, `start`, `stop`, `status`) and watch config keys/env overrides.
- Implement watcher runtime with file filtering, debounce/coalescing, hash-based skip, and heartbeat state.
- Add vector removal/upsert support for FAISS and fallback paths with legacy migration handling.
- Add cache helpers for file metadata deletion and file counts.
- Update README + agent docs and add tests for watch lifecycle and vector correctness.

**Out of Scope**
- IDE plugin development.
- Full OS autostart installers.

**Acceptance Criteria**
- Save-triggered updates index changed files without manual `gloggur index`.
- Unchanged content does not re-embed.
- Deleted/renamed symbols do not appear as stale search hits.
- Watch lifecycle commands report and manage running state predictably.

**Tests Required**
- Unit coverage for vector `remove_ids`/`upsert_vectors` and watcher processing behavior.
- Integration coverage for `watch init/start/status/stop` and stale-result regression scenarios.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

**Blockers (2026-02-20)**
- Scope includes "watch config keys/env overrides", but env overrides for `watch_state_file`, `watch_pid_file`, and `watch_log_file` are not implemented in `src/gloggur/config.py`.
- Do not move to `DONEs.md` until scope is either implemented fully or narrowed explicitly in this TODO item.
