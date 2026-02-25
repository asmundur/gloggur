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

**Status**: ready_for_review

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
- Unit tests that simulate `OSError`/`sqlite3.OperationalError`/`sqlite3.DatabaseError` via monkeypatch/mocks for each category.
- Fast integration tests using temp dirs with restricted permissions (where portable).
- Assertions on:
  - exit code
  - JSON/error payload fields
  - stderr content and remediation hints

**Progress Update (2026-02-24)**
- Added structured classification support for `sqlite3.DatabaseError` (not only `sqlite3.OperationalError`) so malformed/corrupted DB failures map deterministically.
- Expanded cache I/O wrapping to convert `sqlite3.DatabaseError` during connect/pragma/transaction paths into `StorageIOError` payloads.
- Added a core-command unit matrix (`status`, `search`, `inspect`, `clear-cache`, `index`) asserting structured JSON failure payloads for sqlite database-level failures.
- Added an integration matrix for unwritable cache parent across core commands to validate non-zero exits and stable category families in realistic filesystem conditions.

**Progress Update (2026-02-25)**
- Added a full core-command/category matrix in `tests/unit/test_cli_main.py` asserting consistent classification for:
  - `permission_denied`
  - `read_only_filesystem`
  - `disk_full_or_quota`
  - `path_not_writable`
  - `unknown_io_error`
  across `status`, `search`, `inspect`, `clear-cache`, and `index`.
- Added per-operation fault-injection coverage for vector artifact paths during `clear-cache`:
  - permission-denied delete failure on `vectors.json` now asserted as structured `io_failure`.
  - invalid `vectors.index` read now asserted as structured `io_failure` instead of raw traceback.
- Added malformed vector artifact coverage for `clear-cache`:
  - invalid JSON in `vectors.json` is now asserted as structured `io_failure` (`operation=read vector id map`).
  - malformed fallback `vectors.npy` is now asserted as structured `io_failure` (`operation=read fallback vector matrix`).
- Fixed `VectorStore.load()` so FAISS index read failures are wrapped deterministically (`operation=read faiss index file`) for stable CLI JSON/non-JSON handling.
- Hardened vector-store load parsing paths so id-map decode/type failures and malformed fallback matrices are wrapped as deterministic `io_failure` payloads instead of uncaught exceptions.

**Progress Update (2026-02-25, later)**
- Corrected `clear-cache` behavior to avoid loading existing vector artifacts before deletion:
  - `clear-cache` now initializes `VectorStore(..., load_existing=False)` and clears artifacts directly.
  - malformed `vectors.index` / `vectors.json` / `vectors.npy` no longer block cache reset when files are removable.
- Updated unit coverage in `tests/unit/test_cli_main.py` to assert successful artifact cleanup for malformed vector files (instead of failure-on-read semantics) while preserving explicit delete-failure coverage.
- Wrapped metadata-store sqlite connection/pragma/transaction failures as structured `StorageIOError` payloads (`src/gloggur/storage/metadata_store.py`) so `search` surfaces stable `io_failure` JSON instead of raw sqlite exceptions.
- Added regression coverage (`tests/unit/test_cli_main.py`) asserting `search --json` maps metadata-store connect failures to deterministic `io_failure` (`operation=open metadata database connection`).
- Wrapped core config-load parse/file failures as structured `io_failure` (`operation=read gloggur config`) so malformed JSON/YAML config files fail deterministically without traceback leakage.
- Added a core-command malformed-config matrix in `tests/unit/test_cli_main.py` asserting stable JSON `io_failure` payloads for `status`, `search`, `inspect`, `clear-cache`, and `index`.
- Tightened malformed-config failure guidance in `src/gloggur/cli/main.py`:
  - config parse/type failures now emit config-specific probable cause/remediation (fix syntax/top-level mapping), instead of generic cache-path I/O guidance.
- Extended malformed-config regressions in `tests/unit/test_cli_main.py` to assert config-specific probable-cause/remediation text stability.
- Removed a fragile double config-file read in CLI runtime creation:
  - `_create_runtime` now applies `--embedding-provider` override in-memory instead of calling `GloggurConfig.load(...)` a second time.
  - this removes a race window for unwrapped config I/O failures and keeps config error handling fully inside `_load_config`.
- Added regression coverage in `tests/unit/test_cli_main.py` ensuring provider overrides do not trigger a second config load path.
- Fixed `--config` path normalization in `_load_config` so `~` expansion is applied before file reads (all commands now accept tilde-prefixed config paths deterministically).
- Added regression coverage in `tests/unit/test_cli_main.py` asserting `status --json --config ~/.gloggur.yaml` resolves and loads the expected config file.

**Findings (2026-02-24)**
- Deterministic failure semantics now cover a broader sqlite failure surface, including non-operational database exceptions.
- Additional per-operation fault injection (for each write/delete operation inside each command) is still a useful follow-up for deeper exhaustiveness.

**Verification Procedure (How Evidence Is Obtained)**
- Run:
  - `.venv/bin/python -m pytest tests/unit/test_io_failures.py tests/unit/test_cli_main.py tests/integration/test_io_failures_integration.py -q`
- Validate in test outputs/assertions:
  - all core commands exit non-zero on injected I/O failures
  - JSON payload contains `error.type=io_failure`, category, operation, path, remediation, and original detail
  - non-JSON output includes probable cause + remediation steps

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
- Recovery behavior is validated for all core commands (`status`, `search`, `inspect`, `clear-cache`, `index`) with deterministic non-traceback outcomes.
- Corruption integration coverage is provider/backend independent (must run when FAISS is unavailable).
- Sequential reruns after first successful recovery do not create new quarantine artifacts unless fresh corruption is introduced.
- Concurrent recovery attempts against the same corrupted cache do not hang; outcomes are deterministic (both succeed, or one fails with bounded structured lock/IO error).
- JSON and non-JSON outputs remain stable during corruption recovery:
  - JSON payload remains parseable and machine-readable.
  - Human-readable stream includes a one-line corruption notice plus actionable remediation on hard failures.

**Tests Required**
- Unit test that writes invalid bytes to `index.db` and verifies automatic rebuild.
- Unit test that simulates `sqlite3.DatabaseError` during open/integrity-check and verifies deterministic path.
- Integration test that seeds a broken DB then runs `status` and `index` to confirm self-heal path.
- Integration matrix for all core commands on corrupted DB (`status`, `search`, `inspect`, `clear-cache`, `index`) asserting deterministic exit semantics and parseable JSON where requested.
- Integration coverage for corruption recovery in fallback/no-FAISS mode (no `importorskip("faiss")` dependency for R2-required tests).
- Regression test for sequential idempotence:
  - first run recovers and may quarantine
  - second run on healthy cache does not emit another corruption notice or new quarantine suffix.
- Concurrency integration test with two simultaneous recovery-triggering commands against the same corrupted cache, asserting bounded completion and stable outcomes.
- Failure-path unit/integration test where quarantine and delete both fail, asserting clear remediation guidance and non-zero exit from CLI surfaces.

**Gap Notes (2026-02-24)**
- `tests/integration/test_cli.py` is no longer gated by `pytest.importorskip("faiss")`, so corruption recovery coverage is backend-independent at test collection time.

**Progress Update (2026-02-25)**
- Added backend-independent no-FAISS runtime integration coverage in `tests/integration/test_corruption_recovery_integration.py` by simulating `faiss` import failure in subprocess runs and asserting corruption recovery across core commands.
- Added concurrent corruption-recovery-attempt coverage (`status --json` in parallel against the same corrupted cache) with bounded completion and deterministic non-traceback outcomes.
- Tightened concurrent-recovery failure assertions to require parseable structured JSON `io_failure` payloads (no fallback text-only acceptance) when one concurrent command exits non-zero.
- Added CLI-surface failure-path coverage in `tests/unit/test_cli_main.py`:
  - core-command matrix ensures non-zero exit when cache recovery is unrecoverable
  - explicit `status` assertions (JSON and non-JSON) verify remediation guidance when quarantine/delete both fail.
- Updated cache-recovery error mapping in `src/gloggur/cli/main.py` so unrecoverable corruption recovery surfaces as structured `io_failure` payloads (including JSON mode) instead of plain `ClickException` text.

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

**Progress Update (2026-02-24)**
- Added deterministic interruption-window coverage for partial-publication safety.
- Added `test_interrupted_index_run_preserves_needs_reindex_signal` to assert:
  - baseline healthy state before a rerun
  - metadata invalidation becomes observable during an in-flight index
  - forced interruption does not publish a false healthy state
  - recovery index restores healthy status markers

**Progress Update (2026-02-25)**
- Added mixed-writer contention coverage for `clear-cache` under an active cache write lock (`tests/integration/test_concurrency_integration.py`), asserting:
  - bounded non-hanging completion
  - deterministic lock-timeout `io_failure` payload (`operation=acquire cache write lock`).

**Verification Procedure (How Evidence Is Obtained)**
- Run:
  - `.venv/bin/python -m pytest tests/unit/test_concurrency.py tests/integration/test_concurrency_integration.py -q`
- Validate in test outputs/assertions:
  - lock-timeout failures are bounded and explicit
  - interrupted index runs leave `needs_reindex=true`
  - post-recovery index returns `needs_reindex=false`

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
  - non-dry-run fallback executes requested CLI command successfully
  - deterministic failure payload with `--json`
  - human-readable stderr guidance without `--json`
  - warm-path timing checks for both fallback and healthy-venv preflight paths
- Integration tests for bootstrap helper cache hydration:
  - `--seed-cache-mode symlink`
  - `--seed-cache-mode copy`
- Integration tests for bootstrap helper virtualenv hydration:
  - `--seed-venv-mode symlink`
  - `--seed-venv-mode copy`
- Regression test to ensure wrapper exit codes are stable across failure classes.

**Progress Update (2026-02-24)**
- Added integration coverage for non-dry-run fallback execution (`scripts/gloggur status --json` succeeds via system-python fallback when `.venv` is missing).
- Added warm-path timing coverage for healthy-venv preflight selection.
- Added bootstrap integration coverage for `--seed-venv-mode copy`.
- Added bootstrap validation coverage for invalid seed-mode input handling.

**Verification Procedure (How Evidence Is Obtained)**
- Run:
  - `.venv/bin/python -m pytest tests/unit/test_bootstrap_launcher.py tests/integration/test_bootstrap_wrapper.py tests/integration/test_bootstrap_env_script.py -q`
  - `scripts/gloggur status --json`
  - `GLOGGUR_PREFLIGHT_DRY_RUN=1 GLOGGUR_PREFLIGHT_VENV_PYTHON=/tmp/does-not-exist/bin/python GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS=$(command -v python3) GLOGGUR_PREFLIGHT_PROBE_MODULE=gloggur.bootstrap_launcher scripts/gloggur status --json`
  - `GLOGGUR_PREFLIGHT_DRY_RUN=1 GLOGGUR_PREFLIGHT_VENV_PYTHON=$(command -v python3) GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS=$(command -v python3) GLOGGUR_PREFLIGHT_PROBE_MODULE=gloggur.bootstrap_launcher scripts/gloggur status --json`
- Validate in test outputs/assertions:
  - exit-code mapping remains stable across failure classes
  - fallback and healthy-venv preflight paths both report warm-path timings under acceptance target
  - bootstrap venv/cache seed modes behave deterministically for symlink/copy paths

---

## F1 - Configurable On-Save Incremental Indexing (Watch Mode)

**Status**: in_progress
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

**Blockers (2026-02-20, updated 2026-02-24)**
- Resolved: env overrides for `watch_state_file`, `watch_pid_file`, and `watch_log_file` are now implemented in `src/gloggur/config.py` with unit coverage.
- Still do not move to `DONEs.md` until full F1 scope and acceptance criteria are verified end-to-end.

**Progress Update (2026-02-25)**
- Fixed a daemon lifecycle race in `watch start` where the spawned foreground process could exit immediately as `already_running` after reading the parent-written PID file.
- Added integration coverage for `watch init/start/status/stop` with real subprocess execution and env override verification for `watch_state_file`, `watch_pid_file`, and `watch_log_file` (`tests/integration/test_watch_cli_lifecycle_integration.py`).
- Extended unit coverage to assert daemon child marker propagation during spawn (`tests/unit/test_cli_watch.py`).
- Strengthened watch lifecycle integration assertions to require observed save-event processing and searchable post-save updates (using forced polling in test env for deterministic backend behavior).

**Progress Update (2026-02-25, later)**
- Hardened watch subcommands (`watch init/start/stop/status`) with `_with_io_failure_handling` so structured `io_failure` / `embedding_provider_error` payloads are emitted in `--json` mode instead of raw tracebacks.
- Wrapped watch runtime file/config operations with deterministic I/O mapping:
  - config payload read/write
  - pid file write/delete
  - watch state file write
  - daemon log directory/file setup
  - daemon process spawn failure path
  - daemon early-exit startup verification path
  - stop-signal race (`os.kill`) path.
- Added daemon-startup cleanup for partial-initialization failures:
  - if pid/state-file write fails after daemon spawn, watch start now attempts best-effort `SIGTERM` of the spawned process before surfacing structured `io_failure`.
  - termination now escalates to bounded `SIGKILL` fallback when a wait-capable daemon does not exit after `SIGTERM`, reducing orphan-process risk during partial-startup failures.
  - if state-file write fails after pid-file write, watch start now best-effort removes the just-written pid file to avoid stale-runtime artifacts.
  - if daemon exits immediately after pid/state writes, watch start now detects the post-init exit, cleans stale pid file, writes `failed_startup` state, and returns structured `io_failure`.
  - if daemon exits during the first startup liveness check (before pid file publication), watch start now records `failed_startup` state for deterministic observability.
- Added unit regressions in `tests/unit/test_cli_watch.py` asserting structured JSON failures for:
  - config write permission failure in `watch init`
  - daemon log-directory permission failure in `watch start`
  - daemon process spawn failure in `watch start`
  - daemon early-exit startup failure in `watch start`
  - daemon pid-write failure path terminates spawned process and returns stable `io_failure`.
  - daemon state-write failure path terminates spawned process and removes stale pid file.
  - daemon post-init early-exit path clears stale pid file, records failed-startup state, and returns stable `io_failure`.
  - process-signal race failure in `watch stop`.
- Added a guardrail for malformed watch config top-level payload type (non-mapping YAML/JSON now fails deterministically as structured `io_failure` in `watch init`, rather than silently coercing to `{}`).
- Hardened watcher status observability by failing loudly on malformed runtime artifacts:
  - malformed watch PID files now return structured `io_failure` (`operation=read watch pid file`) instead of silently reporting not-running.
  - malformed watch state JSON now returns structured `io_failure` (`operation=read watch state file`) instead of being silently ignored.
- Added regression tests in `tests/unit/test_cli_watch.py` covering malformed PID and malformed state-file status paths.
- Normalized daemon startup failure-state payloads to include `watch_path` consistently across early-exit branches for deterministic status observability.
- Added stale-runtime cleanup regression coverage for first-liveness-check daemon early-exit: pre-existing stale pid artifacts are now asserted to be removed on startup failure.
- Added daemon-termination regression coverage for timed-out shutdown cleanup (`tests/unit/test_cli_watch.py`): `watch start --daemon` partial-init failure paths now assert `SIGTERM` then `SIGKILL` when the spawned process remains stuck.

---

## F2 - Validate OpenAI and Gemini Embedding Providers End-to-End

**Status**: in_progress
**Priority**: P0
**Owner**: codex

**Problem**
- Multi-provider embedding support is expected, but OpenAI and Gemini execution paths are not yet verified end-to-end in this repo workflow.
- Without deterministic checks, failures can hide behind fallback behavior, provider config drift, or environment-specific credential issues.
- Agents and developers need a stable, documented way to confirm both providers can produce usable embeddings for indexing/search.

**Goal**
- Ensure OpenAI and Gemini embedding integrations are both operational, test-covered, and diagnosable with clear failure messages.

**Scope**
- Audit and verify provider wiring for OpenAI and Gemini in config loading, provider factory selection, and runtime embedding calls.
- Add or tighten validation for required provider settings (model name, API key env vars, optional dimensions/task type where applicable).
- Implement deterministic tests for each provider adapter using mocked SDK/network boundaries.
- Add integration-style checks that exercise provider selection + embed flow and validate returned vector shape/typing and error handling.
- Document the exact setup and verification commands for both providers in repo docs used by agents.

**Out of Scope**
- Benchmarking embedding quality/ranking relevance across providers.
- Production cost optimization or model-selection policy work.
- Adding new embedding providers beyond OpenAI and Gemini.

**Acceptance Criteria**
- OpenAI provider path can be selected and produces embeddings successfully with valid configuration.
- Gemini provider path can be selected and produces embeddings successfully with valid configuration.
- Missing/invalid credentials fail with explicit actionable errors (no silent fallback that masks provider failures).
- Tests cover provider selection, success path, and failure path for both providers.
- Documentation includes a copy-paste verification checklist for both providers.

**Tests Required**
- Unit tests for provider config validation and factory dispatch for `openai:*` and `gemini:*` profiles.
- Unit tests for provider adapters that mock SDK responses and assert vector dimensionality/type normalization.
- Unit tests for credential and API failure mapping into stable error messages/codes.
- Integration tests that run `index` and `search` against small fixture data with each provider behind deterministic mocks.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

**Progress Update (2026-02-25)**
- Added explicit actionable error mapping for provider API failures:
  - OpenAI embedding calls now raise `RuntimeError` with model + original error detail on request failure.
  - Gemini embedding calls now raise `RuntimeError` with model + original error detail on request failure.
- Added unit coverage for OpenAI and Gemini API-failure message paths in `tests/unit/test_embeddings.py`.
- Added deterministic integration coverage for provider selection and end-to-end CLI flow (`index` + `search`) using mocked OpenAI/Gemini providers in `tests/integration/test_provider_cli_integration.py`.
- Added CLI-surface credential failure coverage for OpenAI and Gemini in JSON mode, asserting no traceback leakage and stable actionable `embedding_provider_error` payloads.
- Hardened `search` provider-init path for invalid/unset provider config: missing provider now returns structured `embedding_provider_error` JSON/non-JSON output instead of assertion-style failure (`tests/unit/test_cli_main.py`).
- Hardened `index` provider-init path for invalid/unset provider config: missing provider now returns structured `embedding_provider_error` JSON output instead of silently indexing without embeddings (`tests/unit/test_cli_main.py`).
- Added provider verification checklist commands to `README.md` and fixed `scripts/run_provider_probe.py` to support current schema-v2 `vectors.json` format.
- Corrected Gemini probe skip guidance in `scripts/run_provider_probe.py` to list all supported API-key env vars (`GLOGGUR_GEMINI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`) with unit regression coverage in `tests/unit/test_run_provider_probe.py`.
- Remaining closure gap: collect at least one live-key smoke run artifact (outside CI) to confirm real provider account/config behavior in a non-mocked environment.

---

## F3 - Portable Index Artifact Publishing for CI/CD and Codex Cloud

**Status**: planned
**Priority**: P0
**Owner**: codex

**Problem**
- `.gloggur-cache` is local to a workspace and is not directly available in ephemeral CI runners or Codex cloud execution environments.
- There is no single, provider-agnostic command to publish the index artifact to shared file storage.
- Teams currently need custom one-off scripts per platform, which is brittle and slows Codex GitHub integration workflows.

**Goal**
- Provide one non-interactive command that packages and uploads a validated index artifact to generic file storage, so Codex cloud environments can reuse it.

**Scope**
- Define a versioned index artifact contract (archive + manifest + checksums) for `.gloggur-cache` publication.
- Implement a CI-friendly publish command (for example `gloggur artifact publish`) with:
  - source path input
  - destination URI or uploader command template
  - `--json` machine-readable output
  - deterministic non-zero exit semantics
- Support generic transport patterns that work across CI/CD systems:
  - direct `https://` upload (presigned URL style)
  - local/file target for testing
  - pluggable uploader command for provider CLIs (`aws`, `gsutil`, `az`, Artifactory, etc.)
- Emit artifact metadata required for safe reuse:
  - checksum(s)
  - schema/profile compatibility fields
  - creation timestamp and tool version
- Document copy-paste CI examples (including GitHub Actions) and Codex cloud consumption guidance.

**Out of Scope**
- Managing or rotating cloud credentials/secrets.
- Implementing storage lifecycle/retention policy automation.
- Provider-specific optimization features beyond the generic transport contract.

**Acceptance Criteria**
- A single headless command can run in CI/CD and upload the index artifact without interactive prompts.
- Command output includes artifact URI, checksum, and compatibility metadata in JSON.
- Failed upload/auth/network paths return stable actionable errors and non-zero exits.
- Published artifact can be validated and consumed by a downstream Codex cloud workflow without manual file surgery.
- Documentation includes at least one generic workflow and one GitHub integration example.

**Tests Required**
- Unit tests for artifact packaging, manifest schema, and checksum generation.
- Unit tests for destination parsing and uploader command templating/escaping.
- Unit tests for error mapping (auth failure, network timeout, invalid destination).
- Integration tests with:
  - local HTTP upload endpoint
  - local filesystem destination
  - mocked external uploader command
- CI smoke test fixture that runs the publish command and verifies emitted JSON metadata.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

---

## F4 - Expand GitHub Actions Python Coverage to 3.13 and 3.14

**Status**: in_progress
**Priority**: P1
**Owner**: codex

**Importance Assessment**
- High importance for forward-compatibility and contributor confidence because Python minor releases now outpace static CI matrices.
- Not P0 because current supported lanes (`3.10`-`3.12`) still protect primary development flow; this is proactive reliability, not an active outage.

**Priority Assessment**
- Recommended priority: `P1` (next-cycle reliability hardening).
- Escalate to `P0` if any of these are true:
  - users or CI runners already default to `3.13`/`3.14`
  - dependency resolver or runtime incompatibilities are reported
  - project support policy is updated to require latest Python minors immediately

**Problem**
- Current CI verification runs only on Python `3.10`, `3.11`, and `3.12` in `.github/workflows/verification.yml`.
- Python `3.13` and `3.14` compatibility is not continuously validated, so breakage can ship unnoticed.
- Without an explicit support policy, contributors cannot tell whether failures on newer runtimes are blocking or informational.

**Goal**
- Add explicit GitHub Actions coverage for Python `3.13` and `3.14` with a clear pass/fail policy aligned to project support guarantees.

**Scope**
- Update verification workflow matrix to include `3.13` and `3.14`.
- Define runtime support tiers in docs:
  - required versions (blocking CI)
  - provisional/experimental versions (if temporary `continue-on-error` is needed)
- Ensure dependency install and test commands are stable across the expanded matrix.
- Add fast diagnostics in CI logs for interpreter version, pip resolver output, and failing package constraints.
- Document how to evolve the matrix when new Python minors are released.

**Out of Scope**
- Dropping support for existing versions (`3.10`-`3.12`) in this task.
- Rewriting test architecture or significantly increasing test suite runtime beyond matrix changes.
- Packaging/distribution metadata overhauls unless required by CI failures.

**Acceptance Criteria**
- GitHub Actions runs verification jobs on Python `3.10`, `3.11`, `3.12`, `3.13`, and `3.14`.
- CI policy clearly indicates which versions are required vs provisional, and that policy is documented.
- If any version is provisional, the reason and graduation criteria are documented.
- A PR touching Python code surfaces compatibility regressions for supported versions before merge.
- No hidden matrix exclusions; workflow file reflects the published support policy.

**Tests Required**
- Workflow validation check (for example `act` or YAML schema/lint) for updated matrix syntax.
- CI run evidence from at least one PR/branch showing all matrix jobs triggered.
- If provisional mode is used, a regression test ensuring provisional failures do not mask required-version failures.
- Local smoke run on newest interpreter available in repo toolchain (`3.13` or `3.14`) for core test subset.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

**Progress Update (2026-02-25)**
- Updated `.github/workflows/verification.yml` matrix to include Python `3.13` (required) and `3.14` (provisional/non-blocking via lane-level `continue-on-error`).
- Added explicit runtime-lane diagnostics in CI output (`python-version`, `required-lane`) and disabled fail-fast so provisional failures do not hide required-lane results.
- Documented Python support tiers and 3.14 graduation criteria in `README.md`.
- Added a workflow-policy regression test (`tests/unit/test_verification_workflow.py`) asserting:
  - required/provisional lane membership is stable
  - `continue-on-error: ${{ !matrix.required }}`
  - `fail-fast: false` (provisional failures cannot short-circuit required lanes)
- Added local Python 3.13 smoke evidence on core subsets:
  - `.venv/bin/python -m pytest tests/unit/test_cli_main.py tests/unit/test_concurrency.py tests/integration/test_watch_cli_lifecycle_integration.py -q`
- Tightened dependency-step diagnostics in `.github/workflows/verification.yml`:
  - prints `python --version` and `python -m pip --version`
  - emits bounded `pip debug --verbose` output for resolver/environment triage
  - runs verbose install (`python -m pip install ... -v`) and captures a bounded `pip freeze` snapshot
- Added workflow regression coverage in `tests/unit/test_verification_workflow.py` asserting those diagnostics remain present.
- Remaining closure gap: CI run evidence from at least one PR/branch showing all matrix jobs triggered.

---

# Agent Foundations Backlog

These tasks operationalize a minimal, production-grade agent path for Glöggur: durable memory, bounded verify/repair loops, and explicit non-goals to avoid orchestration bloat.

## F5 - Deterministic Index Fingerprinting and Session Continuity

**Status**: planned
**Priority**: P0
**Owner**: codex

**Problem**
- Ephemeral environments (CI runners, Codex cloud, one-shot local sessions) lose continuity between runs.
- Current index freshness checks do not provide a full session-resume contract tied to workspace state.
- Agents can restart without reliable knowledge of whether prior semantic state is reusable.

**Goal**
- Make semantic continuity deterministic across sessions by introducing reproducible index fingerprints and resumable session metadata.

**Scope**
- Define a stable fingerprint schema for index compatibility (workspace path hash, file state digest, embedding profile, schema version, gloggur version).
- Persist and expose fingerprint metadata in cache and CLI JSON output (`status`, `index`, `search`).
- Add `session resume` metadata contract:
  - last successful index fingerprint
  - timestamp
  - cache compatibility decision (`resume_ok` vs `reindex_required`)
- Document how agents should consume this contract in ephemeral environments.

**Out of Scope**
- Remote artifact transport/upload implementation (covered by `F3`).
- Multi-workspace memory graphing or cross-repo memory merge.

**Acceptance Criteria**
- `gloggur status --json` emits deterministic fingerprint and explicit resume decision fields.
- Session resume decisions are reproducible for unchanged workspaces and invalidated for meaningful state/profile changes.
- Compatibility mismatch reasons are machine-readable (not only free-text).
- Documentation includes a copy-paste “resume decision” flow for agent runners.

**Tests Required**
- Unit tests for fingerprint generation stability across path ordering and metadata ordering variations.
- Unit tests for compatibility decisions across schema/profile/version/file-state mismatch cases.
- Integration tests that run `index` then `status`/`search` in a fresh process and verify stable resume behavior.
- Regression tests for JSON payload schema stability.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

---

## F6 - Incremental Rebuild Engine for Changed Files Only

**Status**: planned
**Priority**: P0
**Owner**: codex

**Problem**
- Full reindexing penalizes iterative workflows and undermines agent turnaround in ephemeral environments.
- Small file changes currently force disproportionately expensive index rebuild paths.

**Goal**
- Recompute only the symbols affected by changed files while preserving deterministic index correctness.

**Scope**
- Introduce change-set detection keyed by file content digest + symbol extraction result.
- Rebuild only impacted file embeddings and metadata records; remove deleted/renamed file symbols safely.
- Add clear CLI observability fields for incremental runs:
  - files scanned
  - files changed
  - symbols added/updated/removed
  - elapsed time
- Add fallback to full rebuild on incompatible cache/profile/schema conditions.

**Out of Scope**
- Filesystem watcher redesign beyond existing watch-mode behavior.
- Parallel/distributed indexing across hosts.

**Acceptance Criteria**
- Re-running `index` on unchanged workspace performs near-no-op update with explicit “no changes” reporting.
- Single-file edits update only that file’s symbol rows/vectors and preserve search correctness.
- File renames/deletions remove stale symbols with no ghost search hits.
- Incompatible cache states automatically trigger safe full rebuild with explicit reason.

**Tests Required**
- Unit tests for change detection, rename/deletion handling, and no-op behavior.
- Unit tests for vector/metadata consistency after incremental update paths.
- Integration tests comparing incremental vs full rebuild search results on the same fixture repo.
- Performance regression check asserting unchanged-run speedup versus baseline full rebuild.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

---

## F7 - Retrieval Confidence Scoring and Bounded Re-query

**Status**: planned
**Priority**: P1
**Owner**: codex

**Problem**
- Retrieval currently lacks a first-class confidence signal for downstream agent decision logic.
- Low-quality retrieval can flow directly into responses without a deterministic recovery step.

**Goal**
- Provide confidence-aware retrieval with one bounded repair step (`re-query`) when confidence is below threshold.

**Scope**
- Define and expose a retrieval confidence score per result set (and per top result where useful).
- Add configurable low-confidence threshold and max re-query attempts (default: one retry).
- Implement deterministic fallback strategy for re-query (for example: broaden query terms or increase top-k within limits).
- Emit telemetry fields in JSON output:
  - initial confidence
  - retry performed (boolean)
  - final confidence
  - retry strategy used

**Out of Scope**
- Autonomous long-horizon planning loops.
- Multi-agent negotiation or chain-of-thought planning frameworks.

**Acceptance Criteria**
- Retrieval output includes confidence and retry metadata in JSON mode.
- Low-confidence queries trigger at most one bounded retry by default.
- Retry path measurably improves confidence on a representative fixture set, or exits with explicit low-confidence marker.
- Behavior is fully configurable and can be disabled.

**Tests Required**
- Unit tests for confidence calculation edge cases and threshold handling.
- Unit tests for bounded retry enforcement and deterministic retry strategy selection.
- Integration tests with synthetic low-signal queries verifying retry metadata and bounded behavior.
- Regression tests for backward-compatible default CLI behavior when feature is disabled.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

---

## F8 - Evidence Trace and Validation Hooks for Agent Outputs

**Status**: planned
**Priority**: P1
**Owner**: codex

**Problem**
- Agents using Glöggur can generate answers without a standardized trace of which symbols informed the output.
- There is no common validation hook to gate low-grounding answers before they are emitted.

**Goal**
- Add lightweight verification primitives so answer generation can shift from “generate and hope” to “generate, validate, repair/flag.”

**Scope**
- Emit structured evidence trace payloads for retrieval-backed responses:
  - symbol IDs
  - file paths
  - line spans (where available)
  - confidence contribution
- Add optional validation hook interface for agent integrators (pass/fail + reason + optional suggested repair action).
- Define a minimal default validator:
  - require at least one evidence item above confidence threshold
  - fail closed with explicit low-grounding reason when unmet
- Document reference integration flow for agents consuming `search --json`.

**Out of Scope**
- Full autonomous “self-healing” plan/execution frameworks.
- Building a generalized policy engine for all agent safety concerns.

**Acceptance Criteria**
- JSON output can include an evidence trace payload tied to returned symbols.
- Validation hook can block/flag ungrounded responses deterministically.
- Default validator behavior is documented and test-covered.
- Agent integration docs include one end-to-end example: retrieve -> validate -> emit/repair.

**Tests Required**
- Unit tests for evidence trace schema generation and normalization.
- Unit tests for validator pass/fail behavior and reason codes.
- Integration tests covering grounded vs ungrounded query scenarios.
- Backward-compatibility tests ensuring trace/validation features are opt-in where required.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

---

## F9 - Minimal Reference Agent Loop and Eval Harness

**Status**: planned
**Priority**: P1
**Owner**: codex

**Problem**
- Teams lack a canonical minimal integration showing how to use Glöggur without framework bloat.
- Without a small eval harness, regressions in retrieval+validation quality are hard to detect.

**Goal**
- Ship a compact, framework-agnostic reference loop that demonstrates tool calling, bounded orchestration state, logging, retries, and tiny evals.

**Scope**
- Add a small reference agent example (single tool, single goal) that:
  - calls Glöggur search
  - applies confidence/validation checks
  - retries once when allowed
  - returns structured final output
- Add structured logs for each step (`decide`, `act`, `validate`, `stop`).
- Add timeout/retry guardrails with clear failure modes.
- Add tiny eval suite (minimum 10 representative cases) with pass/fail summary output.

**Out of Scope**
- Multi-agent orchestration.
- Complex planner modules or long-horizon autonomous task decomposition.

**Acceptance Criteria**
- Reference loop runs end-to-end in one command and documents setup expectations.
- Example remains under a modest complexity budget (small code footprint, no heavy framework dependency).
- Eval harness reports deterministic summary metrics and fails non-zero when below threshold.
- Documentation explains how to adapt the loop to real tools without introducing planning complexity.

**Tests Required**
- Unit tests for loop-state transitions and retry/timeout guardrails.
- Integration tests running the reference loop against a fixture repo.
- Eval harness test verifying deterministic result formatting and failure exit semantics.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree
