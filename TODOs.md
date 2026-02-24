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

**Progress Update (2026-02-24)**
- Added deterministic interruption-window coverage for partial-publication safety.
- Added `test_interrupted_index_run_preserves_needs_reindex_signal` to assert:
  - baseline healthy state before a rerun
  - metadata invalidation becomes observable during an in-flight index
  - forced interruption does not publish a false healthy state
  - recovery index restores healthy status markers

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

---

## F2 - Validate OpenAI and Gemini Embedding Providers End-to-End

**Status**: planned
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
