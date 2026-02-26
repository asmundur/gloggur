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

## Ordering / Priority

1. R5 (session bootstrap ergonomics) - low-priority reliability follow-up after feature closure.

---

## R5 - Deterministic New-Session Bootstrap for Local Codex Worktrees

**Status**: planned
**Priority**: P2
**Owner**: codex

**Problem**
- Fresh local Codex sessions can start with inconsistent developer ergonomics (for example, `gloggur` not immediately callable, watch runtime not visibly active, or session assumptions about `.venv`/PATH not met).
- These are startup/bootstrap reliability concerns and can be confused with watch-mode feature correctness.

**Goal**
- Make new local worktree sessions deterministic and self-explanatory: either fully ready (wrapper callable + watch runtime healthy) or failing fast with actionable diagnostics.

**Scope**
- Define and enforce an explicit startup-readiness contract for local Codex worktrees.
- Harden startup flow to guarantee deterministic outcomes for:
  - `gloggur` wrapper availability
  - watch runtime initialization/daemon startup
  - clear status semantics immediately after startup.
- Document a single verification probe for session readiness.

**Out of Scope**
- Core watch indexing correctness semantics (covered by `F1`).
- IDE/plugin autostart behavior and OS-level service installers.

**Acceptance Criteria**
- In a fresh local Codex session inside a worktree, startup either:
  - completes with `gloggur status --json` and `gloggur watch status --json` working predictably, or
  - fails non-zero with explicit actionable diagnostics.
- No ambiguous startup state where the environment appears ready but watch/process status is contradictory.
- README/agent docs include a short startup-readiness check for local worktree sessions.

**Tests Required**
- Integration coverage that simulates a fresh local worktree session and validates startup-readiness contract outcomes.
- Regression tests for contradictory startup runtime artifacts/signals (PID/state/status mismatch cases).

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

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

---

# Inspect Findings Backlog (2026-02-26 forced scan)

Source command:
- `gloggur inspect . --json --force --allow-partial`

Observed problem output (snapshot):
- `total warnings=618` across `reports_total=856`
- warning mix: `Missing docstring=240`, `Low semantic similarity=378`
- warning distribution: `src=197`, `tests=297`, `scripts=119` (tooling noise dominates)
- source hotspots:
  - missing docstrings: `bootstrap_launcher.py=23`, `io_failures.py=6`, `embeddings/errors.py=3`
  - low-semantic warnings: `indexer/cache.py=27`, `storage/vector_store.py=20`, `cli/main.py=17`

## F10 - Make Inspect Output Actionable by Default

**Status**: planned
**Priority**: P0
**Owner**: codex

**Problem**
- Current forced inspect output is too noisy (`618` warnings) to be triaged efficiently.
- Most warnings come from `tests/` and `scripts/`, which obscures production issues in `src/`.

**Goal**
- Make `inspect` produce high-signal output by default and preserve full-audit mode as opt-in.

**Scope**
- Add first-class path-class filters in `inspect` output and CLI options:
  - default focus on `src/`
  - explicit include switches for `tests/` and `scripts/`
- Add grouped warning summaries in JSON payload:
  - by warning type
  - by top file offenders
  - by path class (`src/tests/scripts`)
- Keep backward compatibility for existing fields while adding summary fields.

**Out of Scope**
- Changing parser behavior or embedding model selection.
- Automatically rewriting docstrings.

**Acceptance Criteria**
- `gloggur inspect . --json` returns source-focused output by default (no implicit `tests/` + `scripts/` flood).
- `gloggur inspect . --json --include-tests --include-scripts` preserves full-audit behavior.
- JSON payload includes deterministic grouped summary sections for triage automation.
- Existing consumers of legacy fields (`warnings`, `reports`, `total`) continue to work.

**Tests Required**
- Unit tests for new filter flags and default scope behavior.
- Unit tests for grouped summary payload shape and counts.
- Integration tests validating source-only default and full-audit opt-in.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

## F11 - Burn Down Source Missing-Docstring Hotspots

**Status**: planned
**Priority**: P1
**Owner**: codex

**Problem**
- Forced scan found `36` missing-docstring warnings in `src/`, heavily concentrated in:
  - `src/gloggur/bootstrap_launcher.py` (`23`)
  - `src/gloggur/io_failures.py` (`6`)
  - `src/gloggur/embeddings/errors.py` (`3`)
- Missing docs on core runtime utilities reduce API clarity and weaken inspect signal quality.

**Goal**
- Eliminate source missing-docstring warnings for public/protected runtime interfaces.

**Scope**
- Add meaningful docstrings for module-level helpers, classes, and public methods in hotspot files first.
- Ensure docstrings include behavior and failure semantics where relevant (especially bootstrap/io paths).
- Run focused inspect checks on edited files to verify warning removal.

**Out of Scope**
- Docstring cleanup for `tests/` and `scripts/`.
- Non-docstring style refactors.

**Acceptance Criteria**
- `gloggur inspect src/gloggur --json --force --allow-partial` reports `Missing docstring=0` for `src/gloggur/bootstrap_launcher.py`, `src/gloggur/io_failures.py`, and `src/gloggur/embeddings/errors.py`.
- Added docstrings are specific enough to avoid “semantic filler” text and pass existing lint/test checks.

**Tests Required**
- Unit/integration tests already covering touched functions must still pass.
- Add/adjust tests only if docstring-driven tooling behavior is changed.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree

## F12 - Calibrate Semantic Warning Scoring for Source Code

**Status**: planned
**Priority**: P1
**Owner**: codex

**Problem**
- Forced scan produced `161` low-semantic warnings in `src/` with `28` negative scores, including critical files (`indexer/cache.py`, `storage/vector_store.py`, `cli/main.py`).
- Current single-threshold behavior (`0.200`) likely over-flags legitimate docstrings and reduces trust in warnings.

**Goal**
- Improve precision of semantic warnings so “low similarity” is a reliable indicator instead of broad noise.

**Scope**
- Build a small labeled calibration set from current source hotspots (true-positive vs false-positive warnings).
- Evaluate threshold strategies (global threshold tuning and/or symbol-kind-aware thresholds).
- Implement calibrated scoring policy with explicit config defaults and explainability in output.

**Out of Scope**
- Replacing the embedding model in this task.
- Expanding inspection into a full NLP quality system.

**Acceptance Criteria**
- On the same forced scan command, source low-semantic warnings are reduced by at least `40%` from the baseline (`161` -> `<=96`) without suppressing clearly poor docstrings in calibration fixtures.
- JSON output includes enough score metadata to explain why warnings triggered under the calibrated policy.
- Calibration behavior is deterministic and documented.

**Tests Required**
- Unit tests for threshold policy selection and edge cases.
- Unit tests for score-to-warning classification under calibrated settings.
- Integration regression test asserting warning-count reduction on a stable fixture corpus.

**Links**
- PR/commit/issues/docs: pending local implementation in this worktree
