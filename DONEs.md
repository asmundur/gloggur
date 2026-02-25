# DONEs Workflow

`DONEs.md` is the immutable log of completed work.

## Usage Rules

1. Only move tasks here after acceptance criteria are actually met.
2. Keep the same task ID used in `TODOs.md`.
3. Include completion date (`YYYY-MM-DD`), evidence, and verification commands.
4. Record behavioral impact (what changed for users/agents).
5. If follow-up work is discovered during closure, create a new item in `TODOs.md` and link it.
6. Do not rewrite history; append corrections as follow-up entries.

## Completion Entry Template

Copy this section for completed tasks:

```md
## <ID> - <Title>

**Completed On**: YYYY-MM-DD
**Completed By**: <name or agent>
**Source**: moved from TODOs.md

**Delivered**
- <implemented change 1>
- <implemented change 2>

**Verification**
- Commands run:
  - `<command>`
  - `<command>`
- Results:
  - <pass/fail summary>

**Evidence**
- Files: `/abs/path/one`, `/abs/path/two`
- PR/commit/issues: <links or identifiers>

**Follow-ups**
- None

```

## D1 - Document Semantic Search Embedding Inputs

**Completed On**: 2026-02-12
**Completed By**: codex
**Source**: moved from TODOs.md

**Delivered**
- Traced and documented the actual symbol-level chunking pipeline used for semantic search embeddings.
- Added a concrete Markdown artifact for `gloggur/embeddings/base.py` with exact `chunk_text` payloads and all symbol annotations (`id`, `name`, `kind`, lines, `signature`, `docstring`, `body_hash`, `language`, embedding preview).
- Recorded runtime provider details observed in this workspace (`local:microsoft/codebert-base` profile running in deterministic local fallback mode).

**Verification**
- Commands run:
  - `scripts/gloggur index . --json`
  - `scripts/gloggur status --json`
  - `scripts/gloggur search "embedding" --top-k 5 --json`
  - `scripts/gloggur inspect gloggur/embeddings/base.py --json`
  - `.venv/bin/python - <<'PY' ... extract_symbols + Indexer._symbol_text + embed_batch ... PY`
  - `.venv/bin/python - <<'PY' ... create_embedding_provider + get_dimension ... PY`
- Results:
  - Index/status/search commands succeeded.
  - Inspect command succeeded for the example file.
  - Extracted symbol/chunk output matched the example file and documented chunk text.
  - Provider probe confirmed local provider fallback mode and 256-dim vectors for this workspace.

**Evidence**
- Files: `/Users/auzi/vinnustofa/gloggur/docs/semantic-search-embedding-breakdown.md`
- PR/commit/issues: local working tree changes

**Follow-ups**
- None

## D2 - Automate Semantic Embedding Breakdown Report

**Completed On**: 2026-02-12
**Completed By**: codex
**Source**: moved from TODOs.md

**Delivered**
- Added `/Users/auzi/vinnustofa/gloggur/generate_embedding_breakdown.py` to generate semantic-embedding breakdown Markdown for any supported source file.
- Script reports symbol-level chunk text (the exact text sent to embedding), parsed annotations, embedding previews, and provider runtime details (including local fallback marker when present).
- Added CLI options for output path, config path, embedding-provider override, preview length, and optional full vector inclusion.

**Verification**
- Commands run:
  - `scripts/gloggur index . --json`
  - `scripts/gloggur search "symbol text embedding chunk" --top-k 5 --json`
  - `.venv/bin/python -m ruff check generate_embedding_breakdown.py`
  - `.venv/bin/python generate_embedding_breakdown.py gloggur/embeddings/base.py --output /tmp/semantic-search-embedding-breakdown-auto.md`
  - `scripts/gloggur inspect generate_embedding_breakdown.py --json`
- Results:
  - Index refreshed successfully after adding the new script file.
  - Script lint checks passed.
  - Script generated markdown successfully at `/tmp/semantic-search-embedding-breakdown-auto.md`.
  - Generated report contains per-symbol chunks/annotations and runtime embedding metadata.

**Evidence**
- Files: `/Users/auzi/vinnustofa/gloggur/generate_embedding_breakdown.py`
- Files: `/private/tmp/semantic-search-embedding-breakdown-auto.md`
- PR/commit/issues: local working tree changes

**Follow-ups**
- None

## R4 - Perfect Reliability: Bootstrap, Preflight, and Self-Healing CLI Execution

**Completed On**: 2026-02-25
**Completed By**: codex
**Source**: moved from TODOs.md

**Delivered**
- Added deterministic preflight runtime selection in `scripts/gloggur` + `src/gloggur/bootstrap_launcher.py` with stable fallback order (`.venv` first, then system Python, then actionable failure).
- Added structured early-failure JSON (`operation=preflight`) with stable `error_code` values (`missing_venv`, `missing_python`, `missing_package`, `broken_environment`), remediation, and detected environment details.
- Added canonical setup/repair helper `scripts/bootstrap_gloggur_env.sh` including deterministic `.gloggur-cache` and `.venv` hydration via `--seed-*-mode symlink|copy`.
- Documented bootstrap/preflight and recovery flows in `README.md` and `docs/AGENT_INTEGRATION.md` for agent workflows that rely on `scripts/gloggur`.

**Verification**
- Commands run:
  - `.venv/bin/python -m pytest tests/unit/test_bootstrap_launcher.py tests/integration/test_bootstrap_wrapper.py tests/integration/test_bootstrap_env_script.py -q`
  - `scripts/gloggur status --json`
  - `GLOGGUR_PREFLIGHT_DRY_RUN=1 GLOGGUR_PREFLIGHT_VENV_PYTHON=/tmp/does-not-exist/bin/python GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS=$(command -v python3) GLOGGUR_PREFLIGHT_PROBE_MODULE=gloggur.bootstrap_launcher scripts/gloggur status --json`
  - `GLOGGUR_PREFLIGHT_DRY_RUN=1 GLOGGUR_PREFLIGHT_VENV_PYTHON=$(command -v python3) GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS=$(command -v python3) GLOGGUR_PREFLIGHT_PROBE_MODULE=gloggur.bootstrap_launcher scripts/gloggur status --json`
  - `GLOGGUR_PREFLIGHT_DRY_RUN=1 GLOGGUR_PREFLIGHT_VENV_PYTHON=$(command -v false) GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS=$(command -v false) scripts/gloggur status --json`
  - `GLOGGUR_PREFLIGHT_DRY_RUN=1 GLOGGUR_PREFLIGHT_VENV_PYTHON=/tmp/does-not-exist/bin/python GLOGGUR_PREFLIGHT_SYSTEM_PYTHONS=$(command -v python3) GLOGGUR_PREFLIGHT_REQUIRED_IMPORTS=definitely_missing_pkg_for_gloggur_bootstrap_test scripts/gloggur status --json`
  - `scripts/bootstrap_gloggur_env.sh --skip-install`
  - `scripts/gloggur search "bootstrap preflight" --top-k 1 --json`
  - `scripts/gloggur inspect . --json`
- Results:
  - R4-focused test suite passed (`22 passed`), including matrix coverage for preflight classification, wrapper behavior, warm-path timing checks, seed modes (`symlink`/`copy`), and exit-code stability.
  - Runtime checks produced deterministic structured JSON for representative `missing_venv` and `missing_package` failure paths with remediation and environment details.
  - Warm-path preflight timings were under the target (<200ms) in dry-run checks.
  - One-command bootstrap (`scripts/bootstrap_gloggur_env.sh --skip-install`) succeeded and `scripts/gloggur` workflows (`status`, `search`, `inspect`) succeeded without relying on bare `gloggur` on PATH.

**Evidence**
- Files: `/Users/auzi/.codex/worktrees/1f29/gloggur/scripts/gloggur`
- Files: `/Users/auzi/.codex/worktrees/1f29/gloggur/src/gloggur/bootstrap_launcher.py`
- Files: `/Users/auzi/.codex/worktrees/1f29/gloggur/scripts/bootstrap_gloggur_env.sh`
- Files: `/Users/auzi/.codex/worktrees/1f29/gloggur/tests/unit/test_bootstrap_launcher.py`
- Files: `/Users/auzi/.codex/worktrees/1f29/gloggur/tests/integration/test_bootstrap_wrapper.py`
- Files: `/Users/auzi/.codex/worktrees/1f29/gloggur/tests/integration/test_bootstrap_env_script.py`
- Files: `/Users/auzi/.codex/worktrees/1f29/gloggur/README.md`
- Files: `/Users/auzi/.codex/worktrees/1f29/gloggur/docs/AGENT_INTEGRATION.md`
- PR/commit/issues: local working tree changes

**Follow-ups**
- None

## R2 - Recover Gracefully from Corrupted SQLite Cache Files

**Completed On**: 2026-02-25
**Completed By**: codex
**Source**: moved from TODOs.md

**Delivered**
- Implemented corruption-first cache reset flow in `CacheManager` with `PRAGMA integrity_check(1)` probing and deterministic corruption classification at cache open.
- Added automatic quarantine-or-delete recovery for corrupted `index.db` plus WAL/SHM sidecars, including idempotent `.corrupt.<timestamp>[.<n>]` naming and explicit one-line corruption notices.
- Added explicit hard-failure path (`CacheRecoveryError`) when both quarantine and delete fail, with CLI mapping to structured `io_failure` output and actionable remediation for JSON and non-JSON surfaces.
- Added provider-independent and concurrency-aware integration coverage to validate deterministic behavior when FAISS is unavailable and when two recovery attempts race on the same corrupted cache.

**Behavioral Impact**
- Core commands (`status`, `search`, `inspect`, `clear-cache`, `index`) now recover deterministically from corrupted SQLite cache artifacts without manual cleanup in normal cases.
- If automatic recovery is impossible due to filesystem constraints, commands fail non-zero with stable machine-readable payloads and clear operator remediation, without traceback leakage.

**Verification**
- Commands run:
  - `.venv/bin/python -m pytest tests/unit/test_cache.py tests/unit/test_cli_main.py tests/integration/test_cli.py tests/integration/test_corruption_recovery_integration.py -q`
  - `tmp_dir=$(mktemp -d /tmp/gloggur-r2-manual-XXXXXX); printf 'broken sqlite bytes' > "$tmp_dir/index.db"; printf 'broken wal' > "$tmp_dir/index.db-wal"; printf 'broken shm' > "$tmp_dir/index.db-shm"; GLOGGUR_CACHE_DIR="$tmp_dir" gloggur status; rc=$?; echo "__RC=$rc"; ls -1 "$tmp_dir"`
- Results:
  - Targeted R2 test suite passed (`95 passed`) with no corruption-recovery regressions.
  - Manual probe confirmed one-line corruption notice, quarantined DB artifact, sidecar cleanup, fresh DB rebuild, and successful deterministic status output (`__RC=0`).

**Evidence**
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/indexer/cache.py`
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/cli/main.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/unit/test_cache.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/unit/test_cli_main.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/integration/test_cli.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/integration/test_corruption_recovery_integration.py`
- PR/commit/issues: local working tree changes

**Follow-ups**
- None

## R1 - Harden OS-Level Failure Handling (Permissions, Read-only FS, Disk Full)

**Completed On**: 2026-02-25
**Completed By**: codex
**Source**: moved from TODOs.md

**Delivered**
- Added stable I/O error taxonomy and payload contract (`permission_denied`, `read_only_filesystem`, `disk_full_or_quota`, `path_not_writable`, `unknown_io_error`) via `StorageIOError` + centralized classification/messaging in `src/gloggur/io_failures.py`.
- Wrapped cache/vector/metadata/config filesystem and sqlite failure surfaces so core commands emit deterministic structured `io_failure` payloads (JSON) and actionable human-readable stderr (non-JSON), without traceback leakage.
- Hardened `clear-cache` vector artifact handling:
  - deterministic delete-failure mapping for vector artifacts,
  - no pre-delete loading of existing vector artifacts (`VectorStore(..., load_existing=False)`), so malformed artifact files no longer block cleanup when removable.
- Expanded core-command regression coverage across `status`, `search`, `inspect`, `clear-cache`, and `index` for all category mappings and config/db failure paths.

**Behavioral Impact**
- Hard I/O failures now fail non-zero with stable, machine-readable error contracts suitable for CI parsing.
- Human-facing output now consistently includes operation/path, probable cause, remediation steps, and original exception detail.
- Normal happy-path command flows remain operational while failure handling is stricter and more predictable.

**Verification**
- Commands run:
  - `.venv/bin/python -m pytest tests/unit/test_io_failures.py tests/unit/test_cli_main.py tests/integration/test_io_failures_integration.py -q`
  - `.venv/bin/python -m pytest tests/integration/test_cli.py::test_cli_index_search_status_and_clear_cache -q`
  - `tmp_dir=$(mktemp -d /tmp/gloggur-r1-inspect-XXXXXX); cat > "$tmp_dir/sample.py" <<'PY' ... PY; GLOGGUR_CACHE_DIR="$tmp_dir/.cache" gloggur inspect "$tmp_dir" --json; echo "__RC=$?"`
  - `GLOGGUR_CACHE_DIR=/dev/null/cache gloggur search add; echo "__RC=$?"`
- Results:
  - R1 verification suite passed (`93 passed`), including category matrix + integration checks for unwritable cache parent across core commands.
  - Core happy-path integration flow passed (`1 passed`) confirming no regression in baseline index/search/status/clear-cache behavior.
  - Manual inspect happy-path run returned structured JSON and exit `0`.
  - Manual failure-path run returned deterministic actionable non-JSON message (`IO failure [path_not_writable]`) and exit `1`.

**Evidence**
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/io_failures.py`
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/cli/main.py`
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/storage/vector_store.py`
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/storage/metadata_store.py`
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/indexer/cache.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/unit/test_io_failures.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/unit/test_cli_main.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/integration/test_io_failures_integration.py`
- PR/commit/issues: local working tree changes

**Follow-ups**
- None

## R3 - Concurrency and Race-Condition Hardening for Cache/Vector Operations

**Completed On**: 2026-02-25
**Completed By**: codex
**Source**: moved from TODOs.md

**Delivered**
- Implemented bounded cross-process writer serialization with explicit retry/backoff and deterministic lock-timeout error contracts in `src/gloggur/indexer/concurrency.py`.
- Enforced partial-publication safety in index lifecycle: metadata invalidated before rebuild and republished only after artifact persistence, preventing false healthy status after interrupted writes.
- Added integration coverage for concurrent writer contention (`index` and `clear-cache`) and deterministic non-hanging outcomes under active lock contention.
- Documented explicit concurrency contract and operator tuning knobs in `docs/AGENT_INTEGRATION.md` (safe concurrency modes, lock semantics, timeout behavior, and consistency guarantees).

**Behavioral Impact**
- Concurrent usage now has deterministic writer behavior: one writer proceeds, contending writers fail fast with structured lock-timeout guidance rather than hanging.
- Reader-visible status/search semantics remain consistent under interrupted indexing (`needs_reindex=true` until recovery index succeeds).
- Operational behavior under contention is now documented and test-backed for agent workflows.

**Verification**
- Commands run:
  - `.venv/bin/python -m pytest tests/unit/test_concurrency.py tests/integration/test_concurrency_integration.py -q`
- Results:
  - Concurrency verification suite passed (`7 passed`), covering:
    - bounded backoff/retry schedule and timeout mapping (`operation=acquire cache write lock`)
    - concurrent index contention with deterministic success/fail outcomes
    - clear-cache contention fast-fail behavior
    - interrupted index run preserving `needs_reindex=true` until recovery.

**Evidence**
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/indexer/concurrency.py`
- Files: `/Users/auzi/vinnustofa/gloggur/src/gloggur/indexer/indexer.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/unit/test_concurrency.py`
- Files: `/Users/auzi/vinnustofa/gloggur/tests/integration/test_concurrency_integration.py`
- Files: `/Users/auzi/vinnustofa/gloggur/docs/AGENT_INTEGRATION.md`
- PR/commit/issues: local working tree changes

**Follow-ups**
- None
