# gloggur

## Project Overview

Python CLI indexing and retrieval engine for codebases.

- **Tech Stack:** Python, Click, Pydantic, Tree-sitter, SQLite, FAISS, pytest
- **Language:** Python
- **Source Directory:** src/gloggur
- **Architecture:** CLI-first package with parser, indexer, search, graph, storage, watch, and support modules

## Essential Commands

```bash
# Apply or refresh scaffold
../agent-bootstrap/scripts/scaffold.sh . all

# Build
.venv/bin/python -m build

# Test
.venv/bin/python -m pytest

# Run
scripts/gloggur --help
```

## Feedback Loops

- **Typecheck:** `.venv/bin/python -m mypy src tests scripts`
- **Lint:** `.venv/bin/python -m ruff check .`
- **Browser verification:** `not configured`

If a feedback-loop command is set to `not configured`, skip it. Otherwise, use the fastest applicable loop before moving on to larger changes.

## Scaffold Hydration

Run `/bootstrap` after first scaffold adoption, or when project-specific scaffold values need intentional re-hydration. That skill inspects the existing repository, derives project-specific values, updates `.agent-scaffold.json`, and deterministically refreshes scaffolded docs/config through the scaffold renderer where those values are used.

Re-run `../agent-bootstrap/scripts/scaffold.sh . all` whenever you want to pull the latest forward scaffold changes into the project. It is the only forward refresh path; routine scaffold refreshes do not require `/bootstrap` unless project facts need to be re-read from the codebase.

## Slash-Command Skills

When the user invokes a slash command such as `/feature-start`, `/retro`, or `/tdd`, resolve and load the project-local skill before planning, task tracking, or implementation. Treat the system-provided skill list as a fallback, not as exhaustive.

Resolution order:
1. Current harness skill directory: `.codex/skills/<command>.md`, `.claude/skills/<command>.md`, or `.antigravity/skills/<command>.md`
2. Other generated skill directories for the same command
3. Related workflow docs such as `.agents/workflows/feature-workflow.md`
4. Built-in or system-provided skill instructions

The local skill's rules override the default Beads gate when they are more specific. In particular, `/feature-start` is a planning-only workflow: load the skill, produce the feature spec, present the implementation contract, and stop for explicit user approval before writing code.

## Project-Specific Safety Constraints

Use this section only for constraints that remain genuinely local after `/resolve-adopted-artifacts` has separated universal workflow rules from repository-specific values. Promote general agent-workflow lessons to the upstream scaffold templates instead of storing them here. Keep any local constraints concrete: exact commands, paths, protected files, or operational contracts.

- Run `bd` commands serially in this repository. Parallel `bd` invocations against the embedded Dolt store have reproduced local tracker panics.
- Use `.venv` for Python modules and quality gates.
- Check the index before relying on semantic search: `scripts/gloggur status --json`. If the index is missing or stale, run `scripts/gloggur index . --json`; use `gloggur ...` directly only when it is known to be on `PATH`.
- Use semantic search only when it materially reduces uncertainty: unknown symbol locations, concept discovery, or cross-cutting behavior. Prefer `rg` or direct file reads for local edits, small refactors, and obvious symbols.
- Cache data lives in `.gloggur-cache`; do not commit it. If files are added or renamed, re-run `scripts/gloggur index . --json`.
- Silent failures are forbidden. Fail fast, fail early, and fail loud.
- `TODOs.md` and `DONEs.md` remain as migration evidence and historical records. Do not add net-new work there; use Beads unless the work is explicitly part of Markdown backlog migration. Retire Markdown tracking only after historical open work is imported or explicitly archived/cancelled with provenance, `.beads/issues.jsonl` is in parity with the live Beads DB, hooks are stable, and one release cycle passes with no Markdown-only work.
- `.githooks/pre-commit.local` is this repo's local hook extension. It refreshes `docs/semantic-search-embedding-breakdown-base.md` when embedding/indexer/parser inputs change.

## Code Style Guidelines

- Match the style of surrounding code
- Functions should do one thing
- Name things for what they are, not how they're implemented
- Validate at system boundaries (user input, external APIs) — trust internal code
- No dead code, no commented-out blocks, no TODOs left behind after a feature
- Tests are not optional

## Task Tracking — Beads

This project uses [beads](https://github.com/steveyegge/beads) (`bd`) for task tracking. Issue prefix: `gloggur`.

### Mandatory Gate — One Task Per Tracked Body of Work

Default to no Beads task for conversational or read-only work. Use Beads only when there is a concrete tracked body of work whose outcome needs to survive the chat: implementation, tests, docs, architecture artifacts, durable decisions, bug investigations that will produce follow-up implementation, or a multi-step task with durable state.

Do **not** create or claim Beads tasks just to talk about the codebase, inspect command output, run a retro, answer a question, or decide whether work is needed. Discussion, explanation, read-only exploration, lightweight review, command-output inspection, and agent/process retros stay out of Beads unless the user explicitly asks to track them or they turn into an implementation/change task. If the user invokes a slash-command skill, load that skill first; its task-tracking rules override this default when they are more specific.

Do **not** split bookkeeping into extra tasks. If investigation and a narrow fix are one coherent body of work, use one task. Create a discovered follow-up only for genuinely separate remaining work, not for the act of tracking, confirming, or finishing the current task.

**Before starting any tracked body of work**, you must have a single claimed Beads task in hand. One task covers all the edits for that work.
1. **Run `bd ready --json`** to inspect open tasks.
2. **If a match exists**, claim it: `bd update <id> --claim --json`. Announce the claimed ID in your first response.
3. **If no match exists**, create one first, then claim it. Do not start planning or editing until the task is created and claimed.
4. **Announce the task ID** in your response before any plan or code (e.g., `Working on gloggur-xxx — <title>`).

### Working with Tasks

A high-quality Beads task has a concrete title, scope-bearing description, design notes, observable acceptance criteria, evidence notes, estimate, and dependencies where relevant. The description states current behavior, why it matters, and scope boundaries. The design names likely files, interfaces, contracts, constraints, and non-goals. Notes capture evidence such as user requests, probes, failing commands, relevant commits, or focused test nodes.

Existing markdown backlogs such as `TODOs.md` or `DONEs.md` may be kept only as migration evidence or historical records when a repository is being converted. Do not add net-new work there, and do not keep a parallel side ledger for future tasks.

**Create new issues:**

```bash
bd create "Tight, concrete issue title" \
  --type bug|feature|task \
  --priority 1 \
  --description "Current behavior, why it matters, scope boundaries, and the exact code/docs/tests paths involved." \
  --design "Implementation touchpoints in src/... tests/... docs/... plus key constraints and non-goals." \
  --acceptance "- Observable outcome 1\n- Observable outcome 2\n- Verification or fail-loud contract" \
  --notes "Current evidence: direct probes, failing commands, relevant commits, and focused test nodes." \
  --estimate 120 \
  --json

bd create "Concrete follow-up discovered while landing gloggur-123" \
  --type task \
  --priority 1 \
  --description "Specific follow-up needed after inspecting src/... and tests/... during gloggur-123." \
  --design "Call out the exact files, contracts, or parser/search/index surfaces likely to change." \
  --acceptance "- Define the shipped contract\n- Add or adjust targeted regression coverage\n- Keep scope narrower than the parent issue" \
  --notes "Discovered during gloggur-123; include the exact probe, failure, or code-path evidence that surfaced it." \
  --estimate 90 \
  --deps discovered-from:gloggur-123 \
  --json
```

Create follow-up issues sparingly: only when the new work is separable from the current task and should survive as its own future work item.

**Complete work:**
```bash
bd close <id> --reason "done" --json
```

### Git Integration & Clone Contract

`.beads/issues.jsonl` is the git-tracked snapshot; the pre-commit hook refreshes it via `bd export --no-memories` and auto-stages changes, so task state travels with commits. Do not edit `.beads/issues.jsonl` by hand. Do not bypass the hook (`--no-verify`).

Agents must not run `git add`, `git commit`, or `git push` as an automatic session-close workflow. Do not run `bd dolt push` as session-close workflow. Commits and pushes require explicit user approval for that action and scope.

Important: the presence of `.beads/config.yaml`, `.beads/clone-contract.json`, or `.githooks/pre-commit` does **not** by itself prove that the local Beads database has been bootstrapped. Treat “files scaffolded” and “tool operational” as separate states.

- Fresh clones must bootstrap local Beads state from `.beads/issues.jsonl`:
```bash
bd bootstrap --yes --json
git config core.hooksPath .githooks
bd status --json
```
- `.githooks/` is the only supported git hook path for this repository.
- Machine consumers should read `.beads/clone-contract.json` instead of inferring readability from `.beads/metadata.json`.
- If local Beads runtime state is stale, confirm `git config --get core.hooksPath` points at `.githooks`, remove stale local pins such as `.beads/dolt-server.port`, clear stale `dolt-server.*` lock/log artifacts only when no live process owns them, then rerun `bd status --json` or `bd ready --json`.
- Do not create markdown TODO trackers or side ledgers for net-new work. Use the project task tracker.

### Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below:
1. **File issues for remaining work** - Create issues for anything that needs follow-up, using `--deps discovered-from:<id>`
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Hand off** - Provide context for next session along with a fitting conventional commit message

If the unit of work changed any git-tracked files, the handoff must include a meaningful, high-signal conventional commit message. Do not end a tracked-file work session without one.

## Durable Artifacts

- **Feature specs:** `.agents/plans/<feature-slug>.md`
- **Ubiquitous language:** `.agents/context/ubiquitous-language.md`
- **Module map:** `.agents/architecture/module-map.md`

These files live under `.agents/` so every supported agent provider shares the same project memory. Reuse and update them instead of recreating design context from scratch in provider-specific folders.
If a repository still has legacy `.claude/plans/`, `.claude/context/`, or `.claude/architecture/` artifacts without `.agents/` equivalents, read them as migration evidence, then write the canonical updated artifact under `.agents/`.

## Working Agreements

- Explore the codebase and understand existing patterns before implementing anything
- Reach a shared design concept before writing code; ambiguous work should go through a grilling/interview phase first
- Plan and confirm acceptance criteria with the user before writing code
- Write or update a feature spec in `.agents/plans/` before implementation starts
- Load the ubiquitous-language glossary and module map when present before planning or implementation
- Keep terminology aligned with the glossary; update it when the domain language changes
- Design around module boundaries and simple interfaces, especially for refactors
- Implement in small red/green/refactor steps and stay within the fastest available feedback loop
- When adding or debugging runtime environment variables, verify the full delivery path: documented env template, secret/project env source, deploy template interpolation, service/container injection, and application settings. Add or update a contract test so documented runtime variables cannot drift from the deployment service environment.
- Get explicit user approval before committing changes
- Run the full test suite before any commit — do not commit with failing tests
- Prefer interface-level tests over tests of internal implementation details
- Stage files individually; never blindly add everything
- Semantic commit messages: `<type>(<scope>): summary` with a body explaining *why*
- When handing work back with tracked-file changes, always provide a meaningful, high-signal conventional commit message even if no commit is being created yet
- Never force-push, never bypass commit hooks
- Never implement beyond what was agreed
