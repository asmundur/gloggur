## Required workflow

# Beads Issue Tracking

This project uses [Beads (bd)](https://github.com/steveyegge/beads) for issue tracking.

## Core Rules

- Start with `bd ready` before taking net-new work.
- Claim a task with `bd update <id> --claim`.
- Create new work with `bd create --title "..." -p 2`.
- Use `bd show <id>` for full task context and `bd close <id>` when the work is done.
- Run `bd prime` when you need the current Beads workflow context.

## Coexistence

`TODOs.md` and `DONEs.md` remain part of the repo during the transition:
- existing historical/open Markdown backlog stays there until explicitly migrated or retired
- new tasks created after the Beads rollout should go into `bd`
- keep `TODOs.md` / `DONEs.md` for verification evidence and historical records while both systems coexist
- retire Markdown tracking only after historical open tasks are imported and verified in Beads or explicitly archived/cancelled with provenance, `.beads/issues.jsonl` stays in parity with the live Beads DB, hooks stay stable, and one release cycle passes with no new Markdown-only work
- if Beads sync/export regresses, keep Markdown tracking active and reopen the retirement decision instead of forcing cutover

Operational note:
- run `bd` commands serially in this repo; parallel `bd` invocations against the embedded Dolt store have reproduced local tracker panics

If the index is missing or stale, update it:

```bash
gloggur index . --json
```

If `gloggur` is not on PATH (e.g., codex-cli), use:

```bash
scripts/gloggur status --json
scripts/gloggur index . --json
```


### 2) Use semantic search selectively (avoid the slowdown tax)

Use semantic search when it materially reduces uncertainty, e.g.:
- You don’t know where a symbol is defined
- You need to locate where a concept is implemented
- You’re dealing with cross-cutting behavior across multiple modules

```bash
gloggur search "<query>" --top-k 5 --json
```

If `gloggur` is not on PATH:

```bash
scripts/gloggur search "<query>" --top-k 5 --json
```

Do **not** use semantic search for:
- Editing files you already have open
- Small local refactors
- Obvious symbol locations (prefer normal text search / jump-to-definition)


Silent failures are absolutely forbidden. Fail fast, fail early, fail loud.


## Notes

- Use `.venv` for running Python modules.
- Check `gloggur status --json` to confirm the index is current (or `scripts/gloggur status --json` if the CLI is not on PATH).
- Cache data is stored in `.gloggur-cache`; do **not** commit it.
- If you add or rename files, re-run `gloggur index . --json`.
- Keep existing backlog/history in `TODOs.md` and `DONEs.md`, but track new tasks in Beads unless a task is explicitly part of the Markdown migration history.

For more detail, see `docs/AGENT_INTEGRATION.md`.


When handing of work, always include a conventional commit message for the changes


<!-- BEGIN BEADS INTEGRATION -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for all new issue tracking. Existing `TODOs.md` / `DONEs.md` entries remain during the migration as historical backlog items and verification records, but net-new work should go into `bd`.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

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

bd create "Concrete follow-up discovered while landing bd-123" \
  --type task \
  --priority 1 \
  --description "Specific follow-up needed after inspecting src/... and tests/... during bd-123." \
  --design "Call out the exact files, contracts, or parser/search/index surfaces likely to change." \
  --acceptance "- Define the shipped contract\n- Add or adjust targeted regression coverage\n- Keep scope narrower than the parent issue" \
  --notes "Discovered during bd-123; include the exact probe, failure, or code-path evidence that surfaced it." \
  --estimate 90 \
  --deps discovered-from:bd-123 \
  --json
```

Prefer high-signal Beads issues by default: reference concrete paths, describe current behavior before proposing changes, fill `--design` / `--acceptance` / `--notes` / `--estimate`, and avoid vague umbrella wording unless the issue is intentionally an umbrella.

**Claim and update:**

```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs with git:

- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed!

### Important Rules

- ✅ Use bd for all new task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create new markdown TODO lists for net-new work
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

<!-- END BEADS INTEGRATION -->

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd export -o .beads/issues.jsonl
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
