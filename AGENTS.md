## Required workflow
start each session by running ./.venv/bin/activate 

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
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

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
