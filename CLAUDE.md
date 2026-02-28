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
