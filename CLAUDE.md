## Required workflow
start each session by running ./.venv/bin/activate 


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
- Keep `TODOs.md` and `DONEs.md` for tasks the user mentions that need to be done later; move completed items to DONEs with verification evidence.

For more detail, see `docs/AGENT_INTEGRATION.md`.


When handing of work, always include a conventional commit message for the changes