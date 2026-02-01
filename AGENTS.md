# Agent Instructions

Gloggur is a tool **for** coding agents. When working in this repository, you are expected to use Gloggur to orient yourself in the codebase before making changes.

## Required workflow

1. **Index the repository** (first time per workspace):
   ```bash
   gloggur index . --json
   ```
2. **Search for relevant symbols** before editing files:
   ```bash
   gloggur search "<query>" --top-k 5 --json
   ```
3. **Validate your work** when appropriate:
   ```bash
   gloggur validate . --json
   ```

## Notes

- Check `gloggur status --json` to confirm the index is current.
- Cache data is stored in `.gloggur-cache`; do **not** commit it.
- If you add or rename files, re-run `gloggur index . --json`.

For more detail, see `docs/AGENT_INTEGRATION.md`.
