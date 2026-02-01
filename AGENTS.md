# Agent Instructions

Gloggur is a tool **for** coding agents. When working in this repository, you are expected to use Gloggur to orient yourself in the codebase before making changes.

use .venv for running python modules. 

## Required workflow

1. **Index the repository** (first time per workspace):
   ```bash
   gloggur index . --json
   ```
   If `gloggur` is not on PATH (e.g., codex-cli), use:
   ```bash
   scripts/gloggur index . --json
   ```
2. **Search for relevant symbols** before editing files:
   ```bash
   gloggur search "<query>" --top-k 5 --json
   ```
   Or, if needed:
   ```bash
   scripts/gloggur search "<query>" --top-k 5 --json
   ```
3. **Validate your work** when appropriate:
   ```bash
   gloggur validate . --json
   ```
   Or, if needed:
   ```bash
   scripts/gloggur validate . --json
   ```

## Notes

- Check `gloggur status --json` to confirm the index is current (or `scripts/gloggur status --json` if the CLI is not on PATH).
- Cache data is stored in `.gloggur-cache`; do **not** commit it.
- If you add or rename files, re-run `gloggur index . --json`.

For more detail, see `docs/AGENT_INTEGRATION.md`.
