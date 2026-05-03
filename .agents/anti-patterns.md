# Anti-Patterns

Hard constraints for all agents working on gloggur. These are non-negotiable.

## Workflow Anti-Patterns

- **Implementing without accepted criteria** — Always get explicit user approval on what "done" means before writing code. Use Given/When/Then format.
- **Implementing without a shared design concept** — Resolve the important design decisions first and capture them in `.agents/plans/<feature-slug>.md`.
- **Skipping the planning stage** — Stage 1 exists to avoid wasted implementation effort. Never jump straight to code.
- **Treating built-in skill listings as exhaustive** — When the user invokes a slash command, load the project-local skill from `.codex/skills/`, `.claude/skills/`, or `.antigravity/skills/` before acting. Local slash-command rules are the workflow contract.
- **Ignoring the glossary or module map when they exist** — Shared language and module boundaries are part of the source of truth for future changes.
- **Allowing term drift** — If the code, docs, and conversations start using different names for the same concept, update the ubiquitous language before proceeding.
- **Committing without running tests** — `.venv/bin/python -m pytest` must pass before any commit. No exceptions.
- **Creating commits without user review** — Stage 2.5 human review is mandatory. Never skip it.
- **Force-pushing or hard-resetting** — These are destructive. Ask the user first.
- **Bypassing hooks** — Never run `git commit --no-verify` — it skips the beads task handoff.
- **Creating task-tracking noise** — Do not create Beads tasks for discussion, read-only inspection, retros, or task-management bookkeeping. One coherent body of work gets one task; follow-ups are only for separate future work.
- **Treating scaffold presence as proof that tooling is ready** — Generated files and hook scripts are not the same thing as a bootstrapped local tool state. Verify operational readiness explicitly.
- **Ending tracked-file work without a commit message handoff** — If git-tracked files changed, the handoff must include a meaningful, high-signal conventional commit message.

## Code Quality Anti-Patterns

- **Implementing beyond the acceptance criteria** — Do exactly what was agreed. Extra features introduce risk and aren't reviewed.
- **Duplicating code instead of reusing** — Explore `src/gloggur` for existing implementations before writing new ones.
- **Growing shallow modules by default** — Prefer a smaller number of deeper modules with simple interfaces over many thin layers with leaky boundaries.
- **Changing module internals without checking the public interface** — Design the boundary first, then verify behavior through that boundary.
- **Adding error handling for impossible cases** — Only validate at system boundaries. Don't guard against things that can't happen.
- **Writing comments that describe what the code does** — Well-named identifiers already do that. Only comment the non-obvious *why*.
- **Leaving debug code, TODO markers, or dead code** — Clean up before the commit stage.

## Testing Anti-Patterns

- **Skipping tests "because it's simple"** — Tests catch issues humans miss. No feature is too small to test.
- **Skipping the red phase** — Write the failing test or other observable check first so you know the change is necessary.
- **Outrunning feedback loops** — Do not batch large code drops before running the fastest available typecheck, lint, browser, or test loop.
- **Treating optional feedback loops as optional thinking** — If `.venv/bin/python -m mypy src tests scripts`, `.venv/bin/python -m ruff check .`, or `not configured` is configured, use it.
- **Testing implementation details** — Test observable behavior, not internal structure.
- **Testing below the module boundary by default** — Prefer interface-level tests unless the risk truly lives inside the module.
- **Mocking what you don't own** — Prefer real integrations at system boundaries when feasible.

---

*This file is updated by the `/retro` skill. New entries are added when patterns are discovered through retrospectives.*
