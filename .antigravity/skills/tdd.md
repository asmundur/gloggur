# TDD Skill

You are running the `tdd` skill for **gloggur**. Your job is to drive implementation through small feedback loops instead of large unverified code drops.

## Your Default Loop

Work in strict red/green/refactor cycles:
1. Pick the next smallest observable behavior from the approved feature spec
2. Write the failing test or other failing observable check first
3. Run the fastest relevant feedback loop and confirm it fails for the expected reason
4. Make the smallest code change that can make the check pass
5. Run the relevant feedback loops again
6. Refactor while the checks stay green
7. Repeat

## Required Inputs

Before starting, read:
- The approved `.agents/plans/<feature-slug>.md`
- `.agents/context/ubiquitous-language.md` when it exists
- `.agents/architecture/module-map.md` when it exists

If the approved spec or shared context still exists only under legacy `.claude/...` paths, read it as migration evidence and keep new updates under `.agents/`.

## Feedback Loops

Use these commands when they are configured:
- Typecheck: `.venv/bin/python -m mypy src tests scripts`
- Lint: `.venv/bin/python -m ruff check .`
- Browser verification: `not configured`
- Tests: `.venv/bin/python -m pytest`

If a command is literally `not configured`, skip it. Otherwise, treat it as part of the development speed limit.

## How Small Is Small Enough?

A slice is small enough when:
- A failing check clearly points at the last edit
- The change can be explained by one acceptance criterion or one sub-behavior
- The next refactor can still happen with confidence

If you find yourself writing a lot of code before running a check, you are outrunning your headlights. Stop and reduce the slice size.

## Testing Guidance

- Prefer tests at the module interface
- Avoid locking tests to internal implementation structure
- Mock only what you do not own or cannot conveniently exercise for real
- Use browser verification for user-facing behavior when the project has it configured

This skill is the default implementation style for the Feature-Implementation agent.
