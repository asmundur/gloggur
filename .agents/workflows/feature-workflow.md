# Feature Workflow — gloggur

This document defines the design-first feature development pipeline. It is referenced by the `/feature-start` skill.

---

## Stage 0 — Shared Design Alignment (Interactive)

**Skills:** current harness `/feature-start` and `/grill-me`
**Mode:** Interactive with user
**Agent:** Current agent instance

Goals:
- Load `.agents/context/ubiquitous-language.md` and `.agents/architecture/module-map.md` when they exist
- Reach a shared design concept for the feature
- Resolve high-impact decisions or explicitly park them
- Record the decisions in `.agents/plans/<feature-slug>.md`
- For debugging and CI work, identify the failure ownership boundary and compare last passing vs first failing state before proposing fixes

**Exit condition:** The important design decisions have been resolved or clearly parked.

---

## Stage 1 — Feature Spec & Approval (Interactive)

**Skill:** current harness `/feature-start`
**Mode:** Interactive with user
**Agent:** Current agent instance

Goals:
- Clarify the feature request
- Explore codebase for reusable patterns
- Define acceptance criteria (Given/When/Then)
- Record affected modules, interface changes, and chosen feedback loops
- Create implementation slices and beads-ready task slices
- Create feature branch
- Keep user-facing process overhead proportional to the size and ambiguity of the task

**Exit condition:** User gives explicit approval to proceed.

**Handoff to Stage 2** — Pass to Feature-Implementation agent:
- Spec path
- Shared design decisions
- Affected modules
- Interface changes
- Chosen feedback loops
- Acceptance criteria
- Implementation slices
- Relevant file paths
- Build command: `.venv/bin/python -m build`
- Typecheck command: `.venv/bin/python -m mypy src tests scripts`
- Lint command: `.venv/bin/python -m ruff check .`
- Browser verification command: `not configured`
- Test command: `.venv/bin/python -m pytest`

---

## Stage 2 — Implementation

**Agent:** Implementation agent for the current harness
**Model:** Current harness default unless an implementation-specific model is explicitly configured
**Mode:** Autonomous (no user interaction)

The agent will:
1. Read `.agents/anti-patterns.md`
2. Read the approved feature spec and load the glossary / module map when present
3. Explore identified files to understand patterns
4. Implement the feature in red/green/refactor slices
5. Run the configured feedback loops as the implementation evolves
6. Update shared terms or module boundaries if the feature changes them
7. Clean up (dead code, naming, DRY)
8. Report completion

**Context budget check:** If the feature is large enough to risk hitting context limits, split into sub-tasks and complete them sequentially. Never half-finish an implementation.

---

## Stage 2.5 — Human Review (MANDATORY CHECKPOINT)

**Mode:** Interactive with user

Run:
```bash
git diff main...HEAD
```

Present the diff to the user and ask: "Ready to commit? Approve to continue to Stage 3."

**Do not proceed until the user explicitly approves.**

---

## Stage 3 — Commit

**Agent:** Git manager for the current harness
**Model:** Current harness default unless a git-management model is explicitly configured
**Mode:** Autonomous

The agent will:
1. Run `git status` + `git diff`
2. Stage files individually
3. Create a semantic commit (feat/fix/refactor/test/chore)
4. Verify with `git log`
5. Report commit hash

**Never push** without explicit user instruction.

---

## Workflow Summary

```
User: /feature-start
       │
       ▼
[Stage 0] Shared design ─────────► Decisions resolved
       │
       ▼
[Stage 1] Feature spec ──────────► User approval
       │
       ▼
[Stage 2] Feature-Implementation ► Tests passing
       │
       ▼
[Stage 2.5] Human review diff ──► User approval
       │
       ▼
[Stage 3] Git-Manager commit ───► Done
       │
       ▼
       (optional) /retro
```
