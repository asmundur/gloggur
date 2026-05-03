# Grill-Me Skill

You are running the `grill-me` skill for **gloggur**. Your job is to reach a shared design concept before implementation starts.

## When to Use This

Use this skill when:
- The request is ambiguous
- There are multiple reasonable designs
- The work changes module boundaries, terminology, or public interfaces
- The user has a clear goal but the design is still fuzzy

`/feature-start` may invoke this behavior automatically. You can also run it directly when you need a deeper design interrogation.

## Step 1 — Load the Existing Design Context

Before you ask questions, read these files when they exist:
- `.agent-scaffold.json`
- `.agents/context/ubiquitous-language.md`
- `.agents/architecture/module-map.md`
- Any existing `.agents/plans/<feature-slug>.md` for the same feature

If only legacy `.claude/...` planning artifacts exist, read them as migration evidence and write canonical updates under `.agents/`.

Then explore `src/gloggur` to understand the current implementation shape, similar features, and test boundaries.

## Step 2 — Interview Relentlessly

Ask focused questions until the important design branches are resolved or explicitly parked.

Good questions are:
- Scope-shaping
- Interface-shaping
- Dependency-shaping
- Risk-shaping
- Constraint-shaping

Do not stop at surface preferences. Walk down each branch of the design tree one dependency at a time.

Examples of the decisions you should force into the open:
- What problem are we solving?
- What is out of scope?
- Which module owns the behavior?
- Which public interface changes?
- What should the user observe when this is done?
- Which feedback loops prove the change works?
- What can be postponed without blocking implementation?

If a decision is truly unresolved, mark it as an explicit open question or parked decision. Do not silently guess.

## Step 3 — Write or Update the Feature Spec

Create or update `.agents/plans/<feature-slug>.md` with:
- Summary
- Goal
- Non-goals
- Constraints
- Canonical terms to use
- Affected modules
- Interface changes
- Acceptance criteria
- Feedback loops
- Open questions / parked decisions

This file is the durable record of the shared design concept. Keep it concise, but do not leave hidden decisions buried in chat history.

## Step 4 — Decide Whether Planning Can Continue

You may hand back to `/feature-start` or the user once:
- The blocking design questions are resolved
- The remaining open questions are explicitly non-blocking
- The feature spec is concrete enough for implementation

If the design is still under-specified, keep grilling. The purpose of this skill is to prevent implementation from starting on vague intent.
