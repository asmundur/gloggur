# Improve-Architecture Skill

You are running the `improve-architecture` skill for **gloggur**. Your job is to find architecture hotspots and turn them into clearer, deeper modules with simpler interfaces.

## When to Use This

Use this skill when:
- A feature touches too many files or layers
- The same concept is scattered across small, shallow modules
- Testing feels harder than it should
- A retro identified architecture debt or weak boundaries

## Step 1 — Explore the Existing Structure

Inspect `src/gloggur` and identify:
- Clusters of related code that should probably live behind one boundary
- Public interfaces that are too wide or inconsistent
- Places where the implementation detail leaks into calling code
- Tests that are forced to mock too much because the boundary is poor

## Step 2 — Propose Better Module Boundaries

Favor deep modules:
- More functionality hidden behind a smaller interface
- Fewer boundary concepts for callers to understand
- Clear ownership of related behavior and data

Avoid shallow-module churn:
- Do not split code into tiny wrappers unless the boundary becomes simpler
- Do not create layers that only pass data through unchanged

## Step 3 — Write or Update the Module Map

Create or update `.agents/architecture/module-map.md` with:
- Module name
- Purpose
- Public interface
- Dependencies
- Key invariants
- Recommended interface-level tests

Use Markdown tables when they help, for example:

| Module | Purpose | Public Interface | Depends On | Interface Tests |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## Step 4 — Feed the Result Into Planning

For any planned feature or refactor:
- Name which modules are affected
- Name which public interfaces change
- Keep tests aligned with those interfaces

This skill is a planning and design aid. Do not refactor automatically unless the user asked you to implement the architecture changes.
