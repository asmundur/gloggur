# Ubiquitous-Language Skill

You are running the `ubiquitous-language` skill for **gloggur**. Your job is to build and maintain a shared vocabulary between the user, the codebase, and the AI.

## Your Goal

Create or update `.agents/context/ubiquitous-language.md` so planning and implementation can use the same canonical terms.

## Step 1 — Gather Terminology

Scan terminology from:
- `.agent-scaffold.json`
- `src/gloggur`
- Tests
- README and other docs
- Task history in `.beads/issues.jsonl` when present
- Existing `.agents/plans/` specs when present

Look for:
- Domain nouns
- User-visible labels
- Workflow verbs
- Names of modules, subsystems, and external integrations
- Places where two or more names appear to refer to the same thing

## Step 2 — Normalize the Language

For each important concept, decide:
- The canonical term
- Aliases to avoid
- A concise definition
- The source references that justify the choice

If the codebase uses conflicting terms and the correct one is not obvious, ask the user to resolve the ambiguity instead of guessing.

## Step 3 — Write the Glossary

Create or update `.agents/context/ubiquitous-language.md` with Markdown tables like:

| Canonical Term | Aliases to Avoid | Definition | Source Reference |
|---|---|---|---|
| ... | ... | ... | ... |

Use sections when helpful:
- Domain concepts
- User actions / workflows
- Modules / boundaries
- External systems

## Step 4 — Apply the Result

After writing the glossary:
- Prefer the canonical terms in future planning and code changes
- Flag terminology drift when it appears
- Update the glossary whenever a feature introduces or renames a core concept

This file should stay readable enough that a human can keep it open while planning.
