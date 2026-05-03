# Bootstrap Skill

You are running the `bootstrap` skill for **gloggur**. Your job is to hydrate the scaffold by reading the existing repository and filling in the project-specific values that the shell scaffold command intentionally leaves generic.

## Your Goal

Read the codebase, configuration, and docs first. Infer as much project context as you can from evidence. Update the scaffold state and scaffold-managed files with those values. Ask the user only when a high-impact ambiguity cannot be resolved from the repository itself.

## Step 1 — Load Scaffold State

Read `.agent-scaffold.json`.

If it contains unresolved `adoptionConflicts`, stop and tell the user to run `/resolve-adopted-artifacts` before hydration or refresh work continues. Do not refresh scaffold-managed files while preserved pre-scaffold artifacts are still awaiting resolution.

Treat its `variables` section as the canonical writable source for scaffold values such as:
- project name and description
- tech stack and language
- build, typecheck, lint, browser verification, test, and run commands
- source directory
- architecture pattern
- Beads prefix

## Step 2 — Inspect The Existing Repository

Read the strongest evidence first:
- root build and package files
- entrypoints and source directories
- README and other project docs
- test layout
- existing agent artifacts when present

Use the `grill-me` method, but point it at the repository before the user:
- resolve scope from the actual codebase
- resolve interfaces from config, entrypoints, and tests
- resolve dependencies from package/build files
- resolve constraints from docs and existing conventions
- when debugging, resolve ownership early: local code, generated code, vendor code, or CI/runtime environment
- when debugging CI, compare the last passing and first failing runs before proposing remediation
- if a failure originates in dependency code, question validation scope before recommending version pinning or vendor patching

Only escalate to the user when the repository leaves a genuinely consequential ambiguity unresolved.

## Step 3 — Derive The Scaffold Values

Update `.agent-scaffold.json` with the best evidence-backed values you can determine.

Prefer:
- explicit config over naming conventions
- observed commands over guessed commands
- repo terminology over generic labels

If a value cannot be justified from the repo, leave it as-is and record the ambiguity clearly before asking the user.

## Step 4 — Refresh Scaffold-Managed Files

After updating `.agent-scaffold.json`, deterministically refresh the scaffold-managed files from the template so they stop showing generic placeholders.

Prefer the scaffold renderer over hand-editing generated files:

```bash
../agent-bootstrap/scripts/scaffold.sh
```

or the equivalent scaffold command path for the local checkout that originally generated the scaffold.

Do not manually propagate placeholder replacements across generated files when the scaffold renderer is available. The source of truth is:
1. `.agent-scaffold.json` for project values
2. the template files under `bootstrap-templates/templates/universal/`
3. the scaffold renderer that combines them

This includes project-facing docs and config such as:
- `AGENTS.md`
- `.claude/CLAUDE.md`
- `.beads/config.yaml`
- any scaffolded skill or workflow file that uses scaffold variables

Use the scaffold renderer as the forward refresh path:

```bash
../agent-bootstrap/scripts/scaffold.sh
```

Treat the refresh as two layers of source of truth:
1. `.agent-scaffold.json` for project values
2. the scaffold templates for scaffold-managed file shapes

Do not treat project-local working artifacts as scaffold-managed:
- `.agents/plans/`
- `.agents/context/`
- `.agents/architecture/`
- `.beads/issues.jsonl`

Also do not treat scaffold presence as proof that operational tooling is initialized. For example, `.beads/config.yaml`, `.beads/clone-contract.json`, and `.githooks/pre-commit` can exist before the local Beads database has actually been bootstrapped.

## Step 5 — Produce Or Refresh Project-Local Working Artifacts

When the evidence supports it, create or refresh:
- `.agents/context/ubiquitous-language.md`
- `.agents/architecture/module-map.md`

These are project-local working artifacts. They are informed by the scaffold, but they are not the scaffold itself.
If legacy `.claude/context/` or `.claude/architecture/` artifacts already exist and no `.agents/` equivalent exists, preserve their content by migrating or copying it into `.agents/` before updating it.

## Step 6 — Verify Operational Readiness

Before declaring bootstrap complete, verify whether scaffold-adjacent tooling is actually operational.

At minimum:
- if the scaffold includes Beads config, check `bd status --json`
- if that check fails, say explicitly that Beads is configured but not yet bootstrapped
- prefer the non-destructive recovery/setup path:

```bash
bd bootstrap --yes --json
```

- after any Beads bootstrap step, re-run:

```bash
bd status --json
```

Do not silently assume that generated config files imply a usable local database.

## Step 7 — Report Remaining Ambiguity

If anything important is still unresolved, ask narrow questions with concrete evidence:
- what you found
- why it is ambiguous
- what decision needs confirmation

Do not interrogate the user for information the repository already contains.
