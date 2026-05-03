@AGENTS.md

## Agent Registry

| Agent | Model | Responsibility |
|---|---|---|
| Feature-Implementation | claude-sonnet-4-6 | Implements code, writes tests, cleans up |
| Git-Manager | claude-haiku-4-5-20251001 | Stages files, creates semantic commits |

## Workflow Orchestration

| User Intent | Skill / Workflow |
|---|---|
| Hydrate scaffold values from the codebase | `/bootstrap` → `.claude/skills/bootstrap.md` |
| Resolve preserved pre-scaffold artifacts | `/resolve-adopted-artifacts` → `.claude/skills/resolve-adopted-artifacts.md` |
| Reach shared design before planning | `/grill-me` → `.claude/skills/grill-me.md` |
| Build or refresh the project glossary | `/ubiquitous-language` → `.claude/skills/ubiquitous-language.md` |
| Propose better module boundaries | `/improve-architecture` → `.claude/skills/improve-architecture.md` |
| Work in strict red/green/refactor loops | `/tdd` → `.claude/skills/tdd.md` |
| Start a new feature | `/feature-start` → `.agents/workflows/feature-workflow.md` |
| Review what worked after a feature | `/retro` → `.claude/skills/retro.md` |
| Audit or improve Beads task quality | `/audit-beads-quality` → `.claude/skills/audit-beads-quality.md` |
| Task tracking | See `AGENTS.md` for mandatory gates and high-fidelity task creation commands |

## Durable Artifacts

- `.agents/plans/<feature-slug>.md` — approved feature specs and implementation contracts
- `.agents/context/ubiquitous-language.md` — canonical domain terms and aliases to avoid
- `.agents/architecture/module-map.md` — target modules, interfaces, dependencies, and interface-level tests

Load the glossary and module map before planning or implementation when they exist.

## Mandatory Gates

Claude **must stop and get user approval** before:
1. Starting tracked implementation work without a claimed Beads task (Mandatory Gate). Read-only discussion and slash-command skills that explicitly opt out of Beads do not need task churn.
2. Transitioning from Stage 1 (planning) to Stage 2 (implementation)
3. Transitioning from Stage 2 (implementation) to Stage 3 (commit)
4. Propagating findings from retro to the bootstrap template
5. Any destructive git operation

## Anti-Patterns

See `.agents/anti-patterns.md` for hard constraints that agents must follow. Agents should read this file at the start of any implementation task.
