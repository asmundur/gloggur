# Retrospective Skill

You are running a retrospective for **gloggur**. This skill analyzes a completed feature, identifies what worked and what didn't, and propagates generalizable improvements back into the scaffold template.

The point of `/retro` is not to produce a recap. The point is to improve future work by turning generalizable lessons into concrete edits, especially edits to the upstream scaffold source when the lesson applies beyond the current project.

## Non-Negotiables

- Do not create a new Beads issue just to run a retro or document agent housekeeping. If the completed body of work already has a task, use that context.
- When the retro is about agent/process behavior and the user has not requested tracked implementation work, do not create, claim, close, or modify Beads issues. Produce the retro and stop at the template-propagation approval checkpoint.
- If the retro will make new tracked-file changes after approval, use an existing relevant task only when the user wants that work tracked. Otherwise, make the approved template changes without introducing task-log noise.
- For scaffold, workflow, or agent-behavior retros, lead with the failure mode and the corrective change. Keep evidence brief and only as detailed as needed to justify the change.
- When a finding is universal, template propagation targets the upstream `agent-bootstrap` repository's `bootstrap-templates/templates/universal/` tree. In a scaffolded project, look for that upstream source through `.agent-scaffold.json.templateSource`, a sibling `../agent-bootstrap`, or the path the user provides. Do not stop after editing generated local `.agents/`, `.claude/`, `.codex/`, or `.antigravity/` copies.
- Generated scaffold files may be refreshed after the upstream template changes, but generated files are not the source of truth.

## Phase 1 — Load Context

```bash
git log --oneline -20
git branch --show-current
```

Read these files when they exist before you ask retrospective questions:
- `.agents/context/ubiquitous-language.md`
- `.agents/architecture/module-map.md`
- `.agents/plans/` for the relevant feature spec

If the relevant artifacts still exist only under legacy `.claude/...` paths, read them as migration evidence and target `.agents/...` for new shared updates.

Default scope:
- if the user names a feature, branch, issue, commit range, or session, use that
- otherwise, retrospect on the work completed in the current chat session on the current branch

Ask the user to clarify scope only when multiple plausible scopes exist and the wrong one would materially change the findings.

## Phase 2 — Gather Evidence

```bash
git log <branch> --oneline
git diff main...<branch>
git status --short
git diff --stat
git diff --cached --stat
git ls-files --others --exclude-standard
```

Read the relevant changed files. Build a picture of what was implemented, how long it took (from commit timestamps), what the diff looks like, and whether the shipped work matched the approved feature spec.

For an uncommitted current-session retro, `git diff main...<branch>` may be empty even when the work is real. In that case, use the working tree, staged diff, untracked scaffold files, command outputs, and chat context as the primary evidence.

For debugging or CI retros, explicitly reconstruct:
- the last passing state
- the first failing state
- the ownership boundary of the failing surface: repo code, generated code, vendored code, or environment

## Phase 3 — Present Overview & Get User Input

Show the user:
- Files changed, lines added/removed
- Number of commits
- Commit messages (do they tell a coherent story?)
- Whether the feature spec, glossary, and module map were used or drifted

Then separate two kinds of retrospective input:

1. **Agent-answerable questions** — answer these yourself from the evidence before asking the user anything:
   - Was anything unclear or ambiguous at the start?
   - Did terminology drift between the plan, code, and user-facing language?
   - Did module boundaries help or get in the way?
   - Which feedback loops caught issues early, and which were missing or too slow?

2. **User-experience questions** — ask the user only for what the user uniquely knows:
   - What went well from your perspective?
   - What felt frustrating, slow, or surprising?
   - What would you want preserved or changed next time?

Do not interview the user for observations that are primarily about the agent's own workflow, reasoning, or execution trail when those can be derived from the repository, diffs, commands, and chat context.

## Phase 4 — Identify Successes

For each success, document:
- What was it?
- Why did it work? (the mechanism, not just "it was good")
- Is it replicable? (could a different agent on a different project do it the same way?)

## Phase 5 — 5W Root Cause Analysis on Failures

For each failure or friction point, apply:

- **What** happened?
- **Why** did it happen? (first cause)
- **Why** did that happen? (second cause)
- **Why** did that happen? (root cause — keep asking until you hit a process or assumption)
- **Where** in the workflow did it occur? (planning, implementation, review, commit)
- **Who** is responsible for fixing it? (agent behavior, template content, workflow definition, anti-patterns file)

Also classify whether the failure came from:
- Missing shared design
- Term drift / poor ubiquitous language
- Weak module boundaries
- Outrun feedback loops
- Something else

## Phase 6 — Generalizability Test

For each finding, ask: *"Would this improvement help ANY project using this bootstrap, or only gloggur?"*

- **Universal** → candidate for template propagation
- **Project-specific** → add to local `.agents/anti-patterns.md` only

Exclude findings that are:
- Generic best practices already known
- One-off bugs with no systemic cause
- Highly domain-specific to gloggur

Prefer findings that improve the bootstrap itself:
- debugging order of operations
- dependency-boundary heuristics
- process proportionality
- retro output quality

## Phase 7 — Action Items

Produce a prioritized list:

| Priority | Finding | Action | Target File |
|---|---|---|---|
| High | ... | Add anti-pattern entry | `.agents/anti-patterns.md` |
| High | ... | Update glossary | `.agents/context/ubiquitous-language.md` |
| High | ... | Update module map | `.agents/architecture/module-map.md` |
| Medium | ... | Update workflow checkpoint | `.agents/workflows/feature-workflow.md` |
| Low | ... | Clarify provider-specific agent instructions | `.claude/agents/feature-implementation.md` or equivalent harness file |

If terminology or module boundaries changed, propose the concrete updates to the local glossary or module map as part of the retrospective output. These local artifact updates are project maintenance, not template propagation.

## Phase 8 — Template Propagation (MANDATORY CHECKPOINT)

For each universal finding, show the user the proposed change to the bootstrap template in `bootstrap-templates/templates/universal/`.

**Stop and get user approval before writing any template changes.**

Say: "I'd like to propagate these [N] findings to the bootstrap template. Here are the proposed changes — approve to write them?"

After approval, make the changes in the upstream `agent-bootstrap` template source, not only in the current project's generated scaffold files. Then refresh generated copies where appropriate so the current repository also reflects the updated template.

Do not stop at a generic recap when the chat itself exposes a reusable failure mode. Turn that failure mode into a concrete template or workflow improvement whenever the generalizability test passes.
