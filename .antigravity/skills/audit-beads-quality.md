---
name: audit-beads-quality
description: Audits and enforces high-quality Beads task tracking by improving useful sparse issues, creating missing durable work items, and deleting confirmed noise.
---

# Audit Beads Quality Skill

You are executing the `audit-beads-quality` skill. Your objective is to make the repository's Beads task history useful for coding agents: concrete enough to execute, honest about scope, and free of obvious task-tracking noise.

This is not a commit-history fabrication workflow. Do not create one task per commit by default. Use commit history only as evidence when deciding whether a missing or sparse Beads item represents a real durable body of work.

## Non-Negotiables

- Do not adopt or preserve mandatory automatic git commit, git push, bd sync, or bd dolt push workflows.
- Do not edit `.beads/issues.jsonl` by hand. Use `bd` commands and export/import paths.
- Run `bd` write commands serially when the repository uses an embedded Dolt-backed Beads store.
- Delete noise by default only after presenting a dry-run deletion set. Use `bd delete --dry-run` before `bd delete --force`.
- Do not create Beads for discussion, read-only exploration, retros, command-output inspection, bookkeeping about tracking, or "decide whether to track this" work.
- Preserve migration or historical markdown backlog files only as evidence; net-new work belongs in Beads.

## Step 1: Verify Runtime State

Verify that the local Beads store is operational:

```bash
bd status --json
```

If that fails because Beads has not been bootstrapped yet, run the non-destructive setup path first:

```bash
bd bootstrap --yes --json
bd status --json
```

Do not assume that `.beads/config.yaml`, `.beads/clone-contract.json`, hooks, or `.beads/issues.jsonl` prove the runtime database is readable.

## Step 2: Gather Evidence

Inspect the current tracker and repository context before mutating anything:

```bash
bd list --json
bd ready --json
```

Then sample issue details with `bd show <id> --json` when needed. Also inspect:

- `.beads/issues.jsonl` for exported state and sparse records
- `AGENTS.md` and local skill/workflow instructions for task-tracking policy
- legacy `TODOs.md`, `DONEs.md`, roadmap files, or issue ledgers when they exist
- `git log --oneline --decorate` and focused `git show <commit>` output only when history is needed to understand provenance

## Step 3: Classify the Tracker

Create an audit ledger in your working notes and classify each issue as keep, improve, create, delete-noise, or needs-user-decision.

Use these classifications:

- `keep`: already has concrete scope, useful context, acceptance criteria, and no obvious duplication.
- `improve`: useful work item, but title, description, design, acceptance, notes, estimate, type, priority, or dependencies are too sparse or misleading.
- `create`: real durable work exists in repo evidence but has no Beads item.
- `delete-noise`: placeholder, test task, wrong-project item, pure bookkeeping, duplicate with no independent value, or discussion-only task.
- `needs-user-decision`: deletion or consolidation would be destructive and the evidence is not decisive.

Prefer improving over deleting when an issue records real product, infrastructure, test, documentation, or design work. Prefer deletion when the task would actively mislead future agents.

## Step 4: Enforce the Quality Bar

A high-quality Beads issue has:

- a tight title that names the outcome, not the implementation tactic alone
- a description with current behavior, why it matters, and explicit scope boundaries
- design notes naming likely files, interfaces, contracts, constraints, and non-goals
- observable acceptance criteria, ideally including the fail-loud or verification contract
- notes with evidence: user request, probes, failing commands, relevant commits, or test nodes
- an estimate in minutes
- correct type, priority, assignee/status when active, and dependencies where relevant

For useful sparse issues, update the existing issue instead of creating a replacement:

```bash
bd update <id> \
  --title "<tight outcome title>" \
  --description "<current behavior, why it matters, scope>" \
  --design "<touchpoints, constraints, non-goals>" \
  --acceptance "- Observable outcome\n- Verification contract" \
  --notes "<evidence and provenance>" \
  --estimate 90 \
  --json
```

For real missing work, create one issue per coherent body of work:

```bash
bd create "Tight, concrete issue title" \
  --type bug|feature|task \
  --priority 1 \
  --description "Current behavior, why it matters, scope boundaries, and exact paths/contracts involved." \
  --design "Implementation touchpoints, interfaces, constraints, and non-goals." \
  --acceptance "- Observable outcome 1\n- Observable outcome 2\n- Verification or fail-loud contract" \
  --notes "Evidence: user request, direct probes, failing commands, relevant commits, or focused tests." \
  --estimate 90 \
  --json
```

Create discovered follow-ups only for separate future work:

```bash
bd create "Concrete follow-up discovered while auditing <parent-id>" \
  --type task \
  --priority 1 \
  --description "Specific separate follow-up and why it should survive independently." \
  --design "Likely files, contracts, or interfaces to change." \
  --acceptance "- Shipped contract\n- Focused regression coverage or verification\n- Scope remains narrower than the parent" \
  --notes "Discovered during <parent-id>; include exact evidence." \
  --estimate 60 \
  --deps discovered-from:<parent-id> \
  --json
```

## Step 5: Delete Noise Deliberately

For obvious noise, first preview the exact deletion set:

```bash
bd delete <id> --dry-run
```

For multiple items:

```bash
bd delete --from-file deletions.txt --dry-run
```

Only after the dry run matches the intended set, delete:

```bash
bd delete <id> --force
```

Use `needs-user-decision` instead of deletion when an issue has dependents, ambiguous historical value, or product meaning that cannot be reconstructed confidently.

## Step 6: Verify and Report

After updates, verify the tracker remains readable:

```bash
bd status --json
bd ready --json
```

If `.beads/issues.jsonl` is tracked by this repository, refresh it through Beads export or the repository hook path; never edit it manually.

Report:

- counts by classification
- issue IDs improved, created, and deleted
- any `needs-user-decision` items with the decision needed
- verification commands run
- remaining risks or intentionally preserved historical backlog files
