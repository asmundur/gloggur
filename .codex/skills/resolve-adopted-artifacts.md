# Resolve-Adopted-Artifacts Skill

You are running the `resolve-adopted-artifacts` skill for **gloggur**. Your job is to resolve all unresolved scaffold adoption conflicts in one pass.

This skill exists for the case where `scripts/scaffold.sh` found pre-existing artifacts during first-time scaffold adoption, left those artifacts active at their original paths, and wrote generated scaffold replacements as `*.scaffold-candidate.*` sidecars.

## Goal

Review every active legacy artifact and its scaffold candidate, extract what the upstream scaffold should learn, grill the user on what should be adopted into `agent-bootstrap`, then apply the approved upstream source changes before doing local archival or installation work.

The first deliverable is the extraction and adoption interview, not the file move. You must show what the scaffold would break, lose, or fail to model before you propose replacing, archiving, or deleting anything.

Do not resolve conflicts one-by-one. This is an all-at-once pass.

This skill is part of the scaffold learning loop. Do not treat it as a local cleanup checklist. If the legacy artifacts reveal a reusable workflow failure, missing scaffold instruction, missing extension point, incorrect default, or agent failure mode, capture that as an upstream adoption finding and route it to the upstream `agent-bootstrap` source.

Upstream source means the real scaffold implementation, not generated target-project copies:
- `bootstrap-templates/templates/universal/` for generated instruction, hook, config, and skill content
- `scripts/scaffold.sh` for generation behavior, defaults, state shape, conflict detection, and candidate placement
- `scripts/smoke-test-scaffold.sh` or equivalent tests for regression coverage
- upstream scaffold docs only when the behavior also needs operator-facing documentation

If a scaffold candidate would break or change existing behavior in a way the scaffold could have preserved, treat that as an upstream defect candidate first. Local preservation is only the answer after you have ruled out a reusable template, variable, extension point, or scaffold-script fix.

Do not create target-project Beads issues just to run adoption cleanup or document agent housekeeping. Use the existing body-of-work task when one exists. If the adoption pass discovers real follow-up product work, file that separately and narrowly.

## Step 1 — Load The Conflict Set

Read `.agent-scaffold.json`.

If `adoptionConflicts` is missing or empty, stop and tell the user there is nothing to resolve.

For each conflict, load:
- the active legacy `target`
- the sibling `scaffoldCandidate`
- the recorded checksums and capture timestamp

Before doing anything else:
- verify every active `target` still exists
- verify every `scaffoldCandidate` still exists
- verify their current checksums still match `targetChecksum` and `candidateChecksum`

If any active target or scaffold candidate is missing or drifted, stop and report it clearly. Do not guess at resolution from modified artifacts.

Backward compatibility: if a conflict row uses the older `preservedBackup` shape, treat the backup as the legacy artifact and the current `target` as the scaffolded replacement. Report that you are resolving an older adoption-conflict shape before continuing.

## Step 2 — Read Before Asking

For each conflict, read:
- the active legacy file currently at `target`
- the scaffold replacement at `scaffoldCandidate`
- relevant local context such as `.agents/context/ubiquitous-language.md`, `.agents/architecture/module-map.md`, and the relevant feature spec when useful

Use the `grill-me` method on the artifacts first:
- what value did the legacy file provide?
- what value is already covered by the scaffolded replacement?
- what is genuinely new or missing?
- is that missing value universal enough to help any project using this scaffold?
- would the scaffold candidate break, weaken, or silently change an existing contract?
- is the right fix an upstream template change, `scripts/scaffold.sh` behavior change, scaffold variable, hook/command extension point, or target-project local constraint?

Answer as much as you can from the files before asking the user anything.

Before classifying or proposing any file operation, build an artifact extraction ledger. This ledger is working material, not the user-facing format. Distill it into concise findings and questions later instead of pasting a giant table into chat. For every meaningful legacy rule or behavior, capture:
- the source artifact and heading or nearby text
- the concrete command, path, restriction, safety check, recovery path, or ownership boundary
- the failure it prevents or workflow value it preserves
- whether the scaffold candidate already covers it, partially covers it, conflicts with it, or misses it
- the exact behavior that would be lost or changed if the candidate were installed as-is
- the reusable workflow rule, separated from any repo-specific parameter
- the upstream source surface most likely to need work: template, `scripts/scaffold.sh`, smoke test, docs, or none
- the proposed disposition: upstream source change, scaffold variable/extension point, preserve locally, or archive only
- a concrete patch sketch or test assertion for any upstream source change

Do not start from the diff as a replacement checklist. Start from the legacy artifact as evidence: ask what the pre-existing file taught agents to do, why that mattered, and whether the candidate would preserve that behavior.

Do not treat this as a file cleanup step. The legacy artifacts and scaffold candidates are evidence. If a legacy artifact appears local, stale, or redundant, still ask what the adoption process itself revealed about the scaffold workflow before deciding it has no reusable value.

Do not collapse "candidate changes functionality" into a local preference. Examples that require upstream analysis before local cleanup include:
- replacing a real Beads prefix with the scaffold default
- replacing a project-specific recovery/bootstrap command with a generic one
- dropping hook commands such as lint, secret scanning, validation, or generated-file checks
- dropping no-auto-commit, no-auto-push, no-force-push, no-hook-bypass, or no-generated-instruction-restoration guardrails
- dropping ownership rules for git hook paths, tracked generated files, or source-of-truth metadata
- dropping stale-runtime recovery instructions
- adding session-close git operations that conflict with a repository's safety model
- losing requirements that prevent alternate TODO trackers or durable side ledgers

For each such loss, ask whether the scaffold needs a universal rule, a variable, an extension point, a smarter adoption merge, or a target-project local section. Only call it "local only" after that analysis.

For policy artifacts such as `AGENTS.md`, `CLAUDE.md`, harness instruction files, hook scripts, and setup docs, do a line-item policy inventory instead of summarizing the file as one blob. Extract every imperative rule, safety constraint, command contract, recovery path, ownership boundary, and "do not" instruction. For each extracted rule, decide whether it is:
- a scaffold-standard rule that belongs in the upstream template or upstream skill
- a scaffold variable or extension point with a repo-specific value or command list
- a scaffold-script behavior change, especially in `scripts/scaffold.sh`, needed to avoid breaking or changing adopter behavior
- a genuinely project-local safety constraint that must remain active after adoption
- obsolete/noise that only needs archival

Do not compress a policy artifact into an umbrella phrase such as "stricter repo rules", "local safety constraints", "repo-specific guardrails", or "custom workflow notes" before the user has seen the decomposed rules. Those labels are only acceptable after you have already extracted the individual rules and asked whether each rule's essence belongs upstream, belongs behind a scaffold variable or extension point, stays project-local, or should be archived.

For every bundled policy loss, split the bundle into concrete rule rows before the interview. A sentence like "AGENTS.md would lose no auto-commit, no push, hook path ownership, stale-runtime recovery, and no TODO trackers" is not sufficient. The interview must cut into each rule family with source evidence and a classification question, for example:
- `no git add/commit/push`: is this a universal no-session-close-git-action rule, a project-local restriction, or obsolete?
- `no bd dolt push`: is this a universal tracker-push prohibition, a Beads-specific variable/extension point, or project-local?
- `do not run bd setup codex`: is this a universal generated-instruction ownership rule or local to repositories with repo-owned agent files?
- `.githooks/ is the only supported hook path`: is this a scaffold-standard hook ownership rule, a variable, or local?
- stale Beads runtime cleanup commands: are these a reusable recovery contract, a Beads clone-contract variable, or local runbook content?
- no markdown TODO trackers: is this a universal "use the project tracker only" rule or local preference?

Do not equate repo-specific wording with repo-specific essence. Many legacy instructions are "repo-specific" only because they were written before this scaffold standardized the workflow. Your job is to separate the generalizable rule from the local value. For example, a concrete issue prefix, hook path, validation command, or bootstrap recovery command may point to a missing variable, extension point, or standard guardrail rather than content that should be archived as local trivia.

`AGENTS.md` is usually the highest-signal conflict. Process it before lower-level config and hook files when it appears in `adoptionConflicts`. For `AGENTS.md`, explicitly extract these categories when present:
- task-tracking entry and exit rules
- issue creation quality bar and required fields
- prohibited agent actions such as auto-commit, auto-push, force-push, generated-instruction restoration, or hook bypass
- hook path, generated-file, and source-of-truth ownership boundaries
- clone/bootstrap/recovery commands and stale-runtime cleanup paths
- requirements about not creating alternate TODO trackers or durable work ledgers
- repo-specific validation, lint, secret scanning, or pre-commit commands
- candidate-introduced rules that would newly require git staging, committing, pushing, force-pushing, hook bypass, or tracker push behavior
- generated instruction regeneration commands that would overwrite repository-owned agent policy

If the scaffold candidate would drop or contradict any extracted `AGENTS.md` rule, say that directly and classify each loss before proposing an install action. Do not summarize this as "repo-specific safety constraints" until you have tested whether each rule is a reusable scaffold guardrail, a variable, or an extension point.

For `AGENTS.md`, the user-facing brief may group related extracted rules, but each group must still contain concrete sub-questions. If a group contains multiple rules, ask a continuation question that forces the adoption decision for the included details, not a single yes/no question for the whole group.

## Step 3 — Classify Findings

For each meaningful finding, classify it as exactly one final disposition. Use "Needs user input" only as a temporary interview label; do not use it as the final disposition.

1. **Upstream source change**
   - useful to any project using agent-bootstrap
   - belongs in `bootstrap-templates/templates/universal/`, `scripts/scaffold.sh`, scaffold smoke tests, or upstream scaffold docs
   - includes reusable agent-workflow failures discovered while resolving the adoption, not only content copied out of legacy files
   - must include the intended upstream file or behavior surface and a focused regression assertion

2. **Scaffold variable or extension point**
   - the behavior is reusable, but the value or command is target-project specific
   - examples include issue prefixes, bootstrap command variants, hook command chains, protected generated paths, quality-gate commands, or local policy sections
   - should result in an upstream variable, state field, convention, or local extension slot rather than hard-coded local copying

3. **Project-local preserved constraint**
   - still active and important for this repository
   - too tied to this project's tools, hosting, compliance, security model, or operational contract to put directly into the universal scaffold
   - should be merged into the accepted scaffold target under a local safety-constraints section, not lost into archival only
   - should be rewritten narrowly with concrete evidence, exact commands/paths, and no generic workflow lessons that should have been promoted upstream instead

4. **Obsolete / noise**
   - project-local, stale, redundant, or not worth preserving in active scaffold docs
   - archive it locally only

Do not propagate purely project-local legacy guidance into upstream templates. Preserve active local guidance only when it is a real safety constraint, and keep it in the accepted scaffold target rather than in the universal source.

Do not classify a broken scaffold candidate as project-local just because the legacy value is concrete. A concrete value is often evidence that the scaffold needs a variable, migration rule, or extension point.

Do not conclude "no upstream findings" just because the legacy file contents are phrased around this repository. First grill the boundary: what is the reusable workflow rule, what is the local parameter, and what is truly local? Also check whether the adoption process exposed a general scaffold failure, such as creating durable task noise for cleanup, confusing generated files with source templates, or failing to propagate a reusable lesson upstream.

For a policy artifact with any imperative content, a conclusion of "archive only" is valid only after the extraction ledger shows that each rule is either already fully covered by the scaffold or is genuinely obsolete. If you cannot prove that, classify the row as "Needs user input" rather than burying it in an archive plan.

Adoption-process findings are first-class. A reusable finding may come from how the conflict happened or how the agent almost resolved it, not only from text copied out of the legacy file. Check specifically for:
- premature cleanup before the user sees findings
- missing extension points for project-local behavior
- unclear ownership between generated scaffold files and upstream templates
- lost repo-specific safety checks or workflow hooks
- agent confusion caused by scaffold-managed files versus local source files

## Step 4 — Run The Extraction Interview

The user-facing checkpoint must be a grill-me style adoption interview, not a bloated report and not a local file-operation checklist. Its purpose is to decide what the upstream scaffold should learn.

Before asking questions, give a short decision brief:
- one sentence stating how many conflicts were inspected and whether checksums matched
- 3-7 bullets for the highest-risk losses, behavior changes, or reusable lessons found, with policy-artifact bullets naming the individual rule families instead of hiding them behind umbrella labels
- a compact "my default if you agree" recommendation that separates upstream changes from local preservation and archive-only work

Then ask only the questions that need human judgment. Do not ask the user to inspect your whole ledger. Do not present a wide matrix covering every conflict. Do not require the user to parse artifact-by-artifact tables before they can answer. Do not lead with "should I archive/install these files?" while unresolved upstream adoption decisions remain.

For policy artifacts, the interview must be a continuation of the extraction, not a summary endpoint. After identifying a bundled loss, ask what to incorporate into upstream templates, variables, extension points, or local constraints for each distinct rule family. Never stop at a handy summary of the rules that would be lost.

Err on the side of a more complete questionnaire when multiple independent decisions are hiding under one artifact. If a compact question groups several rule families, the numbered prompt must still name each unresolved family and force a disposition for each one; otherwise split it into additional short questions or a second follow-up checkpoint. A user should never have to infer that approving one broad question also approved unrelated rules inside the same bundle.

Good adoption-resolution questions are:
- disposition-shaping: "Should this rule become universal scaffold behavior, stay local, or be archived?"
- variable-shaping: "Is this a repo-specific value, or should the scaffold expose an extension point for it?"
- safety-shaping: "Is this hook/check/restriction still required here after scaffold adoption?"
- ownership-shaping: "Is this generated target allowed to change, or is another upstream template the source of truth?"
- scope-shaping: "Should I apply this upstream template change now, or record it as follow-up?"
- implementation-shaping: "Should this be a template edit, a `scripts/scaffold.sh` behavior fix, a scaffold variable, or a hook/command extension point?"

Ask questions in small groups. Prefer 2-5 numbered questions in a checkpoint. If there are more unresolved decisions, ask the next group after the user answers. Use terse evidence under each question only when it changes the decision.

Use this shape by default:

```markdown
I inspected <n> conflicts. Checksums still match.

What I think matters:
- <finding with source path and impact>
- <finding with source path and impact>
- <finding with source path and impact>

My default resolution:
- <upstream source changes>
- <scaffold variables or extension points>
- <local constraints to preserve>
- <archive-only items>

Questions:
1. <focused question with 1-2 concrete options>
2. <focused question with 1-2 concrete options>
3. <focused question with 1-2 concrete options>
```

For upstream findings, present proposed source changes in the same style as `/retro`: the observed failure, the general rule, the upstream target file or behavior, and the regression check. Include `scripts/scaffold.sh` when the issue is generation behavior rather than generated text.

For `AGENTS.md` and similar policy artifacts, the internal inventory must still cover individual rules when the file contains multiple independent rules. Do not collapse "repo-specific safety constraints" into one blob internally. In the user-facing interview, group related rules into human-sized question prompts and grill the user on any classification that is unclear:
- "Should this become a universal scaffold rule?"
- "Is this a project variable or hook/command extension point?"
- "Is this still required only for this repository?"
- "Is this obsolete now that the scaffold has standardized the workflow?"
- "Would installing the scaffold candidate as-is break or weaken behavior that `agent-bootstrap` should preserve?"

Bad interview shape:

```markdown
- AGENTS.md would lose stricter repo rules: no auto commit/push, hook path ownership, stale runtime recovery, and no TODO trackers.

Question: Should I preserve these locally?
```

Good interview shape:

```markdown
- AGENTS.md would lose six separate policy rules that may belong in different places: no session-close git actions, no tracker push, generated instruction ownership, hook path ownership, stale Beads runtime recovery, and tracker-only work ledgers.

Questions:
1. Should "no session-close git actions" become a universal scaffold rule, or stay local to this repository?
2. Should tracker push prohibitions and generated-instruction ownership become Beads-specific scaffold guidance?
3. Should stale runtime recovery be modeled as a clone-contract/runbook extension point, or kept as local operational guidance?
4. Should "no markdown TODO trackers" become the universal tracker-only rule?
```

This interview checkpoint is mandatory even when you believe there are no upstream findings. If every finding looks obsolete/noise, say why in a few bullets and ask the user to confirm that archive-only conclusion. If a policy artifact has safety constraints and none are classified as upstream source change, scaffold variable/extension point, or project-local preserve, treat that as suspicious and re-run the line-item inventory before asking.

Keep the checkpoint concise enough to answer. Avoid tables unless there are at most 3 rows and the table is clearly easier to read than bullets. Never create a single wide table that mixes source evidence, classification, and filesystem operations for many unrelated artifacts. Separate extraction questions from resolution actions.

**Stop and get user approval before writing any upstream source changes, archiving legacy artifacts, installing scaffold candidates, or removing `adoptionConflicts`.**

When approved, make upstream changes in the `agent-bootstrap` repository when that source is available. Generated local `.agents/`, `.claude/`, `.codex/`, and `.antigravity/` files are not the source of truth. Refresh generated copies only when explicitly requested or when the accepted scope requires it, and report every generated file changed.

## Step 5 — Apply Upstream Changes, Then Archive And Install

Once the extraction interview is approved:
- apply upstream source changes first when approval included them
- update or add focused scaffold smoke coverage for any upstream behavior change
- run the relevant upstream verification before local cleanup when practical
- only then proceed with target-project archival and accepted scaffold installation

For the target project:
- create `docs/legacy-agent-artifacts/` when needed
- move each active legacy `target` into that tree using mirrored directory structure relative to the project root
- move each `scaffoldCandidate` into its original `target` path, unless the approved resolution explicitly merged upstream-adopted or project-local-preserve value into the target first
- remove any consumed scaffold candidate sidecars

When a project-local preserved constraint applies to `AGENTS.md`, merge it into the scaffold target under `## Project-Specific Safety Constraints`. Keep the wording narrow and concrete. Do not use this local section for broad workflow lessons, because those belong upstream.

Examples:
- `AGENTS.md` -> `docs/legacy-agent-artifacts/AGENTS.md`
- `.claude/CLAUDE.md` -> `docs/legacy-agent-artifacts/.claude/CLAUDE.md`

Do not flatten the tree. Preserve original relative paths.

## Step 6 — Clear Temporary Conflict State

After all legacy artifacts are archived and any approved upstream changes are made:
- remove `adoptionConflicts` from `.agent-scaffold.json`
- verify each resolved `target` now matches the checksum recorded for the scaffold-managed file entry, or update that file entry checksum when the approved resolution intentionally changed the installed target content

The scaffold state should not retain resolved conflict history.

## Step 7 — Report Completion

Report:
- how many adoption conflicts were resolved
- which legacy artifacts were archived
- which findings became upstream source changes, scaffold variables/extension points, project-local preserves, or obsolete/noise
- whether any upstream changes were proposed or applied

Once complete, the repo is eligible to run `scripts/scaffold.sh` again.
