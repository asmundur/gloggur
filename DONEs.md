# DONEs Workflow

`DONEs.md` is the immutable log of completed work.

## Usage Rules

1. Only move tasks here after acceptance criteria are actually met.
2. Keep the same task ID used in `TODOs.md`.
3. Include completion date (`YYYY-MM-DD`), evidence, and verification commands.
4. Record behavioral impact (what changed for users/agents).
5. If follow-up work is discovered during closure, create a new item in `TODOs.md` and link it.
6. Do not rewrite history; append corrections as follow-up entries.

## Completion Entry Template

Copy this section for completed tasks:

```md
## <ID> - <Title>

**Completed On**: YYYY-MM-DD
**Completed By**: <name or agent>
**Source**: moved from TODOs.md

**Delivered**
- <implemented change 1>
- <implemented change 2>

**Verification**
- Commands run:
  - `<command>`
  - `<command>`
- Results:
  - <pass/fail summary>

**Evidence**
- Files: `/abs/path/one`, `/abs/path/two`
- PR/commit/issues: <links or identifiers>

**Follow-ups**
- None
```
