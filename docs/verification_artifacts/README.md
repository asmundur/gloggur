# Verification Artifacts

This directory holds durable, repo-tracked verification artifacts that are needed
to move Beads tasks from blocked to done-ready.

Expected artifact patterns:
- `provider_probe_openai_<YYYY-MM-DD>.md`
- `provider_probe_gemini_<YYYY-MM-DD>.md`
- `packaging_upgrade_from_<version>_<YYYY-MM-DD>.md`

Current state:
- No live-provider or published-release upgrade artifacts are checked in yet.
- `bd-uhf` and `bd-n1f` stay blocked until the corresponding external evidence is
  captured here and referenced from the Beads task descriptions.
