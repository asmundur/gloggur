# State Diagrams

This directory is the canonical state-map for Glöggur application health and
the CLI surfaces that expose stable lifecycle/state contracts. Diagrams use
standard fenced `mermaid` blocks with `stateDiagram-v2` so they render in
GitHub and Markdown environments that support Mermaid.

Legend:
- Exact emitted contract values are used as primary state labels whenever the
  CLI already exposes them in JSON payloads or tests.
- Internal steps without a public enum use implementation names such as
  `bootstrap_model` and `commit_metadata`.
- Transition-driving failure codes are called out inline or in notes; the full
  catalog stays in [docs/ERROR_CODES.md](../ERROR_CODES.md).

## Diagram Index

| File | Scope |
| --- | --- |
| [application-state.md](application-state.md) | Shared health from `status --json`, resume decisions, build-state normalization, and semantic-search readiness. |
| [index-command-state.md](index-command-state.md) | Indexed build lifecycle from `bootstrap_model` through `commit_metadata`. |
| [watch-command-state.md](watch-command-state.md) | Shared runtime watch state manipulated by `watch init/start/status/stop`. |
| [search-command-state.md](search-command-state.md) | Search preflight, cache-ready gate, routed retrieval, and success/error emission. |
| [find-command-state.md](find-command-state.md) | `find_v1` decision states and the ambiguous-path narrowing flow. |
| [artifact-publish-state.md](artifact-publish-state.md) | Artifact packaging and transport-specific publish flow. |
| [artifact-validate-state.md](artifact-validate-state.md) | Validation flow, `valid`, and provenance warning propagation. |
| [artifact-restore-state.md](artifact-restore-state.md) | Validate-first restore flow through staging and destination activation. |
| [simple-command-reference.md](simple-command-reference.md) | Commands and groups that intentionally do not get separate lifecycle diagrams. |

## Notes

- `status` itself is documented through
  [application-state.md](application-state.md); the command wrapper is simple,
  but the payload state is not.
- `artifact validate` and `artifact restore` can emit `warning_codes` while
  still succeeding; those warnings are part of the documented state flow.
- The diagram set is intentionally narrower than the full command tree. Wrapper
  commands remain documented in
  [simple-command-reference.md](simple-command-reference.md).
