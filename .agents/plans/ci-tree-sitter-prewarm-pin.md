# CI Tree-Sitter Prewarm Pin

## Goal

Stabilize the required Python 3.12 CI lane by removing unbounded `tree-sitter-language-pack` drift from the dependency contract.

## Scope

- Add focused regression coverage for the dependency declaration.
- Narrow the dependency declaration in `pyproject.toml`.
- Verify with the fastest relevant local feedback loop first, then the configured quality gates.

## Non-Goals

- Redesign the verification workflow.
- Change parser bootstrap behavior beyond dependency stability.
- Refresh unrelated dependencies.

## Acceptance

- `pyproject.toml` does not leave `tree-sitter-language-pack` unbounded.
- A focused unit test locks the dependency contract.
- Configured verification commands pass locally.
