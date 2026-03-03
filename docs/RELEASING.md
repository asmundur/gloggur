# Releasing Gloggur

## GitHub Publish Workflow (`workflow_dispatch`)

Use **Actions -> Publish to PyPI -> Run workflow**.

- `version` input is optional.
- Leave `version` blank to auto-bump patch from active repo version (`x.y.z` -> `x.y.(z+1)`).
- Provide `version` to override manually (`x.y.z` only); override must be strictly greater than current active repo version.

At run start, the workflow prints and summarizes:

- active repo version
- resolved publish version
- resolution mode (`auto_patch`, `manual_override`, or `release_tag`)

The workflow fails early if `pyproject.toml` and `src/gloggur/__init__.py` versions are out of sync.

## Manual CLI Publish

```bash
python -m pip install -U build twine
python -m build
python -m twine upload dist/*
```
