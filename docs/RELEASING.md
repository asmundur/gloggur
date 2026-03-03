# Releasing Gloggur

## GitHub Publish Workflow (`workflow_dispatch`)

Use **Actions -> Publish to PyPI -> Run workflow**.

- `version` input is optional.
- Leave `version` blank to auto-bump patch from the highest known stable version (`x.y.z` -> `x.y.(z+1)`).
- Highest known version is computed from:
  - active repo version
  - semantic git tags in the repository
  - published PyPI releases for the package
- Provide `version` to override manually (`x.y.z` only); override must be strictly greater than the highest known version.

At run start, the workflow prints and summarizes:

- active repo version
- highest known version
- resolved publish version
- resolution mode (`auto_patch`, `manual_override`, or `release_tag`)

The workflow fails early if `pyproject.toml` and `src/gloggur/__init__.py` versions are out of sync.

## Manual CLI Publish

```bash
python -m pip install -U build twine
python -m build
python -m twine upload dist/*
```
