# Phase 1: Smoke Tests

## Summary
- Total: 5
- Passed: 0
- Failed: 5

## Test 1.1: Basic Indexing - FAIL

- Index command failed: Command failed (exit 1): /opt/homebrew/opt/python@3.13/bin/python3.13 -m gloggur.cli.main index . --json
stderr: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/auzi/vinnustofa/gloggur/gloggur/cli/main.py", line 7, in <module>
    import click
ModuleNotFoundError: No module named 'click'

## Test 1.2: Incremental Indexing - FAIL

- Index command failed: Command failed (exit 1): /opt/homebrew/opt/python@3.13/bin/python3.13 -m gloggur.cli.main index . --json
stderr: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/auzi/vinnustofa/gloggur/gloggur/cli/main.py", line 7, in <module>
    import click
ModuleNotFoundError: No module named 'click'

## Test 1.3: Search Functionality - FAIL

- Search command failed: Command failed (exit 1): /opt/homebrew/opt/python@3.13/bin/python3.13 -m gloggur.cli.main search index repository --json --top-k 5
stderr: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/auzi/vinnustofa/gloggur/gloggur/cli/main.py", line 7, in <module>
    import click
ModuleNotFoundError: No module named 'click'

## Test 1.4: Docstring Audit - FAIL

- Inspect command failed: Command failed (exit 1): /opt/homebrew/opt/python@3.13/bin/python3.13 -m gloggur.cli.main inspect . --json
stderr: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/auzi/vinnustofa/gloggur/gloggur/cli/main.py", line 7, in <module>
    import click
ModuleNotFoundError: No module named 'click'

## Test 1.5: Status & Cache Management - FAIL

- Status/cache command failed: Command failed (exit 1): /opt/homebrew/opt/python@3.13/bin/python3.13 -m gloggur.cli.main status --json
stderr: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/auzi/vinnustofa/gloggur/gloggur/cli/main.py", line 7, in <module>
    import click
ModuleNotFoundError: No module named 'click'
