from __future__ import annotations

import pytest

from scripts.validation.fixtures import FIXTURE_REGISTRY, FixtureFile, TestFixtures


def test_validate_fixture_files_rejects_malformed_sources() -> None:
    """Ensure fixture validation rejects malformed source files."""
    with TestFixtures() as fixtures:
        files = {
            "bad.py": FixtureFile("def oops(:\n    pass\n"),
            "bad.js": FixtureFile("export function broken( { return 1; }\n"),
            "bad.ts": FixtureFile("export interface User { name: ; }\n"),
        }
        with pytest.raises(ValueError) as excinfo:
            fixtures._validate_fixture_files(files)

    message = str(excinfo.value)
    assert "Fixture validation failed" in message
    assert "bad.py" in message
    assert "bad.js" in message
    assert "bad.ts" in message


def test_fixture_registry_compose_duplicate_paths_raises() -> None:
    """Ensure composing duplicate fixtures raises an error."""
    registry = FIXTURE_REGISTRY
    with pytest.raises(ValueError, match="Fixture composition conflict"):
        registry.compose(
            name="dup",
            fixtures=["python_basic", "python_basic"],
            description="Duplicate file path",
        )


def test_fixture_registry_compose_success() -> None:
    """Ensure fixture composition succeeds for different fixtures."""
    registry = FIXTURE_REGISTRY
    composed = registry.compose(
        name="combined",
        fixtures=["python_basic", "javascript_basic"],
        description="Combined fixture",
        tags=("combined",),
    )

    assert "sample.py" in composed.files
    assert "src/index.js" in composed.files
    assert composed.tags == ("combined",)


def test_fixture_registry_list_and_get_with_tags() -> None:
    """Ensure fixture registry list/get behave as expected."""
    registry = FIXTURE_REGISTRY
    templates = registry.list()
    template_names = {template.name for template in templates}

    assert "python_basic" in template_names
    assert "javascript_basic" in template_names
    assert registry.get("python_basic").name == "python_basic"

    filtered = registry.list(tags=["python", "single-file"])
    filtered_names = {template.name for template in filtered}
    assert filtered_names == {"python_basic"}
