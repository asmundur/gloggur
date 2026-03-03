from __future__ import annotations

from pathlib import Path

import yaml


def _load_publish_workflow() -> dict[str, object]:
    workflow_path = Path(".github/workflows/publish.yml")
    payload = yaml.safe_load(workflow_path.read_text(encoding="utf8"))
    assert isinstance(payload, dict)
    return payload


def _workflow_on_block(payload: dict[str, object]) -> dict[str, object]:
    on_block = payload.get("on")
    if not isinstance(on_block, dict):
        # PyYAML can parse bare "on" as boolean true.
        on_block = payload.get(True)
    assert isinstance(on_block, dict)
    return on_block


def _publish_job() -> dict[str, object]:
    payload = _load_publish_workflow()
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    publish = jobs.get("publish")
    assert isinstance(publish, dict)
    return publish


def _publish_step(name: str) -> dict[str, object]:
    publish = _publish_job()
    steps = publish.get("steps")
    assert isinstance(steps, list)
    step = next(
        (
            item
            for item in steps
            if isinstance(item, dict) and item.get("name") == name
        ),
        None,
    )
    assert isinstance(step, dict), f"missing step: {name}"
    return step


def test_publish_workflow_dispatch_version_input_is_optional_and_documented() -> None:
    """Dispatch version input should be optional and describe auto-patch behavior."""
    payload = _load_publish_workflow()
    on_block = _workflow_on_block(payload)
    workflow_dispatch = on_block.get("workflow_dispatch")
    assert isinstance(workflow_dispatch, dict)
    inputs = workflow_dispatch.get("inputs")
    assert isinstance(inputs, dict)
    version_input = inputs.get("version")
    assert isinstance(version_input, dict)
    assert version_input.get("required") is False
    assert version_input.get("type") == "string"
    description = version_input.get("description")
    assert isinstance(description, str)
    normalized = description.lower()
    assert "leave blank" in normalized
    assert "auto-bump patch" in normalized


def test_publish_workflow_resolve_step_emits_expected_outputs_and_modes() -> None:
    """Resolve step should emit active/resolved/mode outputs and all resolution modes."""
    resolve_step = _publish_step("Resolve package version")
    run_script = resolve_step.get("run")
    assert isinstance(run_script, str)
    assert "active_version=" in run_script
    assert "version=" in run_script
    assert "resolution_mode=" in run_script
    assert "release_tag" in run_script
    assert "manual_override" in run_script
    assert "auto_patch" in run_script


def test_publish_workflow_resolve_step_enforces_strict_manual_override_order() -> None:
    """workflow_dispatch override should fail when version is not strictly greater."""
    resolve_step = _publish_step("Resolve package version")
    run_script = resolve_step.get("run")
    assert isinstance(run_script, str)
    assert "must be strictly greater than" in run_script
    assert "active repository version" in run_script


def test_publish_workflow_includes_resolution_summary_step() -> None:
    """Publish workflow should log and summarize active/resolved version context."""
    summary_step = _publish_step("Show publish version resolution")
    run_script = summary_step.get("run")
    assert isinstance(run_script, str)
    assert "GITHUB_STEP_SUMMARY" in run_script
    assert "steps.resolve_version.outputs.active_version" in run_script
    assert "steps.resolve_version.outputs.version" in run_script
    assert "steps.resolve_version.outputs.resolution_mode" in run_script
