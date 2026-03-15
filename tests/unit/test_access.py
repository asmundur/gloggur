from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

import gloggur.access as access_module


def test_build_access_plan_is_ready_when_repo_and_gloggur_paths_are_writable(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf8")
    (repo / ".gloggur").mkdir(mode=0o700)
    (repo / ".gloggur-cache").mkdir(mode=0o700)

    plan = access_module.build_access_plan(repo)

    assert plan.access_ready is True
    assert plan.blocked_paths == []
    assert all(not bool(action["needed"]) for action in plan.automatic_actions)


def test_apply_access_grant_creates_missing_gloggur_dirs_and_persists_state(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf8")

    plan = access_module.build_access_plan(repo)
    result = access_module.apply_access_grant(plan)

    assert result.access_ready is True
    assert (repo / ".gloggur").is_dir()
    assert (repo / ".gloggur-cache").is_dir()
    state_path = repo / ".gloggur" / "access_grants.json"
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf8"))
    assert payload["schema_version"] == "1"
    assert payload["access_ready"] is True


def test_build_access_plan_reports_blocked_repo_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    blocked = repo / "blocked.py"
    blocked.write_text("def blocked():\n    return True\n", encoding="utf8")

    original_open = open

    def _fake_open(path: object, mode: str = "r", *args: object, **kwargs: object):
        if Path(path) == blocked and "rb" in mode:
            raise PermissionError(errno.EACCES, "permission denied")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(access_module, "open", _fake_open, raising=False)

    plan = access_module.build_access_plan(repo)

    assert plan.access_ready is False
    assert [item.path for item in plan.blocked_paths] == ["blocked.py"]
    assert plan.blocked_paths[0].reason == "permission_denied"
    assert plan.manual_action_required is True


def test_build_access_plan_reports_non_owned_gloggur_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gloggur").mkdir(mode=0o700)
    (repo / ".gloggur-cache").mkdir(mode=0o700)

    monkeypatch.setattr(access_module, "_current_uid", lambda: 1000)

    def _fake_owner_uid(path: Path, path_stat: object) -> int | None:
        if path == repo / ".gloggur-cache":
            return 2000
        _ = path, path_stat
        return 1000

    monkeypatch.setattr(access_module, "_path_owner_uid", _fake_owner_uid)

    plan = access_module.build_access_plan(repo)

    assert plan.access_ready is False
    assert [item.path for item in plan.blocked_paths] == [".gloggur-cache"]
    assert plan.blocked_paths[0].reason == "not_owned_by_current_user"


def test_build_access_plan_marks_macos_privacy_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    blocked = repo / "protected.py"
    blocked.write_text("def blocked():\n    return True\n", encoding="utf8")

    original_open = open

    def _fake_open(path: object, mode: str = "r", *args: object, **kwargs: object):
        if Path(path) == blocked and "rb" in mode:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(access_module, "open", _fake_open, raising=False)

    plan = access_module.build_access_plan(repo, platform="darwin")

    assert plan.blocked_paths[0].privacy_blocker is True
    assert plan.manual_os_action_required is True
    assert any("System Settings > Privacy & Security" in item for item in plan.manual_actions)


def test_build_access_plan_keeps_non_macos_permission_blockers_generic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    blocked = repo / "protected.py"
    blocked.write_text("def blocked():\n    return True\n", encoding="utf8")

    original_open = open

    def _fake_open(path: object, mode: str = "r", *args: object, **kwargs: object):
        if Path(path) == blocked and "rb" in mode:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(access_module, "open", _fake_open, raising=False)

    plan = access_module.build_access_plan(repo, platform="linux")

    assert plan.blocked_paths[0].privacy_blocker is False
    assert plan.manual_os_action_required is False
    assert all("System Settings > Privacy & Security" not in item for item in plan.manual_actions)
