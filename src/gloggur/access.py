from __future__ import annotations

import errno
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from gloggur.config import GloggurConfig
from gloggur.io_failures import wrap_io_error
from gloggur.path_filters import filter_index_walk_dirs, is_indexable_source_path

ACCESS_GRANT_STATE_PATH = ".gloggur/access_grants.json"
_ACCESS_STATE_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class AccessBlockedPath:
    path: str
    required_access: list[str]
    reason: str
    detail: str
    privacy_blocker: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "required_access": list(self.required_access),
            "reason": self.reason,
            "detail": self.detail,
            "privacy_blocker": self.privacy_blocker,
        }


@dataclass(frozen=True)
class AccessPlan:
    repo_root: str
    grantee: str
    required_access: list[dict[str, object]]
    automatic_actions: list[dict[str, object]]
    blocked_paths: list[AccessBlockedPath]
    manual_actions: list[str]
    platform: str
    access_ready: bool
    manual_action_required: bool
    manual_os_action_required: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "grantee": self.grantee,
            "required_access": [dict(item) for item in self.required_access],
            "automatic_actions": [dict(item) for item in self.automatic_actions],
            "blocked_paths": [item.to_payload() for item in self.blocked_paths],
            "manual_actions": list(self.manual_actions),
            "platform": self.platform,
            "access_ready": self.access_ready,
            "manual_action_required": self.manual_action_required,
            "manual_os_action_required": self.manual_os_action_required,
        }

    def has_pending_automatic_actions(self) -> bool:
        return any(bool(action.get("needed")) for action in self.automatic_actions)


@dataclass(frozen=True)
class AccessGrantResult:
    repo_root: str
    grantee: str
    platform: str
    applied_at: str
    applied_actions: list[dict[str, object]]
    blocked_paths: list[AccessBlockedPath]
    manual_actions: list[str]
    access_ready: bool
    manual_action_required: bool
    manual_os_action_required: bool
    state_file: str

    def to_payload(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "grantee": self.grantee,
            "platform": self.platform,
            "applied_at": self.applied_at,
            "applied_actions": [dict(item) for item in self.applied_actions],
            "blocked_paths": [item.to_payload() for item in self.blocked_paths],
            "manual_actions": list(self.manual_actions),
            "access_ready": self.access_ready,
            "manual_action_required": self.manual_action_required,
            "manual_os_action_required": self.manual_os_action_required,
            "state_file": self.state_file,
        }


def build_access_plan(repo_root: Path, *, platform: str | None = None) -> AccessPlan:
    resolved_root = repo_root.resolve()
    platform_name = platform or sys.platform
    repo_blockers = _collect_repo_blockers(resolved_root, platform=platform_name)
    gloggur_dir_actions, gloggur_dir_blockers = _assess_gloggur_root(
        resolved_root / ".gloggur",
        repo_root=resolved_root,
    )
    cache_dir_actions, cache_dir_blockers = _assess_gloggur_root(
        resolved_root / ".gloggur-cache",
        repo_root=resolved_root,
    )
    blocked_paths = sorted(
        [*repo_blockers, *gloggur_dir_blockers, *cache_dir_blockers],
        key=lambda item: item.path,
    )
    manual_actions = _manual_actions_for_blockers(
        repo_root=resolved_root,
        blocked_paths=blocked_paths,
    )
    automatic_actions = [*gloggur_dir_actions, *cache_dir_actions]
    manual_os_action_required = any(item.privacy_blocker for item in blocked_paths)
    manual_action_required = bool(blocked_paths)
    access_ready = not manual_action_required and not any(
        bool(action.get("needed")) for action in automatic_actions
    )
    return AccessPlan(
        repo_root=str(resolved_root),
        grantee="current_user",
        required_access=_required_access_payload(),
        automatic_actions=automatic_actions,
        blocked_paths=blocked_paths,
        manual_actions=manual_actions,
        platform=platform_name,
        access_ready=access_ready,
        manual_action_required=manual_action_required,
        manual_os_action_required=manual_os_action_required,
    )


def apply_access_grant(plan: AccessPlan) -> AccessGrantResult:
    repo_root = Path(plan.repo_root)
    applied_at = _utc_now_iso()
    applied_actions: list[dict[str, object]] = []
    blocked_paths: list[AccessBlockedPath] = []

    for relative_path in (".gloggur", ".gloggur-cache"):
        root_path = repo_root / relative_path
        result = _apply_gloggur_root_access(root_path, repo_root=repo_root)
        applied_actions.extend(result["applied_actions"])
        blocked_paths.extend(result["blocked_paths"])

    repo_blockers = _collect_repo_blockers(repo_root, platform=plan.platform)
    blocked_paths.extend(repo_blockers)
    blocked_paths = sorted(blocked_paths, key=lambda item: item.path)
    manual_actions = _manual_actions_for_blockers(repo_root=repo_root, blocked_paths=blocked_paths)
    manual_os_action_required = any(item.privacy_blocker for item in blocked_paths)
    manual_action_required = bool(blocked_paths)
    state_file = repo_root / ACCESS_GRANT_STATE_PATH
    result = AccessGrantResult(
        repo_root=str(repo_root),
        grantee=plan.grantee,
        platform=plan.platform,
        applied_at=applied_at,
        applied_actions=applied_actions,
        blocked_paths=blocked_paths,
        manual_actions=manual_actions,
        access_ready=not blocked_paths,
        manual_action_required=manual_action_required,
        manual_os_action_required=manual_os_action_required,
        state_file=str(state_file),
    )
    _write_access_grant_result(state_file, result)
    return result


def _required_access_payload() -> list[dict[str, object]]:
    return [
        {
            "path": ".",
            "permissions": ["read", "traverse"],
            "scope": "repo_root_and_indexable_paths",
        },
        {
            "path": ".gloggur",
            "permissions": ["read", "write", "create", "delete", "traverse"],
            "scope": "gloggur_runtime_state",
        },
        {
            "path": ".gloggur-cache",
            "permissions": ["read", "write", "create", "delete", "traverse"],
            "scope": "gloggur_cache_state",
        },
    ]


def _access_check_config() -> GloggurConfig:
    config = GloggurConfig()
    config.excluded_dirs = [*config.excluded_dirs, ".gloggur"]
    return config


def _collect_repo_blockers(repo_root: Path, *, platform: str) -> list[AccessBlockedPath]:
    config = _access_check_config()
    blocked_paths: list[AccessBlockedPath] = []

    def _walk(current_root: Path) -> None:
        try:
            entries = sorted(os.scandir(current_root), key=lambda entry: entry.name)
        except OSError as exc:
            blocked_paths.append(
                _blocked_path(
                    repo_root=repo_root,
                    path=current_root,
                    required_access=["read", "traverse"],
                    exc=exc,
                    platform=platform,
                )
            )
            return

        dir_names: list[str] = []
        file_paths: list[Path] = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    dir_names.append(entry.name)
                    continue
            except OSError as exc:
                blocked_paths.append(
                    _blocked_path(
                        repo_root=repo_root,
                        path=Path(entry.path),
                        required_access=["traverse"],
                        exc=exc,
                        platform=platform,
                    )
                )
                continue
            file_paths.append(Path(entry.path))

        kept_dirs = filter_index_walk_dirs(
            str(current_root),
            dir_names,
            excluded_dirs=config.excluded_dirs,
        )
        for directory_name in kept_dirs:
            _walk(current_root / directory_name)

        for file_path in sorted(file_paths):
            if not is_indexable_source_path(
                str(file_path),
                supported_extensions=config.supported_extensions,
                excluded_dirs=config.excluded_dirs,
                include_minified_js=config.include_minified_js,
            ):
                continue
            try:
                with open(file_path, "rb") as handle:
                    handle.read(1)
            except OSError as exc:
                blocked_paths.append(
                    _blocked_path(
                        repo_root=repo_root,
                        path=file_path,
                        required_access=["read"],
                        exc=exc,
                        platform=platform,
                    )
                )

    _walk(repo_root)
    return blocked_paths


def _assess_gloggur_root(
    root: Path,
    *,
    repo_root: Path,
) -> tuple[list[dict[str, object]], list[AccessBlockedPath]]:
    relative_path = _repo_relative_display(repo_root, root)
    actions = [
        {
            "path": relative_path,
            "action": "create_directory_if_missing",
            "mode": "0700",
            "needed": not root.exists(),
        }
    ]
    blocked_paths: list[AccessBlockedPath] = []
    ensure_needed = False

    if not root.exists():
        actions.append(
            {
                "path": relative_path,
                "action": "ensure_user_rwX_existing",
                "needed": False,
            }
        )
        return actions, blocked_paths

    if root.is_symlink():
        writable = os.access(root, os.R_OK | os.W_OK | os.X_OK)
        actions.append(
            {
                "path": relative_path,
                "action": "ensure_user_rwX_existing",
                "needed": False,
            }
        )
        if writable:
            return actions, blocked_paths
        blocked_paths.append(
            AccessBlockedPath(
                path=relative_path,
                required_access=["read", "write", "create", "delete", "traverse"],
                reason="symlink_target_not_auto_repaired",
                detail="Symlinked Glöggur state paths are not rewritten automatically.",
                privacy_blocker=False,
            )
        )
        return actions, blocked_paths

    queue = [root]
    current_uid = _current_uid()
    while queue:
        current = queue.pop(0)
        try:
            current_stat = current.lstat()
        except OSError as exc:
            blocked_paths.append(
                AccessBlockedPath(
                    path=_repo_relative_display(repo_root, current),
                    required_access=["read", "write", "create", "delete", "traverse"],
                    reason="permission_probe_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    privacy_blocker=False,
                )
            )
            continue
        if stat.S_ISLNK(current_stat.st_mode):
            continue
        if not _is_owned_by_current_user(current, current_stat, current_uid=current_uid):
            blocked_paths.append(
                AccessBlockedPath(
                    path=_repo_relative_display(repo_root, current),
                    required_access=["read", "write", "create", "delete", "traverse"],
                    reason="not_owned_by_current_user",
                    detail="Glöggur only auto-repairs paths owned by the invoking user.",
                    privacy_blocker=False,
                )
            )
            continue
        if _path_needs_user_access_fix(current_stat.st_mode):
            ensure_needed = True
        if not stat.S_ISDIR(current_stat.st_mode):
            continue
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError as exc:
            blocked_paths.append(
                AccessBlockedPath(
                    path=_repo_relative_display(repo_root, current),
                    required_access=["read", "write", "create", "delete", "traverse"],
                    reason="permission_probe_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    privacy_blocker=False,
                )
            )
            continue
        for entry in entries:
            queue.append(Path(entry.path))

    actions.append(
        {
            "path": relative_path,
            "action": "ensure_user_rwX_existing",
            "needed": ensure_needed,
        }
    )
    return actions, blocked_paths


def _apply_gloggur_root_access(
    root: Path,
    *,
    repo_root: Path,
) -> dict[str, object]:
    applied_actions: list[dict[str, object]] = []
    blocked_paths: list[AccessBlockedPath] = []
    relative_path = _repo_relative_display(repo_root, root)

    if not root.exists():
        try:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            blocked_paths.append(
                AccessBlockedPath(
                    path=relative_path,
                    required_access=["read", "write", "create", "delete", "traverse"],
                    reason="create_directory_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    privacy_blocker=False,
                )
            )
            return {
                "applied_actions": applied_actions,
                "blocked_paths": blocked_paths,
            }
        applied_actions.append(
            {
                "path": relative_path,
                "action": "create_directory",
                "mode": "0700",
            }
        )

    if root.is_symlink():
        if not os.access(root, os.R_OK | os.W_OK | os.X_OK):
            blocked_paths.append(
                AccessBlockedPath(
                    path=relative_path,
                    required_access=["read", "write", "create", "delete", "traverse"],
                    reason="symlink_target_not_auto_repaired",
                    detail="Symlinked Glöggur state paths are not rewritten automatically.",
                    privacy_blocker=False,
                )
            )
        return {
            "applied_actions": applied_actions,
            "blocked_paths": blocked_paths,
        }

    updated_count = 0
    queue = [root]
    current_uid = _current_uid()
    while queue:
        current = queue.pop(0)
        try:
            current_stat = current.lstat()
        except OSError as exc:
            blocked_paths.append(
                AccessBlockedPath(
                    path=_repo_relative_display(repo_root, current),
                    required_access=["read", "write", "create", "delete", "traverse"],
                    reason="permission_probe_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    privacy_blocker=False,
                )
            )
            continue
        if stat.S_ISLNK(current_stat.st_mode):
            continue
        if not _is_owned_by_current_user(current, current_stat, current_uid=current_uid):
            blocked_paths.append(
                AccessBlockedPath(
                    path=_repo_relative_display(repo_root, current),
                    required_access=["read", "write", "create", "delete", "traverse"],
                    reason="not_owned_by_current_user",
                    detail="Glöggur only auto-repairs paths owned by the invoking user.",
                    privacy_blocker=False,
                )
            )
            continue

        desired_mode = _desired_user_mode(current_stat.st_mode)
        if desired_mode != current_stat.st_mode:
            try:
                os.chmod(current, desired_mode)
            except OSError as exc:
                blocked_paths.append(
                    AccessBlockedPath(
                        path=_repo_relative_display(repo_root, current),
                        required_access=["read", "write", "create", "delete", "traverse"],
                        reason="chmod_failed",
                        detail=f"{type(exc).__name__}: {exc}",
                        privacy_blocker=False,
                    )
                )
            else:
                updated_count += 1

        if not stat.S_ISDIR(current_stat.st_mode):
            continue
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError as exc:
            blocked_paths.append(
                AccessBlockedPath(
                    path=_repo_relative_display(repo_root, current),
                    required_access=["read", "write", "create", "delete", "traverse"],
                    reason="permission_probe_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    privacy_blocker=False,
                )
            )
            continue
        for entry in entries:
            queue.append(Path(entry.path))

    if updated_count > 0:
        applied_actions.append(
            {
                "path": relative_path,
                "action": "ensure_user_rwX",
                "updated_path_count": updated_count,
            }
        )
    return {
        "applied_actions": applied_actions,
        "blocked_paths": blocked_paths,
    }


def _write_access_grant_result(path: Path, result: AccessGrantResult) -> None:
    payload = {
        "schema_version": _ACCESS_STATE_SCHEMA_VERSION,
        **result.to_payload(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    except OSError as exc:
        raise wrap_io_error(
            exc,
            operation="write access grant state",
            path=str(path),
        ) from exc


def _manual_actions_for_blockers(
    *,
    repo_root: Path,
    blocked_paths: list[AccessBlockedPath],
) -> list[str]:
    if not blocked_paths:
        return []
    if any(item.privacy_blocker for item in blocked_paths):
        return [
            "Grant the terminal or host app that runs gloggur access to the "
            "protected repo paths in System Settings > Privacy & Security.",
            "Re-run `gloggur access grant "
            f"{repo_root} --yes` after macOS privacy access is granted.",
        ]
    return [
        "Make the blocked repo paths readable/traversable for the current user, "
        "or repair the listed Glöggur-owned paths manually.",
        f"Re-run `gloggur access grant {repo_root} --yes` after the path permissions are fixed.",
    ]


def _blocked_path(
    *,
    repo_root: Path,
    path: Path,
    required_access: list[str],
    exc: OSError,
    platform: str,
) -> AccessBlockedPath:
    privacy_blocker = _is_privacy_blocker(exc, platform=platform)
    reason = "privacy_blocker" if privacy_blocker else _blocked_reason(exc)
    return AccessBlockedPath(
        path=_repo_relative_display(repo_root, path),
        required_access=required_access,
        reason=reason,
        detail=f"{type(exc).__name__}: {exc}",
        privacy_blocker=privacy_blocker,
    )


def _blocked_reason(exc: OSError) -> str:
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if exc.errno == errno.ENOENT:
        return "path_missing"
    return "access_probe_failed"


def _is_privacy_blocker(exc: OSError, *, platform: str) -> bool:
    if platform != "darwin":
        return False
    if exc.errno == errno.EPERM:
        return True
    detail = str(exc).lower()
    return "operation not permitted" in detail


def _path_needs_user_access_fix(mode: int) -> bool:
    return _desired_user_mode(mode) != mode


def _desired_user_mode(mode: int) -> int:
    desired = mode | stat.S_IRUSR | stat.S_IWUSR
    if stat.S_ISDIR(mode) or mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        desired |= stat.S_IXUSR
    return desired


def _is_owned_by_current_user(
    path: Path,
    path_stat: os.stat_result,
    *,
    current_uid: int | None,
) -> bool:
    if current_uid is None:
        return True
    return _path_owner_uid(path, path_stat) == current_uid


def _path_owner_uid(path: Path, path_stat: os.stat_result) -> int | None:
    _ = path
    return getattr(path_stat, "st_uid", None)


def _current_uid() -> int | None:
    getuid = getattr(os, "getuid", None)
    if not callable(getuid):
        return None
    return int(getuid())


def _repo_relative_display(repo_root: Path, path: Path) -> str:
    try:
        relative = os.path.relpath(path, repo_root)
    except ValueError:
        return str(path)
    normalized = relative.replace(os.sep, "/")
    return "." if normalized == "." else normalized


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
